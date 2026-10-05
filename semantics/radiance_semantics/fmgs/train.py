"""Train the FMGS feature field against the frozen Gaussians of a splatfacto run.

Each step: one training view (refined pose, as the lift uses) → the trainable Gaussians in its
frustum → field(centre) per Gaussian → render the features with gsplat (32-channel chunks; the
backward kernel's limit), divide by the rendered alpha (a blend-weighted average, so dropping the
untrained Gaussians does not darken the map) → losses against the view's teacher maps upsampled
to the render size, on pixels with alpha ≥ alpha_min.

Trainable subset (FMGS): the most opaque `subset_frac` of the Gaussians, picked once; per view only
those whose centre projects into the image (10 % margin). The field still answers every centre at
bake time, so the subset only limits training cost.

Variants:
  faithful  render the 512 + 384 head outputs per Gaussian (FMGS; 28 chunks per step)
  blite     render the 192-d hash encoding (6 chunks) and apply the heads per pixel — faster and
            lighter, but NOT FMGS: results must label it as a variant
  auto      faithful, falling back on CUDA out-of-memory through the ladder below

Out-of-memory ladder (auto; docs/SEMANTICS_PLAN.md, after the 480×270 default): 1. hash table
2^19 · 2. half of the visible subset per step · 3. B-lite. A fallback restarts training from step 0
and is recorded in the stats.

Writes into `out`: ckpt/step-NNNNNN.pt every `ckpt_every` steps (resumable), tb/ (TensorBoard),
field.pt (final field), train.json (settings, fallback, losses, time, peak VRAM, Gaussian checksum).
The Gaussians are detached tensors that never reach the optimizer; their checksum before and after
training is recorded anyway as the gate's proof that the splat was not modified.
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field as dc_field, replace
from pathlib import Path

import numpy as np
import torch

from ..lift import render_size, upsample, view_K
from ..render import Gaussians, render_features, viewmat_from_c2w
from .field import FeatureField, FieldConfig, scene_box
from .losses import total_loss

LADDER = [("default", {}), ("hash table 2^19", {"log2_table": 19}),
          ("half the visible Gaussians per step", {"log2_table": 19, "subsample": 0.5}),
          ("B-lite (render the encoding, heads per pixel)", {"log2_table": 19, "subsample": 0.5, "variant": "blite"})]


@dataclass
class TrainConfig:
    steps: int = 4200
    feat_width: int = 480
    variant: str = "auto"          # auto | faithful | blite
    impl: str = "tcnn"             # tcnn | torch
    log2_table: int = 20
    subset_frac: float = 0.4
    subsample: float = 1.0
    lr: float = 1e-2               # LERF's hash-field settings: Adam 1e-2 → 1e-3 exponential, eps 1e-15
    lr_final: float = 1e-3
    eps: float = 1e-15
    w_clip: float = 0.2
    w_dino: float = 0.8
    w_pa: float = 0.01
    delta: float = 1.25
    alpha_min: float = 0.5
    pa_samples: int = 4096
    ckpt_every: int = 1000
    log_every: int = 50
    seed: int = 0
    field_cfg: dict = dc_field(default_factory=dict)   # FieldConfig overrides (tests: a small field)

    def field_config(self):
        return FieldConfig(**{**self.field_cfg, "log2_table": self.log2_table})

    def key(self):
        """What a checkpoint must match to be resumed."""
        d = asdict(self)
        for k in ("steps", "ckpt_every", "log_every"):
            d.pop(k)
        return d


def gauss_checksum(g):
    h = hashlib.sha256()
    for t in (g.means, g.quats, g.scales, g.opacities):
        h.update(t.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def trainable_subset(g, frac):
    """Indices of the most opaque `frac` of the Gaussians (all when frac ≥ 1)."""
    n = len(g)
    if frac >= 1:
        return torch.arange(n, device=g.means.device)
    k = max(1, int(round(n * frac)))
    return torch.topk(g.opacities, k).indices.sort().values


def in_view(means, vm, K, W, H, margin=0.1, near=0.05):
    """Mask of centres in front of the camera that project into the image (± margin)."""
    p = means @ vm[:3, :3].T + vm[:3, 3]
    z = p[:, 2]
    zc = z.clamp_min(1e-6)
    u = K[0, 0] * p[:, 0] / zc + K[0, 2]
    v = K[1, 1] * p[:, 1] / zc + K[1, 2]
    return (z > near) & (u > -margin * W) & (u < (1 + margin) * W) & (v > -margin * H) & (v < (1 + margin) * H)


class TeacherCache:
    """Teacher grids per view stem, held on the CPU in float16 and moved per step."""

    def __init__(self, maps_for):
        self.maps_for, self.d = maps_for, {}

    def get(self, stem):
        if stem not in self.d:
            self.d[stem] = {k: torch.from_numpy(np.asarray(v, dtype=np.float16)) for k, v in self.maps_for(stem).items()}
        return self.d[stem]


def _step(field, cfg, g, sub, v, teachers, render_backend, gen, dev):
    W, H = render_size(v, cfg.feat_width)
    vm, K = viewmat_from_c2w(v.c2w.to(dev).float()), view_K(v, W, H).to(dev)
    sel = sub[in_view(g.means[sub], vm, K, W, H)]
    if cfg.subsample < 1 and len(sel):
        keep = torch.randperm(len(sel), generator=gen, device="cpu")[:max(1, int(len(sel) * cfg.subsample))]
        sel = sel[keep.to(sel.device)]
    if len(sel) == 0:
        return None
    gs = Gaussians(g.means[sel], g.quats[sel], g.scales[sel], g.opacities[sel])
    maps = teachers.get(v.stem)
    tgt = {k: upsample(t.to(dev), W, H, dev) for k, t in maps.items()}
    has_dino = "dino" in tgt
    if cfg.variant == "blite":
        enc = field.encode(gs.means)
        img, alpha = render_features(gs, enc, vm, K, W, H, render_backend)
        mask = alpha[..., 0] >= cfg.alpha_min
        e = img / alpha.clamp_min(1e-6)
        out = field.heads(e[mask])
        clip = torch.zeros(H, W, cfg_dim(field, "clip"), device=dev)
        clip[mask] = out["clip"]
        dino = None
        if has_dino:
            dino = torch.zeros(H, W, cfg_dim(field, "dino"), device=dev)
            dino[mask] = out["dino"]
    else:
        out = field(gs.means)
        feats = torch.cat([out["clip"], out["dino"]], -1) if has_dino else out["clip"]
        img, alpha = render_features(gs, feats, vm, K, W, H, render_backend)
        mask = alpha[..., 0] >= cfg.alpha_min
        pred = img / alpha.clamp_min(1e-6)
        C = out["clip"].shape[-1]
        clip, dino = pred[..., :C], (pred[..., C:] if has_dino else None)
    losses = total_loss(clip, tgt["clip"], dino, tgt.get("dino"), mask, cfg.w_clip, cfg.w_dino, cfg.w_pa, cfg.delta,
                        cfg.pa_samples, generator=gen)
    losses["n"] = len(sel)
    losses["valid"] = float(mask.float().mean())
    return losses


def cfg_dim(field, which):
    return field.cfg.clip_dim if which == "clip" else field.cfg.dino_dim


def _latest_ckpt(d):
    c = sorted(Path(d).glob("step-*.pt"))
    return c[-1] if c else None


def train(g, views, maps_for, cfg: TrainConfig, out, device=None, render_backend="gsplat", log=print, resume=True):
    """Train; returns the stats dict (also written to out/train.json) and the field."""
    out = Path(out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    dev = torch.device(device) if device else g.means.device
    g = g.to(dev)
    checksum0 = gauss_checksum(g)
    lo, hi = scene_box(g.means)
    teachers = TeacherCache(maps_for)
    views = [v for v in views if teachers.get(v.stem)]          # loads every view's maps once (fp16, CPU)
    if not views:
        raise RuntimeError("no view has teacher maps")
    level = 0
    if cfg.variant == "auto":
        cfg = replace(cfg, variant="faithful")
        auto = True
    else:
        auto = False
    base = cfg
    t_start = time.time()
    while True:
        cfg = replace(base, **LADDER[level][1]) if auto else base
        oom = False
        try:
            stats, field = _train_once(g, views, teachers, cfg, out, dev, render_backend, log, resume, lo, hi)
            break
        except torch.cuda.OutOfMemoryError:
            if not auto or level + 1 >= len(LADDER):
                raise
            oom = True
        if oom:                      # outside the except: the failed step's tensors are released by now
            import gc
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            level += 1
            log(f"  ! CUDA out of memory — fallback {level}: {LADDER[level][0]}; restarting from step 0")
            for p in (out / "ckpt").glob("step-*.pt"):
                p.unlink()
            resume = False
    checksum1 = gauss_checksum(g)
    stats.update(fallback={"level": level, "name": LADDER[level][0]} if auto else None,
                 variant=cfg.variant, config=asdict(cfg), field=asdict(cfg.field_config()),
                 gauss_checksum_before=checksum0, gauss_checksum_after=checksum1,
                 gauss_unchanged=checksum0 == checksum1, wall_seconds=round(time.time() - t_start, 1),
                 views=len(views), gaussians=len(g))
    field.save(out / "field.pt", train=stats)
    (out / "train.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats, field


def _train_once(g, views, teachers, cfg, out, dev, render_backend, log, resume, lo, hi):
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    field = FeatureField(lo, hi, cfg.field_config(), cfg.impl).to(dev)
    opt = torch.optim.Adam(field.parameters(), lr=cfg.lr, eps=cfg.eps)
    gamma = (cfg.lr_final / cfg.lr) ** (1.0 / max(1, cfg.steps))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    start, hist = 0, []
    ck = _latest_ckpt(out / "ckpt") if resume else None
    if ck is not None:
        d = torch.load(ck, map_location="cpu")
        if d.get("key") == cfg.key():
            field.load_state_dict(d["field"]); opt.load_state_dict(d["opt"]); sched.load_state_dict(d["sched"])
            start, hist = d["step"], d.get("hist", [])
            gen.set_state(d["gen"])
            log(f"  resumed from {ck.name} (step {start})")
        else:
            log(f"  {ck.name} was written with other settings — starting over")
    sub = trainable_subset(g, cfg.subset_frac)
    log(f"  field {cfg.impl}: {sum(p.numel() for p in field.parameters()) / 1e6:.1f} M parameters "
        f"(table 2^{cfg.log2_table}); variant {cfg.variant}; {len(sub):,} of {len(g):,} Gaussians trainable; "
        f"{len(views)} views at {cfg.feat_width} px wide")
    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(out / "tb"))
    except Exception as e:                            # tensorboard missing: keep training, say so
        log(f"  (no TensorBoard logging: {e})")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    run_sum = {}
    for step in range(start, cfg.steps):
        v = views[int(torch.randint(len(views), (1,), generator=gen))]
        losses = _step(field, cfg, g, sub, v, teachers, render_backend, gen, dev)
        if losses is None:
            continue
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        opt.step()
        sched.step()
        rec = {k: float(x) for k, x in losses.items() if k in ("loss", "clip", "dino", "pa")}
        hist.append(rec["loss"])
        for k, x in rec.items():
            run_sum[k] = run_sum.get(k, 0.0) + x
        run_sum["_n"] = run_sum.get("_n", 0) + 1
        if tb is not None:
            for k, x in rec.items():
                tb.add_scalar(f"fmgs/{k}", x, step + 1)
            tb.add_scalar("fmgs/lr", sched.get_last_lr()[0], step + 1)
        s1 = step + 1
        if s1 % cfg.log_every == 0 or s1 == cfg.steps:
            m = run_sum.pop("_n")
            avg = {k: x / m for k, x in run_sum.items()}
            run_sum = {}
            rate = (s1 - start) / max(1e-6, time.time() - t0)
            eta = (cfg.steps - s1) / max(rate, 1e-6)
            vram = f" · {torch.cuda.max_memory_allocated(dev) / 2 ** 20:.0f} MiB peak" if dev.type == "cuda" else ""
            log(f"  step {s1}/{cfg.steps}  loss {avg['loss']:.4f} (clip {avg['clip']:.4f}"
                + (f" dino {avg['dino']:.4f}" if "dino" in avg else "") + (f" pa {avg['pa']:.3f}" if "pa" in avg else "")
                + f")  {losses['n']:,} Gaussians  {rate:.2f} it/s  ETA {eta / 60:.0f} min{vram}")
        if s1 % cfg.ckpt_every == 0 or s1 == cfg.steps:
            p = out / "ckpt" / f"step-{s1:06d}.pt"
            torch.save({"field": field.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": s1,
                        "key": cfg.key(), "hist": hist, "gen": gen.get_state()}, p)
            for old in sorted((out / "ckpt").glob("step-*.pt"))[:-2]:
                old.unlink()
    if tb is not None:
        tb.close()
    k = max(1, min(50, len(hist) // 4))
    stats = {"steps": cfg.steps, "resumed_from": start, "seconds": round(time.time() - t0, 1),
             "it_per_s": round((cfg.steps - start) / max(1e-6, time.time() - t0), 2),
             "loss_first": round(float(np.mean(hist[:k])), 5) if hist else None,
             "loss_last": round(float(np.mean(hist[-k:])), 5) if hist else None,
             "peak_vram_mib": int(torch.cuda.max_memory_allocated(dev) / 2 ** 20) if dev.type == "cuda" else None,
             "params_m": round(sum(p.numel() for p in field.parameters()) / 1e6, 2),
             "trainable_gaussians": len(sub)}
    return stats, field


def train_run(run, teacher_dir, cfg, out, device=None, render_backend="gsplat", log=print, limit=None):
    """Load the scene's active run (refined poses), train on every training view with teacher maps."""
    from ..cameras import in_workspace, load_pipeline, train_views
    from ..lift import gaussians_from_model, views_from_pipeline
    from ..teachers import load_maps
    with in_workspace(run):
        _, pipeline, _, step = load_pipeline(run)
        model = pipeline.model
        cams, files, _ = train_views(pipeline)
        views = views_from_pipeline(model, cams, files)
        if limit:
            views = views[:limit]
        g = gaussians_from_model(model)
        del pipeline
    if device:
        g = g.to(device)
    log(f"  {len(g):,} Gaussians, {len(views)} training views (refined poses), teachers {Path(teacher_dir).name}")
    stats, field = train(g, views, lambda stem: load_maps(teacher_dir, stem), cfg, out, device, render_backend, log)
    stats.update(splat_step=int(step))
    (Path(out) / "train.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats, field


def _latest_teachers(run):
    d = run.semantics_dir / "teachers"
    c = [p.parent for p in d.glob("*/meta.json")] if d.is_dir() else []
    if not c:
        raise SystemExit(f"no teacher maps under {d}: run semantic_pipeline.py --scene {run.scene} --only teachers")
    return max(c, key=lambda p: (p / "meta.json").stat().st_mtime)


def main(argv=None):
    from ..paths import find_scene_run
    ap = argparse.ArgumentParser(description="train the FMGS feature field on a frozen splat (Phase 4)")
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True, help="output directory (checkpoints, tb/, field.pt, train.json)")
    ap.add_argument("--teachers-dir", help="teacher maps (default: the newest under semantics/teachers/)")
    ap.add_argument("--steps", type=int, default=TrainConfig.steps)
    ap.add_argument("--feat-width", type=int, default=TrainConfig.feat_width)
    ap.add_argument("--variant", choices=["auto", "faithful", "blite"], default="auto")
    ap.add_argument("--impl", choices=["tcnn", "torch"], default="tcnn")
    ap.add_argument("--log2-table", type=int, default=TrainConfig.log2_table)
    ap.add_argument("--device")
    ap.add_argument("--render-backend", choices=["gsplat", "reference"], default="gsplat")
    ap.add_argument("--limit", type=int, help="first N views (tests)")
    a = ap.parse_args(argv)
    run = find_scene_run(Path(a.project_root), a.scene)
    tdir = Path(a.teachers_dir) if a.teachers_dir else _latest_teachers(run)
    cfg = TrainConfig(steps=a.steps, feat_width=a.feat_width, variant=a.variant, impl=a.impl, log2_table=a.log2_table,
                      ckpt_every=min(TrainConfig.ckpt_every, max(1, a.steps)), log_every=min(50, max(1, a.steps // 10)))
    stats, _ = train_run(run, tdir, cfg, a.out, a.device, a.render_backend, limit=a.limit)
    fell = stats["loss_last"] is not None and stats["loss_first"] is not None and stats["loss_last"] < stats["loss_first"]
    print(json.dumps({k: stats.get(k) for k in ("steps", "seconds", "it_per_s", "loss_first", "loss_last", "peak_vram_mib",
                                                 "fallback", "variant", "gauss_unchanged", "gauss_checksum_after")}))
    print(f"loss {'FELL' if fell else 'DID NOT FALL'}: {stats['loss_first']} → {stats['loss_last']}")
    return 0 if fell and stats["gauss_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
