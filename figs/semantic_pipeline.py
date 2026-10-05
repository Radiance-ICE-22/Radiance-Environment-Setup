#!/usr/bin/env python3
"""
semantic_pipeline.py — per-Gaussian CLIP + DINOv2 features for a trained FiGS splat, resumable.

The sibling of figs_pipeline.py and svnet_pipeline.py. It never retrains or modifies the splat:
features are attached to the frozen Gaussians of the scene's single active splatfacto run.

    source <prefix>/figs_env.sh                                  # REQUIRED (kitchen + caches)
    ./semantic_pipeline.py --scene backroom                      # everything, lift backend
    ./semantic_pipeline.py --scene backroom --backend fmgs       # FMGS hash-grid field (Phase 4)
    ./semantic_pipeline.py --scene backroom --status
    ./semantic_pipeline.py --scene backroom --from lift --feat-width 480
    ./semantic_pipeline.py --scene backroom --only teachers

Steps (docs/SEMANTICS_PLAN.md, Phases 1 and 4; the first three are shared by both backends):
    preflight  environment pins, CUDA, model weights, the scene's active run, time/disk estimate
    cameras    refined vs raw pose render check (radiance_semantics.cameras; must not regress)
    teachers   CLIP pyramid + DINOv2 per frame → gsplats/workspace/<scene>/semantics/teachers/<tag>/
  --backend lift (default):
    lift       blend-weighted average of the teacher maps onto every Gaussian (refined poses)
    export     rows in Galley's .splat order → gsplats/workspace/<scene>/semantics/<run>/lift/
  --backend fmgs:
    fmgs       train the hash-grid feature field against the frozen Gaussians (radiance_semantics.fmgs;
               checkpoints + TensorBoard in semantics/<run>/fmgs_train/, resumable)
    bake       evaluate the field at every Gaussian in .splat order → semantics/<run>/fmgs/

State lives in <PROJECT_ROOT>/.semantic_pipeline_state/<scene>/<run>/ (settings, markers,
results). Each marker's fingerprint includes the previous step's marker and the checkpoint key
(run + checkpoint name + mtime), so a retrain or a changed setting re-runs what depends on it.
Each GPU step runs in its own child process (this script with --in-step), as in svnet_pipeline.py.

Query the result with semantic_query.py.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figs_pipeline import (  # noqa: E402  (shared helpers; figs_pipeline imports only stdlib)
    StepFailed, VramMonitor, _C, diagnose, elapsed, fail, info, ok, resolve_project_root, section, warn)

try:
    import radiance_semantics  # noqa: F401
except ImportError:
    sys.exit("radiance_semantics is not installed in this env: run setup_scripts/install_semantics.sh "
             "--prefix <FiGS prefix>, then `source <prefix>/figs_env.sh`")

DEFAULTS = {"backend": "lift", "teachers": ["clip", "dino"], "feat_width": 960, "dino_width": 896,
            "scales": None, "batch": 256, "device": None, "render_backend": "gsplat", "limit": None,
            "fmgs_steps": 4200, "fmgs_width": 480, "fmgs_variant": "auto", "fmgs_impl": "auto", "fmgs_table": 20}
LISTS = ("teachers",)
NOT_STICKY = ("backend",)        # never taken from config.json: a bare run (Galley's Continue) is the lift


class Ctx:
    def __init__(self, a):
        from radiance_semantics.paths import find_scene_run
        self.a = a
        self.project_root = resolve_project_root(a.project_root).resolve()
        self.run = find_scene_run(self.project_root, a.scene)
        self.state = self.run.state_dir
        self.state.mkdir(parents=True, exist_ok=True)
        self.results = {"scene": a.scene, "run": self.run.run}
        self.reload_results()

    def reload_results(self):
        rp = self.state / "results.json"
        if rp.exists():
            try:
                self.results.update(json.loads(rp.read_text()))
            except ValueError:
                pass

    def save_results(self):
        _atomic(self.state / "results.json", json.dumps(self.results, indent=2, default=str) + "\n")

    def marker(self, name):
        m = self.state / f"{name}.done"
        return m.read_text().strip() if m.exists() else ""

    def fingerprint(self, keys, prev):
        vals = {k: getattr(self.a, k, None) for k in keys}
        vals["_key"] = self.a.key
        vals["_prev"] = self.marker(prev) if prev else None
        return hashlib.sha256(json.dumps(vals, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def is_done(self, name, fp):
        m = self.state / f"{name}.done"
        if not m.exists():
            return False
        if m.read_text().strip().split("\n")[0] != fp:
            warn(f"'{name}' was completed with different settings, an older checkpoint or older inputs — redoing")
            m.unlink()
            return False
        return True

    def mark_done(self, name, fp):
        _atomic(self.state / f"{name}.done", f"{fp}\n{datetime.now().isoformat(timespec='seconds')}\n")

    def clear(self, name):
        (self.state / f"{name}.done").unlink(missing_ok=True)

    # settings → objects
    def teacher_settings(self):
        from radiance_semantics.teachers import TeacherSettings
        s = TeacherSettings(dino_width=int(self.a.dino_width), batch=int(self.a.batch),
                            teachers=tuple(self.a.teachers))
        if self.a.scales:
            s.scales = [round(float(x), 4) for x in self.a.scales]
        return s

    @property
    def teacher_dir(self):
        return self.run.semantics_dir / "teachers" / self.teacher_settings().tag()

    @property
    def backend_dir(self):
        return self.run.backend_dir(self.a.backend)

    @property
    def device(self):
        if self.a.device:
            return self.a.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"


def _atomic(p, text):
    p = Path(p)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(text)
    tmp.replace(p)


def _free_gb(p):
    p = Path(p)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 2 ** 30


# ════════════════════════════════════════════════════════════════════════════════
#  Steps
# ════════════════════════════════════════════════════════════════════════════════

def step_preflight(c):
    from importlib.metadata import version
    from radiance_semantics import env_check
    from radiance_semantics.models import provenance_file
    from radiance_semantics.teachers import estimate_crops, frames_of, grid_bytes
    problems = env_check.problems(env_check.versions())
    for p in problems:
        fail(p)
    try:
        ok(f"open_clip {version('open_clip_torch')}, radiance_semantics {radiance_semantics.__version__}")
    except Exception as e:
        problems.append(f"open_clip_torch not installed: {e}")
    import torch
    if c.device == "cuda":
        if not torch.cuda.is_available():
            problems.append("torch cannot see the GPU")
        else:
            total = torch.cuda.get_device_properties(0).total_memory // 2 ** 20
            ok(f"GPU: {torch.cuda.get_device_name(0)} {total} MiB")
    else:
        warn(f"device {c.device}: teachers and the lift will be very slow (tests only)")
    if not provenance_file().exists() and "clip" in c.a.teachers:
        warn(f"model weights not fetched ({provenance_file()} missing): the teachers step will try to download them")
    run = c.run
    ok(f"scene {c.a.scene}: run {run.run}, {run.checkpoint.name} (key {c.a.key})")
    frames = frames_of(run.transforms_json)
    from PIL import Image
    with Image.open(run.scene_dir / frames[0][1]) as im:
        W, H = im.size
    s = c.teacher_settings()
    crops = estimate_crops(H, W, s)
    t_bytes = len(frames) * grid_bytes(H, W, s)
    have = len(list((c.teacher_dir / "clip").glob("*.npy"))) if (c.teacher_dir / "clip").is_dir() else 0
    ck = torch.load(run.checkpoint, map_location="cpu", weights_only=False)
    st = ck.get("pipeline", ck)
    n = next(v.shape[0] for k, v in st.items() if k.endswith("gauss_params.means") or k.endswith("_model.means"))
    del ck, st
    dims = (512 if "clip" in c.a.teachers else 0) + (384 if "dino" in c.a.teachers else 0)
    out_bytes = n * dims * 2 * 2 + n * 32          # raw lift output + table (≤ n rows) + small files
    need_gb = (t_bytes * (have < len(frames)) + out_bytes) / 2 ** 30
    free = _free_gb(run.semantics_dir)
    info(f"{len(frames)} frames at {W}x{H}; {crops} CLIP crops per frame ({len(s.scales)} scales); "
         f"teacher tag {s.tag()} ({have} frames already cached)")
    info(f"{n:,} Gaussians; needs ≈ {need_gb:.1f} GB (teachers {t_bytes / 2 ** 30:.1f} GB, lift + table "
         f"{out_bytes / 2 ** 30:.1f} GB); {free:.0f} GB free")
    if need_gb > free * 0.9:
        problems.append(f"not enough disk: ≈ {need_gb:.1f} GB needed, {free:.0f} GB free on {run.semantics_dir}")
    c.results["preflight"] = {"frames": len(frames), "image": [W, H], "crops_per_frame": crops,
                              "teacher_tag": s.tag(), "n_gaussians": int(n), "need_gb": round(need_gb, 2),
                              "free_gb": round(free, 1), "key": c.a.key}
    if problems:
        raise StepFailed("\n".join(f"  - {p}" for p in problems))


def step_cameras(c):
    from radiance_semantics import cameras
    rc = cameras.main(["--project-root", str(c.project_root), "--scene", c.a.scene, "--views", "6", "--record"])
    if rc != 0:
        raise StepFailed("refined poses do not render at least as well as transforms.json's — the lift would "
                         "project features through the wrong cameras. See the per-view PSNR above.")
    rec = sorted(c.run.runs_dir.glob(f"semantics_p0_cameras_{c.a.scene}_*.json"))
    if rec:
        r = json.loads(rec[-1].read_text())
        c.results["cameras"] = {k: r.get(k) for k in ("psnr_refined_mean", "psnr_raw_mean", "eval_psnr_mean",
                                                       "corrections")}


def step_teachers(c):
    from radiance_semantics import teachers
    s = c.teacher_settings()
    t0 = time.time()
    with VramMonitor() as vm:
        meta = teachers.extract(c.run.scene_dir, c.run.semantics_dir / "teachers", s, device=c.device,
                                log=info, limit=c.a.limit)
    st = meta["stats"]
    ok(f"teachers {meta['tag']}: {st['done']} frames extracted, {st['skipped']} already cached, "
       f"{st['mb']:.0f} MB, {elapsed(t0)}")
    secs = [r["seconds"] for r in meta["images"].values() if "seconds" in r]
    c.results["teachers"] = {"tag": meta["tag"], **st, "s_per_frame": round(sum(secs) / len(secs), 2) if secs else None,
                             "peak_vram_mib": vm.peak, "dir": str(c.teacher_dir)}


def step_lift(c):
    import numpy as np
    from radiance_semantics.lift import lift_run
    if not (c.teacher_dir / "meta.json").exists():
        raise StepFailed(f"no teacher features at {c.teacher_dir}: run the teachers step")
    with VramMonitor() as vm:
        feats, w, stats, _ = lift_run(c.run, c.teacher_dir, feat_width=int(c.a.feat_width),
                                      backend=c.a.render_backend, log=info, limit=c.a.limit)
    raw = c.backend_dir.with_name(c.backend_dir.name + "_raw")
    raw.mkdir(parents=True, exist_ok=True)
    for name, f in feats.items():
        np.save(raw / f"{name}.npy", f.astype(np.float16))
    np.save(raw / "weight.npy", w.astype(np.float32))
    stats.update(peak_vram_mib=vm.peak, teacher_tag=c.teacher_settings().tag(), key=c.a.key)
    _atomic(raw / "lift.json", json.dumps(stats, indent=2) + "\n")
    ok(f"lifted {', '.join(feats)} onto {stats['seen']:,} of {stats['n']:,} Gaussians "
       f"({stats['seen_frac']:.0%} seen) in {stats['seconds']:.0f} s, {stats['passes']} render passes")
    c.results["lift"] = stats


def _ckpt_gaussians(c):
    """(means, log scales, opacity logits, .splat order) from the active checkpoint."""
    import torch
    from course_tools import splat_order
    ck = torch.load(c.run.checkpoint, map_location="cpu", weights_only=False)
    st = ck.get("pipeline", ck)

    def p(name):
        return next(v for k, v in st.items() if k.endswith(f"gauss_params.{name}") or k.endswith(f"_model.{name}")
                    ).float().numpy()
    means, scales, opac = p("means"), p("scales"), p("opacities").reshape(-1)
    return means, scales, opac, splat_order(means, scales, opac)      # exactly the .splat file's records


def _geom(means, scales, opac, order):
    import numpy as np
    return np.concatenate([means[order], (1 / (1 + np.exp(-opac[order])))[:, None],
                           np.exp(scales[order]).max(1, keepdims=True)], 1).astype(np.float32)


def _file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


def step_export(c):
    import numpy as np
    from radiance_semantics.lift import normalise_rows
    from radiance_semantics.store import write_table
    raw = c.backend_dir.with_name(c.backend_dir.name + "_raw")
    meta = json.loads((raw / "lift.json").read_text()) if (raw / "lift.json").exists() else None
    if meta is None or meta.get("key") != c.a.key:
        raise StepFailed(f"no lift output for this checkpoint in {raw}: run the lift step")
    means, scales, opac, order = _ckpt_gaussians(c)
    clip = normalise_rows(np.load(raw / "clip.npy").astype(np.float32)[order]) if (raw / "clip.npy").exists() else None
    dino = normalise_rows(np.load(raw / "dino.npy").astype(np.float32)[order]) if (raw / "dino.npy").exists() else None
    if clip is None:
        raise StepFailed("the lift produced no CLIP features (teachers without clip?)")
    w = np.load(raw / "weight.npy")[order]
    geom = _geom(means, scales, opac, order)
    ckpt = c.run.checkpoint
    index = write_table(c.backend_dir, clip=clip, dino=dino, weight=w, geom=geom, order=order, meta={
        "scene": c.a.scene, "run": c.run.run, "checkpoint": ckpt.name, "checkpoint_mtime": int(ckpt.stat().st_mtime),
        "key": c.a.key, "backend": c.a.backend, "teacher_tag": meta["teacher_tag"], "n_total": int(len(means)),
        "settings": {k: getattr(c.a, k) for k in DEFAULTS if k not in ("device", "limit")},
        "metrics": {"lift": meta, "seen_rows": int((w > 0).sum())}})
    size = sum(f.stat().st_size for f in c.backend_dir.iterdir()) / 2 ** 20
    ok(f"table: {index['n']:,} rows (.splat order, {index['order_sha']}), {int((w > 0).sum()):,} seen, "
       f"{size:.0f} MB → {c.backend_dir}")
    c.results["export"] = {"rows": index["n"], "seen_rows": int((w > 0).sum()), "mb": round(size, 1),
                           "order_sha": index["order_sha"], "dir": str(c.backend_dir)}
    c.run.runs_dir.mkdir(parents=True, exist_ok=True)
    rec = c.run.runs_dir / f"semantics_p1_{c.a.scene}_{datetime.now():%Y-%m-%d_%H%M}.json"
    _atomic(rec, json.dumps(c.results, indent=2, default=str) + "\n")
    info(f"record: {rec}")


# ── FMGS backend (Phase 4) ─────────────────────────────────────────────────────
def _fmgs_cfg(c):
    from radiance_semantics.fmgs.train import TrainConfig
    return TrainConfig(steps=int(c.a.fmgs_steps), feat_width=int(c.a.fmgs_width), variant=c.a.fmgs_variant,
                       impl=c.a.fmgs_impl, log2_table=int(c.a.fmgs_table))


def step_fmgs(c):
    from radiance_semantics.fmgs.train import train_run
    if not (c.teacher_dir / "meta.json").exists():
        raise StepFailed(f"no teacher features at {c.teacher_dir}: run the teachers step")
    out = c.run.semantics_dir / c.run.run / "fmgs_train"
    sha0 = _file_sha(c.run.checkpoint)
    with VramMonitor() as vm:
        stats, _ = train_run(c.run, c.teacher_dir, _fmgs_cfg(c), out, c.a.device, c.a.render_backend, log=info,
                             limit=c.a.limit)
    sha1 = _file_sha(c.run.checkpoint)
    stats.update(peak_vram_mib_device=vm.peak, checkpoint_sha_before=sha0, checkpoint_sha_after=sha1,
                 teacher_tag=c.teacher_settings().tag(), key=c.a.key, dir=str(out))
    (out / "train.json").write_text(json.dumps(stats, indent=2, default=str) + "\n")
    fb = stats.get("fallback")
    ok(f"field trained: {stats['steps']} steps in {stats['seconds'] / 60:.1f} min ({stats['it_per_s']} it/s), loss "
       f"{stats['loss_first']} → {stats['loss_last']}, peak {stats['peak_vram_mib']} MiB (torch) / {vm.peak} MiB (device); "
       f"variant {stats['variant']}" + (f", fallback {fb['level']}: {fb['name']}" if fb and fb["level"] else ""))
    if not stats["gauss_unchanged"] or sha0 != sha1:
        raise StepFailed("the Gaussians or the checkpoint changed during training — the splat must stay frozen")
    ok(f"Gaussians unchanged (checksum {stats['gauss_checksum_after']}; checkpoint file {sha1})")
    if not (stats["loss_last"] < stats["loss_first"]):
        warn("the loss did not fall — check the TensorBoard curves in " + str(out / "tb"))
    c.results["fmgs"] = stats


def step_bake(c):
    import numpy as np
    import torch
    from radiance_semantics.fmgs.bake import bake
    from radiance_semantics.fmgs.field import FeatureField
    from radiance_semantics.store import read_table, write_table
    tdir = c.run.semantics_dir / c.run.run / "fmgs_train"
    tj = json.loads((tdir / "train.json").read_text()) if (tdir / "train.json").exists() else None
    if tj is None or tj.get("key") != c.a.key or not (tdir / "field.pt").exists():
        raise StepFailed(f"no trained field for this checkpoint in {tdir}: run the fmgs step")
    dev = c.device
    field, saved = FeatureField.load(tdir / "field.pt", dev)
    means, scales, opac, order = _ckpt_gaussians(c)
    t0 = time.time()
    clip, dino = bake(field, means[order], device=dev)
    info(f"field evaluated at {len(order):,} Gaussians in {time.time() - t0:.1f} s")
    # which Gaussians the training cameras saw: the lift table's weights when it is for this checkpoint,
    # otherwise the same blend-weight pass the lift runs
    lift_dir = c.run.backend_dir("lift")
    w = None
    if (lift_dir / "index.json").exists():
        lt = read_table(lift_dir)
        if lt.index.get("key") == c.a.key and lt.n == len(order):
            w = np.asarray(lt.weight, np.float32)
            info("seen / unseen rows from the lift table (same checkpoint)")
    if w is None:
        from radiance_semantics.cameras import in_workspace, load_pipeline, train_views
        from radiance_semantics.lift import blend_weights, gaussians_from_model, views_from_pipeline
        with in_workspace(c.run):
            _, pipeline, _, _ = load_pipeline(c.run)
            cams, files, _ = train_views(pipeline)
            views = views_from_pipeline(pipeline.model, cams, files)
            g = gaussians_from_model(pipeline.model)
            del pipeline
        w = blend_weights(g, views[:c.a.limit] if c.a.limit else views, int(c.a.fmgs_width), c.a.render_backend)[order]
        del g
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    clip[w <= 0] = 0
    dino[w <= 0] = 0
    ckpt = c.run.checkpoint
    index = write_table(c.backend_dir, clip=clip, dino=dino, weight=w, geom=_geom(means, scales, opac, order),
                        order=order, meta={
        "scene": c.a.scene, "run": c.run.run, "checkpoint": ckpt.name, "checkpoint_mtime": int(ckpt.stat().st_mtime),
        "key": c.a.key, "backend": "fmgs", "teacher_tag": tj.get("teacher_tag"), "n_total": int(len(means)),
        "settings": {k: getattr(c.a, k) for k in DEFAULTS if k not in ("device", "limit")},
        "metrics": {"fmgs": {k: tj.get(k) for k in ("steps", "seconds", "it_per_s", "loss_first", "loss_last",
                                                     "peak_vram_mib", "peak_vram_mib_device", "variant", "fallback",
                                                     "params_m", "trainable_gaussians", "gauss_unchanged")},
                    "lift": {"seconds": tj.get("seconds"), "peak_vram_mib": tj.get("peak_vram_mib_device"),
                             "render_width": int(c.a.fmgs_width), "views": tj.get("views")},
                    "seen_rows": int((w > 0).sum())}})
    shutil.copy2(tdir / "field.pt", c.backend_dir / "field.pt")        # the field answers any xyz (Stage 3)
    size = sum(f.stat().st_size for f in c.backend_dir.iterdir()) / 2 ** 20
    ok(f"table: {index['n']:,} rows (.splat order, {index['order_sha']}), {int((w > 0).sum()):,} seen, "
       f"{size:.0f} MB → {c.backend_dir}")
    c.results["bake"] = {"rows": index["n"], "seen_rows": int((w > 0).sum()), "mb": round(size, 1),
                         "order_sha": index["order_sha"], "dir": str(c.backend_dir)}
    c.run.runs_dir.mkdir(parents=True, exist_ok=True)
    rec = c.run.runs_dir / f"semantics_p4_{c.a.scene}_{datetime.now():%Y-%m-%d_%H%M}.json"
    _atomic(rec, json.dumps(c.results, indent=2, default=str) + "\n")
    info(f"record: {rec}")


STEPS = [
    ("preflight", step_preflight, [], "environment, GPU, weights, active run, time and disk estimate"),
    ("cameras", step_cameras, [], "refined vs raw pose render check"),
    ("teachers", step_teachers, ["teachers", "scales", "dino_width"], "CLIP pyramid + DINOv2 per frame (LONG, GPU)"),
    ("lift", step_lift, ["backend", "feat_width", "render_backend", "limit"], "lift teacher features onto the Gaussians (GPU)"),
    ("export", step_export, [], "write the per-Gaussian table in .splat order"),
    ("fmgs", step_fmgs, ["fmgs_steps", "fmgs_width", "fmgs_variant", "fmgs_impl", "fmgs_table", "render_backend", "limit"],
     "train the FMGS hash-grid field on the frozen Gaussians (LONG, GPU)"),
    ("bake", step_bake, [], "evaluate the field at every Gaussian → the fmgs table in .splat order"),
]
BACKEND_STEPS = {"lift": ["preflight", "cameras", "teachers", "lift", "export"],
                 "fmgs": ["preflight", "cameras", "teachers", "fmgs", "bake"]}


def steps_for(backend):
    return [s for s in STEPS if s[0] in BACKEND_STEPS[backend]]
_ALWAYS = {"preflight"}


def run_step_here(c, name):
    fn = {n: f for n, f, _, _ in STEPS}[name]
    try:
        fn(c)
    except KeyboardInterrupt:
        print()
        warn(f"interrupted during '{name}' — marker NOT written, this step will re-run (teachers resume per frame)")
        c.save_results()
        return 130
    except StepFailed as e:
        print()
        fail(f"step '{name}' failed:\n\n{e}\n")
        c.save_results()
        return 1
    except Exception:
        print()
        tb = traceback.format_exc()
        traceback.print_exc()
        if re.search(r"CUDA out of memory|OutOfMemoryError", tb):
            fail("GPU out of memory. Check nvidia-smi for other GPU users (Galley job, ns-viewer); then "
                 "--feat-width 480 (lift), --batch 64 (teachers), or for fmgs --fmgs-variant blite / --fmgs-table 19.")
        else:
            d = diagnose(tb)
            if d:
                fail(d)
        info(f"after fixing, resume with:  semantic_pipeline.py --scene {c.a.scene} --from {name}")
        c.save_results()
        return 1
    c.save_results()
    return 0


def run_step_child(c, name):
    import subprocess
    argv = [sys.executable, str(Path(__file__).resolve()), "--project-root", str(c.project_root),
            "--scene", c.a.scene, "--backend", c.a.backend, "--in-step", name]
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    env["PYTHONUNBUFFERED"] = "1"
    try:
        return subprocess.run(argv, env=env).returncode
    except KeyboardInterrupt:
        warn(f"interrupted during '{name}' — marker NOT written, this step will re-run")
        return 130


def main(argv=None):
    ap = argparse.ArgumentParser(description="per-Gaussian semantic features for a trained splat (resumable)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--backend", choices=["lift", "fmgs"], help="feature backend (default lift; fmgs = Phase 4)")
    ap.add_argument("--teachers", help="comma-separated: clip,dino (default both)")
    ap.add_argument("--feat-width", type=int, help="lift render width in px (default 960; 480 halves time)")
    ap.add_argument("--dino-width", type=int, help="DINOv2 input width, multiple of 14 (default 896)")
    ap.add_argument("--scales", type=float, nargs="+", help="CLIP pyramid scales (default 7 from 0.05 to 0.5)")
    ap.add_argument("--batch", type=int, help="CLIP crops per forward pass (default 256)")
    ap.add_argument("--fmgs-steps", type=int, help="FMGS training iterations (default 4200)")
    ap.add_argument("--fmgs-width", type=int, help="FMGS feature render width in px (default 480; teachers are ~71 cells wide)")
    ap.add_argument("--fmgs-variant", choices=["auto", "faithful", "blite"],
                    help="auto = faithful FMGS with the out-of-memory fallback ladder (default)")
    ap.add_argument("--fmgs-impl", choices=["auto", "tcnn", "torch", "torch/tcnn", "tcnn/torch"],
                    help="field implementation, encoding/heads (default auto: tiny-cuda-nn where its probe passes)")
    ap.add_argument("--fmgs-table", type=int, help="log2 of the hash table size (default 20)")
    ap.add_argument("--device", help=argparse.SUPPRESS)               # tests: cpu
    ap.add_argument("--render-backend", choices=["gsplat", "reference"], help=argparse.SUPPRESS)
    ap.add_argument("--limit", type=int, help=argparse.SUPPRESS)      # tests: first N frames/views
    ap.add_argument("--redo", action="append", default=[], metavar="STEP")
    ap.add_argument("--from", dest="from_step", metavar="STEP")
    ap.add_argument("--only", metavar="STEP")
    ap.add_argument("--stop-after", metavar="STEP")
    ap.add_argument("--list-steps", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--in-step", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if not a.backend:
        a.backend = "lift"
    steps = steps_for(a.backend)

    names = [s[0] for s in steps]
    if a.in_step:
        names = [s[0] for s in STEPS]
    if a.list_steps:
        for n, _, keys, d in steps:
            print(f"  {n:<10} {d}" + (f"   {_C['d']}[{','.join(keys)}]{_C['x']}" if keys else ""))
        return 0
    for opt in (a.from_step, a.only, a.stop_after, *a.redo):
        if opt and opt not in names:
            ap.error(f"unknown step '{opt}' for --backend {a.backend}. Options: {', '.join(names)}")
    if a.only and (a.from_step or a.stop_after):
        ap.error("--only cannot be combined with --from/--stop-after")

    from radiance_semantics.paths import SemanticsError
    try:
        c = Ctx(a)
        a.key = c.run.key
    except SemanticsError as e:
        fail(str(e))
        return 1

    if a.status:
        section(f"Status — {a.scene} {c.run.run}")
        for n, _, _, d in STEPS:                      # both backends
            m = c.state / f"{n}.done"
            when = m.read_text().strip().split("\n")[-1] if m.exists() else ""
            print(f"  {'✔' if m.exists() else '·'} {n:<10} {when:<20} {d}")
        info(f"state: {c.state}")
        info(f"tables: {c.run.backend_dir('lift')} · {c.run.backend_dir('fmgs')}")
        return 0

    cfg_path = c.state / "config.json"
    saved = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    for k, dflt in DEFAULTS.items():
        v = getattr(a, k)
        if v is None:
            v = dflt if k in NOT_STICKY else saved.get(k, dflt)
        elif k in LISTS:
            v = [x.strip() for x in v.split(",") if x.strip()]
        setattr(a, k, v)
    if not set(a.teachers) <= {"clip", "dino"} or "clip" not in a.teachers:
        ap.error("--teachers: clip (required) and optionally dino")
    if a.in_step:
        return run_step_here(c, a.in_step)
    _atomic(cfg_path, json.dumps({k: getattr(a, k) for k in DEFAULTS}, indent=2) + "\n")
    c.results["config"] = {k: getattr(a, k) for k in DEFAULTS}
    c.results["key"] = a.key
    c.save_results()

    for n in a.redo:
        c.clear(n)
        info(f"cleared marker for '{n}'")
    selected = names
    if a.only:
        selected = [a.only]
    else:
        if a.from_step:
            selected = names[names.index(a.from_step):]
        if a.stop_after:
            selected = [n for n in selected if names.index(n) <= names.index(a.stop_after)]
    if "preflight" not in selected:
        selected = ["preflight"] + selected

    print(f"{_C['b']}Semantic pipeline{_C['x']}  scene={a.scene}  run={c.run.run}  backend={a.backend}  "
          f"teachers={','.join(a.teachers)}  steps={','.join(selected)}")
    info(f"state: {c.state}")

    t_all = time.time()
    prev = None
    for n, fn, keys, desc in steps:
        this_prev = prev if n != "preflight" else None
        if n != "preflight":
            prev = n
        if n not in selected:
            continue
        fp = c.fingerprint(keys, this_prev)
        if n not in _ALWAYS and c.is_done(n, fp) and n not in a.redo:
            print(f"  {_C['d']}· {n:<10} already done — skipping{_C['x']}")
            continue
        section(f"{n}  —  {desc}")
        t0 = time.time()
        rc = run_step_child(c, n)
        c.reload_results()
        if rc != 0:
            return rc
        if n not in _ALWAYS:
            c.mark_done(n, c.fingerprint(keys, this_prev))
        c.save_results()
        info(f"'{n}' completed in {elapsed(t0)}")

    section(f"Done — total {elapsed(t_all)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
