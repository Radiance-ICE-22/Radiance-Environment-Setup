"""Phase 0 probe: does the installed gsplat render N-channel features with correct gradients,
and what does one lift pass cost on this GPU?

    python -m radiance_semantics.probe --quick                       # seconds; used by verify_figs.sh
    python -m radiance_semantics.probe                               # synthetic, 1.5 M Gaussians
    python -m radiance_semantics.probe --scene backroom --record     # the real splat + a real camera

Checks, at each resolution (all exact identities because rendering is linear in features):

  linearity      L(f) = <f, dL/df>  for a random f, where dL/df was taken at f = 0
  finite diff    L(e_ic) = dL/df_ic  for the 5 most-visible Gaussians (L(0) = 0, so no cancellation)
  weights        rendering all-ones features reproduces the alpha image, and the per-Gaussian
                 blend weights (the lift denominators) sum to the total alpha and are ≥ 0
  chunking       rendering 2C channels at once equals two C-channel chunks

Timing is one forward + backward with C channels (default 32: gsplat 1.0.0's backward kernel
takes at most 32, so that is the lift's chunk size); from it the probe estimates one full lift
(512 CLIP + 384 DINO channels + one weight pass per image). A further check renders 2C channels
WITH gradients, which goes through render_features' automatic chunking.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime

from .log import Timer, VramMonitor, emit, fail, info, ok, section, warn

TOL = 2e-3                  # relative; fp32 sums over ~10^5–10^6 pixel contributions


def synthetic(n, seed, device):
    """Gaussians in front of an identity camera (OpenCV: +z forward), room-like sizes."""
    import torch
    from .render import Gaussians
    g = torch.Generator().manual_seed(seed)
    means = torch.stack([torch.empty(n).uniform_(-3, 3, generator=g),
                         torch.empty(n).uniform_(-2, 2, generator=g),
                         torch.empty(n).uniform_(1, 8, generator=g)], 1)
    scales = torch.exp(torch.randn(n, 3, generator=g) * 0.6 + math.log(0.03))
    quats = torch.randn(n, 4, generator=g)
    quats = quats / quats.norm(dim=1, keepdim=True)
    opac = torch.empty(n).uniform_(0.05, 1.0, generator=g)
    return Gaussians(means, quats, scales, opac).to(device)


def synthetic_camera(width, height, device):
    import torch
    from .render import intrinsics
    f = 0.8 * width
    return torch.eye(4, device=device), intrinsics(f, f, width / 2, height / 2, device)


def check(g, viewmat, K, width, height, channels, backend, seed=0, fd_count=5):
    """The four identities above. Returns a dict of errors (relative) and visibility counts."""
    import torch
    from .render import render_features
    dev = g.means.device
    n, C = len(g), channels
    gen = torch.Generator(device="cpu").manual_seed(seed)
    F2D = torch.randn(height, width, C, generator=gen).to(dev)

    def L(f):
        with torch.no_grad():
            return float((render_features(g, f, viewmat, K, width, height, backend)[0] * F2D).sum())

    f0 = torch.zeros(n, C, device=dev, requires_grad=True)
    img, _ = render_features(g, f0, viewmat, K, width, height, backend)
    (img * F2D).sum().backward()
    grad = f0.grad.detach()
    del f0, img

    f = (torch.randn(n, C, generator=gen) * 0.1).to(dev)
    Lf, Lhat = L(f), float((f * grad).sum())
    lin = abs(Lf - Lhat) / max(abs(Lf), 1e-12)
    del f

    rows = grad.norm(dim=1)
    k = min(fd_count, int((rows > 0).sum()))
    fd = []
    for i in rows.topk(k).indices.tolist():
        c = int(torch.randint(C, (1,), generator=gen))
        e = torch.zeros(n, C, device=dev)
        e[i, c] = 1.0
        fd.append(abs(L(e) - float(grad[i, c])) / max(abs(float(grad[i, c])), 1e-12))
        del e

    ones = torch.ones(n, 1, device=dev, requires_grad=True)
    img1, alpha1 = render_features(g, ones, viewmat, K, width, height, backend)
    img1.sum().backward()
    w = ones.grad.detach()[:, 0]
    a_sum = float(alpha1.sum())
    ones_vs_alpha = float((img1 - alpha1).abs().max())
    w_sum = abs(float(w.sum()) - a_sum) / max(a_sum, 1e-12)
    w_min = float(w.min())
    visible = int((w > 0).sum())
    del ones, img1, alpha1

    with torch.no_grad():
        both = torch.randn(n, 2 * C, generator=gen).to(dev)
        whole = render_features(g, both, viewmat, K, width, height, backend)[0]
        parts = torch.cat([render_features(g, both[:, :C], viewmat, K, width, height, backend)[0],
                           render_features(g, both[:, C:], viewmat, K, width, height, backend)[0]], -1)
        chunk = float((whole - parts).abs().max() / whole.abs().max().clamp_min(1e-12))
        del both, whole, parts

    # 2C channels with gradients: on gsplat this exceeds the backward kernel's limit when C = 32,
    # so render_features must chunk it; the gradient must equal the two C-channel gradients.
    F2 = torch.randn(height, width, 2 * C, generator=gen).to(dev)
    f2 = torch.zeros(n, 2 * C, device=dev, requires_grad=True)
    (render_features(g, f2, viewmat, K, width, height, backend)[0] * F2).sum().backward()
    ga = torch.zeros(n, C, device=dev, requires_grad=True)
    gb = torch.zeros(n, C, device=dev, requires_grad=True)
    (render_features(g, ga, viewmat, K, width, height, backend)[0] * F2[..., :C]).sum().backward()
    (render_features(g, gb, viewmat, K, width, height, backend)[0] * F2[..., C:]).sum().backward()
    both_grad = torch.cat([ga.grad, gb.grad], -1)
    grad_chunk = float((f2.grad - both_grad).abs().max() / both_grad.abs().max().clamp_min(1e-12))
    del F2, f2, ga, gb, both_grad

    res = {"linearity_rel": lin, "finite_diff_rel_max": max(fd) if fd else None, "finite_diff_n": len(fd),
           "ones_vs_alpha_abs": ones_vs_alpha, "weight_sum_rel": w_sum, "weight_min": w_min,
           "chunk_rel": chunk, "grad_chunk_rel": grad_chunk, "visible": visible, "visible_frac": visible / n,
           "alpha_mean": a_sum / (width * height)}
    res["ok"] = (lin < TOL and (not fd or max(fd) < TOL) and ones_vs_alpha < 1e-4 and w_sum < TOL
                 and w_min > -1e-5 and chunk < 1e-4 and grad_chunk < 1e-4 and visible > 0)
    return res


def bench(g, viewmat, K, width, height, channels, backend, reps=3):
    """Seconds per forward + backward with `channels` features, and peak allocated MiB."""
    import torch
    from .render import render_features
    cuda = g.means.is_cuda

    def once():
        f = torch.zeros(len(g), channels, device=g.means.device, requires_grad=True)
        img, _ = render_features(g, f, viewmat, K, width, height, backend)
        img.sum().backward()
        if cuda:
            torch.cuda.synchronize()

    once()                                                 # warm-up: JIT/caches
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        once()
    s = (time.perf_counter() - t0) / reps
    peak = round(torch.cuda.max_memory_allocated() / 2 ** 20) if cuda else None
    return s, peak


def lift_passes(channels, clip_dim=512, dino_dim=384):
    """Render passes per image for one lift: CLIP and DINO in chunks, plus the weight pass."""
    return math.ceil(clip_dim / channels) + math.ceil(dino_dim / channels) + 1


def parse_res(s):
    out = []
    for part in s.split(","):
        w, h = part.lower().split("x")
        out.append((int(w), int(h)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", help="FiGS prefix (default: from figs_env.sh)")
    ap.add_argument("--scene", help="use this scene's trained splat and a training camera (default: synthetic)")
    ap.add_argument("--n", type=int, default=1_500_000, help="synthetic Gaussians (≈ backroom's 371 MB checkpoint)")
    ap.add_argument("--channels", type=int, default=32, help="≤ 32 on gsplat 1.0.0 (backward kernel limit)")
    ap.add_argument("--res", default="480x270,960x540", help="comma-separated WxH")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--backend", default="gsplat", choices=["gsplat", "reference"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="20k Gaussians at 256x144, one rep (a few seconds)")
    ap.add_argument("--record", action="store_true", help="also write SousVide/runs/semantics_p0_probe_*.json")
    a = ap.parse_args(argv)
    if a.quick:
        a.n, a.res, a.reps = 20_000, "256x144", 1

    import torch
    from .paths import SemanticsError
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.backend == "gsplat" and device != "cuda":
        fail("gsplat's rasterizer needs CUDA (torch.cuda.is_available() is False)")
        emit({"ok": False, "error": "no CUDA"})
        return 1

    out = {"tool": "probe", "backend": a.backend, "device": device, "channels": a.channels,
           "started": datetime.now().isoformat(timespec="seconds")}
    if device == "cuda":
        out["gpu"] = torch.cuda.get_device_name(0)
    run = None
    try:
        if a.scene:
            from .cameras import scene_camera
            from .paths import find_scene_run, resolve_project_root
            from .render import Gaussians
            run = find_scene_run(resolve_project_root(a.project_root), a.scene)
            ck = torch.load(run.checkpoint, map_location="cpu", weights_only=False)
            g = Gaussians.from_splatfacto(ck.get("pipeline", ck)).to(device)
            out.update(source=f"{a.scene} {run.run}/{run.checkpoint.name}", n_images=len(json.loads(
                run.transforms_json.read_text())["frames"]))
            del ck
        else:
            g = synthetic(a.n, a.seed, device)
            out.update(source="synthetic", n_images=300)
    except SemanticsError as e:
        fail(str(e))
        emit({"ok": False, "error": str(e)})
        return 1
    out["n"] = len(g)
    section(f"gsplat N-channel probe — {out['source']}, {len(g):,} Gaussians, {a.channels} channels, {device}")

    all_ok = True
    out["res"] = {}
    for (W, H) in parse_res(a.res):
        if run is not None:
            c2w, K = scene_camera(run, W, H, device=device)
            from .render import viewmat_from_c2w
            vm = viewmat_from_c2w(c2w)
        else:
            vm, K = synthetic_camera(W, H, device)
        key = f"{W}x{H}"
        try:
            with VramMonitor(0.5) as mon, Timer() as tm:
                r = check(g, vm, K, W, H, a.channels, a.backend, seed=a.seed)
                s, peak = bench(g, vm, K, W, H, a.channels, a.backend, a.reps)
        except RuntimeError as e:                    # CUDA errors: report and go on to the next size
            fail(f"{key}: {type(e).__name__}: {e}")
            out["res"][key] = {"ok": False, "error": str(e)}
            all_ok = False
            if device == "cuda":
                torch.cuda.empty_cache()
            continue
        r.update(fwd_bwd_s=round(s, 4), peak_alloc_mib=peak, peak_device_mib=mon.peak, check_s=round(tm.s, 1))
        r["lift_estimate_s"] = round(out["n_images"] * lift_passes(a.channels) * s, 1)
        out["res"][key] = r
        all_ok &= r["ok"]
        (ok if r["ok"] else fail)(
            f"{key}: linearity {r['linearity_rel']:.1e}, finite diff {r['finite_diff_rel_max'] or 0:.1e} "
            f"(n={r['finite_diff_n']}), weights {r['weight_sum_rel']:.1e} (min {r['weight_min']:.1e}), "
            f"chunks {r['chunk_rel']:.1e} (with grad {r['grad_chunk_rel']:.1e})")
        info(f"{r['visible']:,} Gaussians visible ({r['visible_frac']:.0%}), mean alpha {r['alpha_mean']:.2f}")
        info(f"forward+backward {r['fwd_bwd_s'] * 1000:.0f} ms; peak allocated {peak} MiB, device {mon.peak} MiB; "
             f"one lift over {out['n_images']} images ≈ {r['lift_estimate_s'] / 60:.1f} min")
        if r["visible"] < 0.01 * len(g):
            warn("under 1% of the Gaussians are visible from this camera; timing may be optimistic")

    out["ok"] = all_ok
    if a.record:
        root = run.runs_dir if run else None
        if root is None:
            try:
                from .paths import resolve_project_root
                root = resolve_project_root(a.project_root) / "SousVide" / "runs"
            except SemanticsError:
                root = None
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            p = root / f"semantics_p0_probe_{a.scene or 'synthetic'}_{datetime.now():%Y-%m-%d_%H%M}.json"
            p.write_text(json.dumps(out, indent=2) + "\n")
            info(f"record: {p}")
        else:
            warn("--record: no FiGS project root found, nothing written")
    emit(out)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
