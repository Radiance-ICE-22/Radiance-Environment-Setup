"""Lift our teacher maps with the ORIGINAL LUDVIG code (naver/ludvig) — run in the `ludvig` env.

    source semantics/baselines/env.sh ludvig
    python semantics/baselines/ludvig_lift.py --scene backroom --teachers d7a30d7264 --suffix _ludvig_pyr

Controlled comparison: the 2D features are our cached teachers (the same files our lift reads),
upsampled to the same render size the same way (bilinear, align_corners=False); only the lift is
LUDVIG's — its CUDA kernel `GaussianModel.apply_weights` (Inria rasteriser + accumulation of
w_i(p)·F(p) and w_i(p) over every pixel, eq. 3 of the paper) and its normalisation
(features / (weights + 1e-8), `utils.solver.uplifting`).

Output: <semantics_dir>/lift<suffix>_raw/{<teacher>.npy (fp16, checkpoint order), weight.npy,
lift.json} — the layout our lift step writes — so the table is built by our own export step:

    source ~/Radiance/figs/figs_env.sh
    python figs/semantic_pipeline.py --scene backroom --backend lift --table-suffix _ludvig_pyr \
        --clip-mode pyramid --only export

LUDVIG's optional pruning (keep the top-k Gaussians by total weight) is `--prune K`: pruned rows are
written as zeros with weight 0 (never-seen), so table rows stay in .splat order.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

THIRD_PARTY = Path(os.environ.get("THIRD_PARTY", Path.home() / "Radiance/third_party"))
KERNEL_CHANNELS = (512, 256, 200, 100, 50, 40, 30, 20, 10, 3, 2, 1)   # the sizes apply_weights is compiled for


def chunks(C):
    """Split C channels into sizes the LUDVIG kernel supports (e.g. 384 → 256, 100, 20, 3, 3, 2)."""
    out, c0 = [], 0
    while c0 < C:
        k = next(k for k in KERNEL_CHANNELS if k <= C - c0)
        out.append((c0, c0 + k))
        c0 += k
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--teachers", required=True, help="teacher cache tag, e.g. d7a30d7264 (Old) or 9c437cce68 (A)")
    ap.add_argument("--suffix", required=True, help="table suffix, e.g. _ludvig_pyr → lift_ludvig_pyr")
    ap.add_argument("--names", nargs="*", help="teachers to lift (default: dino + clip, or dino + clip_s* for A)")
    ap.add_argument("--width", type=int, default=960, help="render width (our lift's default 960)")
    ap.add_argument("--prune", type=int, default=0, help="LUDVIG pruning: keep the K highest-weight Gaussians")
    ap.add_argument("--exports", default=str(THIRD_PARTY / "exports"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, str(THIRD_PARTY / "ludvig"))
    import torch
    import torch.nn.functional as F
    from gaussiansplatting.scene import GaussianModel
    from gaussiansplatting.scene.cameras import Simple_Camera

    ex = Path(a.exports) / a.scene
    man = json.loads((ex / "manifest.json").read_text())
    cams = json.loads((ex / "cameras.json").read_text())
    if a.limit:
        cams = cams[:a.limit]
    tdir = Path(man["semantics_dir"]) / "teachers" / a.teachers
    names = a.names or (["dino"] + (sorted(p.name for p in tdir.glob("clip_s*") if p.is_dir()) or ["clip"]))
    raw = Path(man["semantics_dir"]) / man["run"] / f"lift{a.suffix}_raw"
    raw.mkdir(parents=True, exist_ok=True)

    g = GaussianModel(sh_degree=0)
    g.load_ply(str(ex / "point_cloud.ply"))
    N = g.get_xyz.shape[0]
    assert N == man["n_gaussians"], (N, man["n_gaussians"])

    def camera(c):
        W = a.width
        H = max(1, int(round(c["height"] * W / c["width"])))
        return Simple_Camera(colmap_id=0, R=np.asarray(c["R"]), T=np.asarray(c["T"]), FoVx=c["fovx"], FoVy=c["fovy"],
                             h=H, w=W, image_name=c["stem"], uid=0), H, W

    print(f"{a.scene}: {N:,} Gaussians, {len(cams)} views, teachers {a.teachers} {names}, render {a.width} px → {raw}")
    t_all = time.time()
    weight = None
    for name in names:
        t0 = time.time()
        acc = None
        w_name = torch.zeros((N, 1), device="cuda")
        for k, c in enumerate(cams):
            grid = np.load(tdir / name / f"{c['stem']}.npy").astype(np.float32)        # [gh, gw, C]
            cam, H, W = camera(c)
            C = grid.shape[-1]
            if acc is None:
                acc = torch.zeros((N, C), device="cuda")
            t = torch.from_numpy(grid).cuda().permute(2, 0, 1)[None]
            for i, (c0, c1) in enumerate(chunks(C)):
                F2D = F.interpolate(t[:, c0:c1], size=(H, W), mode="bilinear", align_corners=False)[0].contiguous()
                part = torch.zeros((N, c1 - c0), device="cuda")
                cnt = w_name if i == 0 else torch.zeros_like(w_name)   # count the blend weights once per view
                with torch.no_grad():
                    g.apply_weights(cam, part, cnt, F2D)
                acc[:, c0:c1] += part
                del F2D, part
            if k == 0 or (k + 1) % 50 == 0 or k == len(cams) - 1:
                print(f"  {name}: {k + 1}/{len(cams)} views, {time.time() - t0:.0f} s, "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
        # the kernel adds alpha·T to cnt once per CHANNEL (inside its channel loop), so cnt = C0 · Σ w for the
        # first chunk's size C0; LUDVIG's own features carry that uniform 1/C0 factor (harmless after
        # normalisation). Divide it out so features and weight are the plain weighted average / blend weight.
        w_name /= chunks(acc.shape[1])[0][1]
        feats = acc / (w_name + 1e-8)                                   # LUDVIG utils.solver.uplifting
        w = w_name[:, 0]
        if weight is None:
            weight = w.clone()
            if a.prune and a.prune < N:
                drop = torch.argsort(weight)[:N - a.prune]
                weight[drop] = 0
        feats[weight <= 1e-8] = 0
        np.save(raw / f"{name}.npy", feats.cpu().numpy().astype(np.float16))
        print(f"  {name}: done in {time.time() - t0:.0f} s")
        del acc, feats
    weight[weight <= 1e-8] = 0
    np.save(raw / "weight.npy", weight.cpu().numpy().astype(np.float32))
    seen = int((weight > 0).sum())
    stats = {"views": len(cams), "seconds": round(time.time() - t_all, 1), "seen": seen, "n": N,
             "seen_frac": round(seen / N, 4), "render_width": a.width, "teacher_tag": a.teachers, "key": man["key"],
             "impl": "LUDVIG original (naver/ludvig apply_weights, Inria rasteriser)", "prune": a.prune,
             "peak_vram_mib": int(torch.cuda.max_memory_allocated() / 2**20)}
    (raw / "lift.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"lifted {names} onto {seen:,} of {N:,} Gaussians in {stats['seconds']:.0f} s → {raw}")


if __name__ == "__main__":
    main()
