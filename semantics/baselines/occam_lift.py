"""Lift our teacher maps with the ORIGINAL Occam's LGS code (insait-institute/OccamLGS) — `occamlgs` env.

    source semantics/baselines/env.sh occamlgs
    python semantics/baselines/occam_lift.py --scene backroom --teachers d7a30d7264 --suffix _occam_pyr

Controlled comparison: the 2D features are our cached teachers, upsampled to the render size exactly as
our lift does (bilinear, align_corners=False). The lift is Occam's own: their renderer
(`gaussian_renderer.render`, on their gsplat fork, which records per Gaussian the blend weight
`significance` at the pixel containing its projected centre and an `activated` flag) and their
accumulation `GaussianModel.accumulate_gaussian_feature_per_view` (feature at the centre pixel × that
weight, summed over views, eq. 7 of the paper). Their `finalize_gaussian_features` divides by the summed
weight and then PRUNES never-activated Gaussians; we do the division the same way but keep those rows
as zeros with weight 0, so the table stays in .splat order (pruned = never-seen, the same thing).

Output: <semantics_dir>/lift<suffix>_raw/ in our lift step's layout; build the table with
    python figs/semantic_pipeline.py --scene backroom --backend lift --table-suffix _occam_pyr \
        --clip-mode pyramid --only export        (kitchen env)

Note on `weight`: it is Occam's accumulated centre-pixel weight, not the all-pixel blend weight of our
lift; the query only uses weight > 0 (seen), so the two are interchangeable there.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

THIRD_PARTY = Path(os.environ.get("THIRD_PARTY", Path.home() / "Radiance/third_party"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--teachers", required=True, help="teacher cache tag, e.g. d7a30d7264 (Old) or 9c437cce68 (A)")
    ap.add_argument("--suffix", required=True, help="table suffix, e.g. _occam_pyr → lift_occam_pyr")
    ap.add_argument("--names", nargs="*", help="teachers to lift (default: dino + clip, or dino + clip_s* for A)")
    ap.add_argument("--width", type=int, default=960, help="render width (our lift's default 960)")
    ap.add_argument("--exports", default=str(THIRD_PARTY / "exports"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, str(THIRD_PARTY / "OccamLGS"))
    import torch
    import torch.nn.functional as F
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import getWorld2View2

    ex = Path(a.exports) / a.scene
    man = json.loads((ex / "manifest.json").read_text())
    cams = json.loads((ex / "cameras.json").read_text())
    if a.limit:
        cams = cams[:a.limit]
    tdir = Path(man["semantics_dir"]) / "teachers" / a.teachers
    names = a.names or (["dino"] + (sorted(p.name for p in tdir.glob("clip_s*") if p.is_dir()) or ["clip"]))
    raw = Path(man["semantics_dir"]) / man["run"] / f"lift{a.suffix}_raw"
    raw.mkdir(parents=True, exist_ok=True)

    def camera(c):
        W = a.width
        H = max(1, int(round(c["height"] * W / c["width"])))
        wv = torch.tensor(getWorld2View2(np.asarray(c["R"]), np.asarray(c["T"]))).transpose(0, 1).cuda()
        return SimpleNamespace(FoVx=c["fovx"], FoVy=c["fovy"], image_width=W, image_height=H,
                               world_view_transform=wv), H, W

    bg = torch.zeros(3, device="cuda")
    t_all = time.time()
    weight = None
    N = None
    for name in names:
        t0 = time.time()
        g = GaussianModel(3)                       # fresh model per teacher: its accumulators live on it
        g.load_ply(str(ex / "point_cloud.ply"))
        N = g.get_xyz.shape[0]
        assert N == man["n_gaussians"], (N, man["n_gaussians"])
        if name == names[0]:
            print(f"{a.scene}: {N:,} Gaussians, {len(cams)} views, teachers {a.teachers} {names}, "
                  f"render {a.width} px → {raw}")
        for k, c in enumerate(cams):
            cam, H, W = camera(c)
            grid = np.load(tdir / name / f"{c['stem']}.npy").astype(np.float32)        # [gh, gw, C]
            t = torch.from_numpy(grid).cuda().permute(2, 0, 1)[None]
            feat = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0]   # [C, H, W]
            with torch.no_grad():
                pkg = render(cam, g, None, bg, override_color=torch.ones_like(g.get_xyz))
                info = pkg["info"]
                activated, significance, means2D = info["activated"], info["significance"], info["means2d"]
                mask = activated[0] > 0
                ones = torch.ones((H, W), dtype=torch.bool, device="cuda")   # our teachers cover every pixel
                # --- Occam's code (gaussian_feature_extractor.extract_gaussian_features) ---
                g.accumulate_gaussian_feature_per_view(feat.permute(1, 2, 0), ones, mask, significance[0, mask],
                                                       means2D[0, mask], feature_dim=feat.shape[0])
            del feat, t, pkg, info
            if k == 0 or (k + 1) % 50 == 0 or k == len(cams) - 1:
                print(f"  {name}: {k + 1}/{len(cams)} views, {time.time() - t0:.0f} s, "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
        with torch.no_grad():
            # finalize_gaussian_features without its prune: never-activated rows → 0, weight 0
            feats = g._language_feature / (g._language_feature_weight + 1e-15)
            seen = g._activated_views > 0
            feats[~seen] = 0
            w = g._language_feature_weight[:, 0].clone()
            w[~seen] = 0
        if weight is None:
            weight = w
        np.save(raw / f"{name}.npy", feats.detach().cpu().numpy().astype(np.float16))
        print(f"  {name}: done in {time.time() - t0:.0f} s, {int(seen.sum()):,} activated")
        del g, feats
        torch.cuda.empty_cache()
    np.save(raw / "weight.npy", weight.cpu().numpy().astype(np.float32))
    seen = int((weight > 0).sum())
    stats = {"views": len(cams), "seconds": round(time.time() - t_all, 1), "seen": seen, "n": N,
             "seen_frac": round(seen / N, 4), "render_width": a.width, "teacher_tag": a.teachers, "key": man["key"],
             "impl": "Occam's LGS original (OccamLGS accumulate_gaussian_feature_per_view, gsplat fork)",
             "peak_vram_mib": int(torch.cuda.max_memory_allocated() / 2**20)}
    (raw / "lift.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"lifted {names} onto {seen:,} of {N:,} Gaussians in {stats['seconds']:.0f} s → {raw}")


if __name__ == "__main__":
    main()
