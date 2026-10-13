"""Export a trained scene for the original LUDVIG / Occam's LGS code (run in kitchen).

    source ~/Radiance/figs/figs_env.sh
    python ~/Radiance/Radiance-Environment-Setup/semantics/baselines/export_scene.py --scene backroom

Writes ~/Radiance/third_party/exports/<scene>/ with
  point_cloud.ply   the splat in the Inria 3DGS layout, rows in CHECKPOINT order (raw parameters:
                    log scales, opacity logits, wxyz quaternions, SH DC + rest)
  cameras.json      the lift's training views (refined poses, pose outliers dropped, same order as
                    lift.views_from_pipeline): nerfstudio c2w (OpenGL) and the Inria/COLMAP form
                    (R = c2w rotation in OpenCV axes, T = w2c translation), FoV, intrinsics
  images/           symlinks to the training images, named <stem><ext>
  manifest.json     scene, run, checkpoint key, Gaussian count, teacher cache dirs

Everything is in the trained model's frame (after nerfstudio's dataparser transform), the frame our
tables and the lift use. Nothing in the repo or the semantics dirs is written.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "semantics"), str(REPO / "figs")]

GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def write_inria_ply(path, means, scales, quats, opac, f_dc, f_rest):
    """Inria 3DGS point_cloud.ply. f_rest [N, K, 3] is written channel-major (f_rest_0.. = R's K coeffs,
    then G, then B), as GaussianModel.save_ply does with features_rest.transpose(1, 2)."""
    N = len(means)
    rest = f_rest.transpose(0, 2, 1).reshape(N, -1)
    cols = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
            + [f"f_rest_{i}" for i in range(rest.shape[1])] + ["opacity"]
            + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.concatenate([means, np.zeros_like(means), f_dc, rest, opac[:, None], scales, quats], 1)
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {N}\n"
              + "".join(f"property float {c}\n" for c in cols) + "end_header\n")
    with open(path, "wb") as f:                      # plain binary PLY (kitchen has no plyfile)
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(data, dtype="<f4").tobytes())   # row-major = one vertex per row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--project-root", default=str(Path.home() / "Radiance/figs"))
    ap.add_argument("--out", default=str(Path.home() / "Radiance/third_party/exports"))
    a = ap.parse_args()

    import torch
    from radiance_semantics.cameras import in_workspace, load_pipeline, train_views
    from radiance_semantics.lift import views_from_pipeline
    from radiance_semantics.paths import find_scene_run

    run = find_scene_run(Path(a.project_root), a.scene)
    out = Path(a.out) / a.scene
    (out / "images").mkdir(parents=True, exist_ok=True)

    with in_workspace(run):
        _, pipeline, _, step = load_pipeline(run)
        model = pipeline.model
        cams, files, _ = train_views(pipeline)
        files = [Path(f).resolve() for f in files]
        views = views_from_pipeline(model, cams, files, log=print)
        with torch.no_grad():
            P = {k: getattr(model, k).detach().float().cpu().numpy()
                 for k in ("means", "scales", "quats", "opacities", "features_dc", "features_rest")}
        del pipeline

    N = len(P["means"])
    write_inria_ply(out / "point_cloud.ply", P["means"], P["scales"], P["quats"], P["opacities"].reshape(-1),
                    P["features_dc"].reshape(N, 3), P["features_rest"].reshape(N, -1, 3))

    by_stem = {f.stem: f for f in files}
    cam_out = []
    for v in views:
        src = by_stem[v.stem]
        link = out / "images" / f"{v.stem}{src.suffix}"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src)
        c2w = np.eye(4)
        c2w[:3] = v.c2w.cpu().numpy()
        c2w_cv = c2w @ GL_TO_CV
        w2c = np.linalg.inv(c2w_cv)
        cam_out.append({
            "stem": v.stem, "image": str(link), "width": v.width, "height": v.height,
            "fx": v.fx, "fy": v.fy, "cx": v.cx, "cy": v.cy,
            "fovx": 2 * math.atan(v.width / (2 * v.fx)), "fovy": 2 * math.atan(v.height / (2 * v.fy)),
            "c2w_gl": c2w[:3].tolist(), "R": c2w_cv[:3, :3].tolist(), "T": w2c[:3, 3].tolist(),
            "w2c_cv": w2c.tolist()})
        if abs(v.cx - v.width / 2) > 0.5 or abs(v.cy - v.height / 2) > 0.5:
            print(f"  ! {v.stem}: principal point ({v.cx:.1f}, {v.cy:.1f}) is off-centre; "
                  f"the Inria rasteriser assumes the centre")
    (out / "cameras.json").write_text(json.dumps(cam_out, indent=1) + "\n")

    teachers = sorted(p for p in (run.semantics_dir / "teachers").iterdir() if (p / "meta.json").exists())
    manifest = {"scene": a.scene, "run": run.run, "key": run.key, "checkpoint": str(run.checkpoint),
                "step": int(step), "n_gaussians": N, "n_views": len(cam_out),
                "semantics_dir": str(run.semantics_dir),
                "teachers": {p.name: json.loads((p / "meta.json").read_text()).get("settings", {}) for p in teachers}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    print(f"exported {N:,} Gaussians and {len(cam_out)} views → {out}")


if __name__ == "__main__":
    main()
