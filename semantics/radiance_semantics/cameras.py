"""Training cameras of a trained splat, with the poses the splat was actually fitted to.

figs_pipeline.py trains splatfacto with `--pipeline.model.camera-optimizer.mode SO3xR3`, so
nerfstudio refines every training pose while it trains and keeps the corrections in the
checkpoint (`camera_optimizer.pose_adjustment`). transforms.json still holds the SfM/ArUco
poses. Any 2D feature map we lift onto the Gaussians must be projected with the REFINED pose
of its image, or the features land a little off. Note that splatfacto applies the correction
only while training (get_outputs: `if self.training`); in eval mode it renders the raw pose,
so this module applies it explicitly.

    python -m radiance_semantics.cameras --scene backroom [--views 10] [--save-images] [--record]

renders evenly spaced training views with the refined and with the raw pose, and reports PSNR
of each against the training image, plus held-out (eval split) PSNR and the size of the
corrections. Phase 0's gate: refined-pose PSNR must not be below raw-pose PSNR.
"""

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from datetime import datetime

import numpy as np

from .log import VramMonitor, emit, fail, info, ok, section, warn


# ── transforms.json cameras (no nerfstudio needed) ────────────────────────────────

def _intr(frame, tf, key):
    v = frame.get(key, tf.get(key))
    if v is None:
        raise KeyError(f"transforms.json has no '{key}'")
    return float(v)


def camera_from_transforms(tf, dp, index, width, height):
    """(c2w [3, 4], K [3, 3]) as float64 numpy for frame `index` of transforms.json (sorted by
    file_path, the extraction order), in the trained model's frame — the same transform the
    nerfstudio-data parser applies: c2w = T @ M, translation × scale (dataparser_transforms.json)
    — with intrinsics rescaled to width × height."""
    frames = sorted(tf["frames"], key=lambda f: f.get("file_path", ""))
    fr = frames[index]
    M = np.asarray(fr["transform_matrix"], dtype=np.float64)
    T = np.asarray(dp.get("transform", np.eye(4)[:3]), dtype=np.float64)
    c2w = T @ M
    c2w[:3, 3] *= float(dp.get("scale", 1.0))
    sx, sy = width / _intr(fr, tf, "w"), height / _intr(fr, tf, "h")
    K = np.array([[_intr(fr, tf, "fl_x") * sx, 0.0, _intr(fr, tf, "cx") * sx],
                  [0.0, _intr(fr, tf, "fl_y") * sy, _intr(fr, tf, "cy") * sy],
                  [0.0, 0.0, 1.0]])
    return c2w, K


def scene_camera(run, width, height, index=None, device="cpu"):
    """A training camera of the scene as torch tensors (default: the middle frame)."""
    import torch
    tf = json.loads(run.transforms_json.read_text())
    dp = json.loads(run.dataparser_transforms.read_text()) if run.dataparser_transforms.exists() else {}
    i = len(tf["frames"]) // 2 if index is None else index
    c2w, K = camera_from_transforms(tf, dp, i, width, height)
    return (torch.tensor(c2w, dtype=torch.float32, device=device),
            torch.tensor(K, dtype=torch.float32, device=device))


# ── the trained pipeline ─────────────────────────────────────────────────────────

@contextmanager
def in_workspace(run):
    """ns-train ran with cwd = gsplats/workspace, and config.yml's data/output paths are
    relative to it. Images are read lazily, so callers keep the cwd for the whole job."""
    old = os.getcwd()
    os.chdir(run.workspace_root)
    try:
        yield
    finally:
        os.chdir(old)


def load_pipeline(run, test_mode="test"):
    """nerfstudio eval_setup on the run, images cached on the CPU (keeps VRAM for rendering).
    Must be called inside in_workspace(run). Returns (config, pipeline, checkpoint, step)."""
    from nerfstudio.utils.eval_utils import eval_setup

    def cpu_images(cfg):
        dm = cfg.pipeline.datamanager
        if hasattr(dm, "cache_images"):
            dm.cache_images = "cpu"
        return cfg

    return eval_setup(run.config_yml.resolve(), test_mode=test_mode, update_config_callback=cpu_images)


def train_views(pipeline):
    """(cameras, image paths, dataset) of the training split, in camera-optimizer index order."""
    ds = pipeline.datamanager.train_dataset
    return ds.cameras, list(ds._dataparser_outputs.image_filenames), ds


def optimized_c2w(model, cameras, idx):
    """Refined camera-to-world [3, 4] of training camera `idx` (raw pose if the optimizer is off)."""
    import torch
    cam = cameras[idx:idx + 1].to(model.device)
    if cam.metadata is None:
        cam.metadata = {}
    cam.metadata["cam_idx"] = idx
    with torch.no_grad():
        return model.camera_optimizer.apply_to_camera(cam)[0]


def render_rgb(model, cameras, idx, c2w):
    """Render camera `idx` with an explicit pose (eval mode renders exactly the pose it is given)."""
    cam = cameras[idx:idx + 1].to(model.device)
    cam.camera_to_worlds = c2w[None].to(model.device)
    return model.get_outputs_for_camera(cam)["rgb"]


def gt_rgb(model, dataset, idx):
    img = dataset.get_image_float32(idx)
    if img.shape[-1] == 4:                                  # composite like splatfacto's eval
        bg = model._get_background_color().to(img.device)
        img = img[..., :3] * img[..., 3:] + bg * (1 - img[..., 3:])
    return img


def psnr(a, b):
    import torch
    mse = float(torch.mean((a.float().cpu() - b.float().cpu()) ** 2))
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def correction_stats(correction):
    """Translation (m) and rotation (deg) of pose corrections [N, 3, 4]."""
    import torch
    t = correction[:, :3, 3].norm(dim=-1)
    tr = correction[:, 0, 0] + correction[:, 1, 1] + correction[:, 2, 2]
    ang = torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1.0, 1.0)))
    return {"translation_mean_m": float(t.mean()), "translation_max_m": float(t.max()),
            "rotation_mean_deg": float(ang.mean()), "rotation_max_deg": float(ang.max())}


def spread(n, k):
    """k indices evenly spread over range(n)."""
    k = max(1, min(k, n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _save(path, *imgs):
    from PIL import Image
    row = np.concatenate([np.clip(i.detach().float().cpu().numpy(), 0, 1) for i in imgs], axis=1)
    Image.fromarray((row * 255).round().astype(np.uint8)).save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--views", type=int, default=10, help="training views to render (evenly spread)")
    ap.add_argument("--eval-views", type=int, default=5, help="held-out views, raw pose (model quality reference)")
    ap.add_argument("--save-images", action="store_true",
                    help="write refined | raw | training image strips to <scene>/semantics/<run>/p0_cameras/")
    ap.add_argument("--record", action="store_true", help="also write SousVide/runs/semantics_p0_cameras_*.json")
    a = ap.parse_args(argv)

    from .paths import SemanticsError, find_scene_run, resolve_project_root
    try:
        run = find_scene_run(resolve_project_root(a.project_root), a.scene)
    except SemanticsError as e:
        fail(str(e))
        emit({"ok": False, "error": str(e)})
        return 1

    import torch
    section(f"Camera check — {a.scene} {run.run}")
    out = {"tool": "cameras", "scene": a.scene, "run": run.run, "checkpoint": run.checkpoint.name,
           "started": datetime.now().isoformat(timespec="seconds")}
    with in_workspace(run), VramMonitor(1.0) as mon:
        _, pipeline, ckpt, step = load_pipeline(run)
        model = pipeline.model
        cams, files, ds = train_views(pipeline)
        opt = model.camera_optimizer
        mode = opt.config.mode
        out.update(step=int(step), camera_optimizer=mode, n_train=len(cams))
        ok(f"loaded step {step}: {len(model.means):,} Gaussians, {len(cams)} training cameras, camera optimizer {mode}")
        if mode != "off":
            if opt.num_cameras != len(cams):
                fail(f"camera optimizer has {opt.num_cameras} cameras but the training split has {len(cams)}")
                emit({"ok": False, "error": "camera count mismatch", **out})
                return 1
            with torch.no_grad():
                out["corrections"] = correction_stats(opt.get_correction_matrices().cpu())
            c = out["corrections"]
            info(f"pose corrections: translation mean {c['translation_mean_m'] * 1000:.1f} mm "
                 f"(max {c['translation_max_m'] * 1000:.1f}), rotation mean {c['rotation_mean_deg']:.3f}° "
                 f"(max {c['rotation_max_deg']:.3f}°)")
        else:
            warn("camera optimizer is off: refined and raw poses are identical")

        img_dir = run.semantics_dir / run.run / "p0_cameras"
        if a.save_images:
            img_dir.mkdir(parents=True, exist_ok=True)
        views = []
        for i in spread(len(cams), a.views):
            raw = cams.camera_to_worlds[i]
            refined = optimized_c2w(model, cams, i)
            gt = gt_rgb(model, ds, i)
            r_opt, r_raw = render_rgb(model, cams, i, refined), render_rgb(model, cams, i, raw)
            v = {"index": i, "image": os.path.basename(str(files[i])),
                 "psnr_refined": round(psnr(r_opt, gt), 3), "psnr_raw": round(psnr(r_raw, gt), 3)}
            views.append(v)
            info(f"#{i:<4} {v['image']:<28} refined {v['psnr_refined']:6.2f} dB   raw {v['psnr_raw']:6.2f} dB")
            if a.save_images:
                _save(img_dir / f"{i:04d}.png", r_opt, r_raw, gt)
        out["views"] = views

        held = []
        eds = getattr(pipeline.datamanager, "eval_dataset", None)
        if eds is not None and len(eds) and a.eval_views > 0:
            for i in spread(len(eds), a.eval_views):
                held.append(round(psnr(render_rgb(model, eds.cameras, i, eds.cameras.camera_to_worlds[i]),
                                       gt_rgb(model, eds, i)), 3))
        out["eval_psnr"] = held
    out["peak_device_mib"] = mon.peak

    m_opt = float(np.mean([v["psnr_refined"] for v in views]))
    m_raw = float(np.mean([v["psnr_raw"] for v in views]))
    out.update(psnr_refined_mean=round(m_opt, 3), psnr_raw_mean=round(m_raw, 3),
               eval_psnr_mean=round(float(np.mean(held)), 3) if held else None)
    out["ok"] = bool(np.isfinite(m_opt) and m_opt >= m_raw - 0.05)
    held_txt = f"; held-out (raw pose) {out['eval_psnr_mean']:.2f} dB" if held else ""
    (ok if out["ok"] else fail)(f"training views: refined {m_opt:.2f} dB vs raw {m_raw:.2f} dB "
                                f"({m_opt - m_raw:+.2f} dB){held_txt}")
    if a.save_images:
        info(f"image strips (refined | raw | training image): {img_dir}")
    if a.record:
        run.runs_dir.mkdir(parents=True, exist_ok=True)
        p = run.runs_dir / f"semantics_p0_cameras_{a.scene}_{datetime.now():%Y-%m-%d_%H%M}.json"
        p.write_text(json.dumps(out, indent=2) + "\n")
        info(f"record: {p}")
    emit(out)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
