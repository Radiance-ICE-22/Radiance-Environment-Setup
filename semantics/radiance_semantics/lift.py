"""Lift backend: per-Gaussian features as the blend-weighted average of 2D teacher features.

For Gaussian i and teacher map F (upsampled to the render size), over all training views v and
pixels p:

    f_i = sum_{v,p} w_i(v,p) F_v(p)  /  sum_{v,p} w_i(v,p)

where w_i(v,p) is i's alpha-blending weight at p in view v (Occam's LGS / LUDVIG). Rendering is
linear in the features, so the numerator is the gradient of sum_p <render(f), F_v> at f = 0 and
the denominator the gradient of sum_p render(1) — two backward passes, no training (probe.py
verified these identities on the installed gsplat, 5 Oct). Features go through gsplat in chunks
of render.GRAD_CHUNK (32) channels, its backward kernel's limit.

Views use the REFINED training poses (cameras.optimized_c2w): on backroom they render 4.2 dB
closer to the training images than transforms.json's poses.
"""

import time
from dataclasses import dataclass

import numpy as np

from .render import GRAD_CHUNK, Gaussians, render_features, viewmat_from_c2w

EPS = 1e-8


@dataclass
class View:
    stem: str            # training image stem (frame_00001), keys the teacher maps
    c2w: object          # torch [3, 4] refined camera-to-world (nerfstudio / OpenGL)
    fx: float            # intrinsics at the image's own resolution
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def view_K(v, W, H):
    """Intrinsics of view v rescaled to a W × H render."""
    import torch
    sx, sy = W / v.width, H / v.height
    return torch.tensor([[v.fx * sx, 0.0, v.cx * sx], [0.0, v.fy * sy, v.cy * sy], [0.0, 0.0, 1.0]],
                        device=v.c2w.device)


def render_size(v, feat_width):
    """Render resolution for the lift: `feat_width` wide, the view's aspect (≥ 1 px)."""
    W = int(feat_width)
    H = max(1, int(round(v.height * W / v.width)))
    return W, H


def upsample(grid, W, H, device):
    """Teacher grid [gh, gw, C] (cells tiling the image) → [H, W, C] at the render size."""
    import torch
    import torch.nn.functional as F
    t = torch.as_tensor(grid, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
    return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).contiguous()


def lift_views(g, views, maps_for, feat_width=960, backend="gsplat", chunk=GRAD_CHUNK, acc_device=None, log=print):
    """Lift teacher features onto Gaussians g over `views`.

    maps_for(stem) → {teacher: array [gh, gw, C]} (missing teachers are skipped for that view).
    Returns ({teacher: float32 [N, C] weighted averages, unnormalised}, weight float32 [N],
    stats). Rows of never-seen Gaussians are 0 with weight 0.
    """
    import torch
    dev = g.means.device
    N = len(g)
    acc_device = acc_device or dev
    acc, weight = {}, torch.zeros(N, device=acc_device)
    t0, n_pass = time.time(), 0
    for k, v in enumerate(views):
        maps = maps_for(v.stem)
        if not maps:
            log(f"  ! no teacher maps for {v.stem}; skipped")
            continue
        W, H = render_size(v, feat_width)
        vm, K = viewmat_from_c2w(v.c2w.to(dev).float()), view_K(v, W, H).to(dev)

        ones = torch.ones(N, 1, device=dev, requires_grad=True)
        img, _ = render_features(g, ones, vm, K, W, H, backend)
        img.sum().backward()
        weight += ones.grad[:, 0].to(acc_device)
        n_pass += 1
        del ones, img

        for name, grid in maps.items():
            C = grid.shape[-1]
            if name not in acc:
                acc[name] = torch.zeros(N, C, device=acc_device)
            for c0 in range(0, C, chunk):
                c1 = min(C, c0 + chunk)
                # upsample only this chunk: the whole 512-channel CLIP map at 960x540 is ~1 GB of float32,
                # which put the lift's peak at 7.6 of 8 GB on backroom (5 Oct)
                F2D = upsample(grid[..., c0:c1], W, H, dev)
                f = torch.zeros(N, c1 - c0, device=dev, requires_grad=True)
                out, _ = render_features(g, f, vm, K, W, H, backend)
                (out * F2D).sum().backward()
                acc[name][:, c0:c1] += f.grad.to(acc_device)
                n_pass += 1
                del f, out, F2D
        if k == 0 or (k + 1) % 25 == 0 or k == len(views) - 1:
            log(f"  lift {k + 1}/{len(views)} views, {time.time() - t0:.0f} s")
    seen = weight > EPS
    feats = {}
    for name, a in acc.items():
        a = a / weight.clamp_min(EPS)[:, None]
        a[~seen] = 0
        feats[name] = a.float().cpu().numpy()
    weight[~seen] = 0                         # one definition of "unseen" everywhere: weight exactly 0
    w = weight.float().cpu().numpy()
    stats = {"views": len(views), "passes": n_pass, "seconds": round(time.time() - t0, 1),
             "seen": int(seen.sum()), "n": N, "seen_frac": round(float(seen.float().mean()), 4),
             "render_width": int(feat_width)}
    return feats, w, stats


def normalise_rows(x):
    """Unit rows; zero rows stay zero."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.where(n > EPS, x / np.maximum(n, EPS), 0.0).astype(np.float32)


def views_from_pipeline(model, cameras, files):
    """View list for every training camera, with refined poses."""
    from pathlib import Path
    from .cameras import optimized_c2w
    out = []
    for i in range(len(cameras)):
        c = cameras[i:i + 1]
        out.append(View(stem=Path(str(files[i])).stem, c2w=optimized_c2w(model, cameras, i).detach(),
                        fx=float(c.fx.reshape(-1)[0]), fy=float(c.fy.reshape(-1)[0]),
                        cx=float(c.cx.reshape(-1)[0]), cy=float(c.cy.reshape(-1)[0]),
                        width=int(c.width.reshape(-1)[0]), height=int(c.height.reshape(-1)[0])))
    return out


def gaussians_from_model(model):
    """Activated, detached Gaussians of a loaded splatfacto model (the frozen geometry)."""
    import torch
    with torch.no_grad():
        q = model.quats.detach()
        return Gaussians(model.means.detach().float(), (q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)).float(),
                         torch.exp(model.scales.detach()).float(), torch.sigmoid(model.opacities.detach()).reshape(-1).float())


def lift_run(run, teacher_dir, feat_width=960, backend="gsplat", log=print, acc_budget_gb=2.5, limit=None):
    """Load the trained run (refined poses) and lift every training view's teacher maps.
    Returns (feats {teacher: [N_total, C]}, weight [N_total], stats); rows are in checkpoint order."""
    import torch
    from .cameras import in_workspace, load_pipeline, train_views
    from .teachers import load_maps
    with in_workspace(run):
        _, pipeline, _, step = load_pipeline(run)
        model = pipeline.model
        cams, files, _ = train_views(pipeline)
        views = views_from_pipeline(model, cams, files)
        if limit:
            views = views[:limit]
        g = gaussians_from_model(model)
        del pipeline
    dev = g.means.device
    total_c = 0
    for t in ("clip", "dino"):
        p = next(iter((teacher_dir / t).glob("*.npy")), None) if (teacher_dir / t).is_dir() else None
        if p is not None:
            total_c += int(np.load(p, mmap_mode="r").shape[-1])
    need_gb = len(g) * (total_c + 1) * 4 / 2 ** 30
    acc_device = dev if (dev.type == "cpu" or need_gb <= acc_budget_gb) else torch.device("cpu")
    log(f"  {len(g):,} Gaussians, {len(views)} training views, render {feat_width} px wide, "
        f"accumulators {need_gb:.2f} GB on {acc_device}")
    feats, w, stats = lift_views(g, views, lambda stem: load_maps(teacher_dir, stem), feat_width, backend,
                                 acc_device=acc_device, log=log)
    stats.update(step=int(step), acc_device=str(acc_device))
    return feats, w, stats, g
