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


LANGSPLAT_LEVELS = (1, 2, 3)          # LangSplat seg-map rows s, m, l (row 0 'default' is not lifted, as in Occam's LGS)


def seg_to_render(seg, W, H):
    """LangSplat seg maps [L, Hs, Ws] (mask ids, -1 = none) → [L, H, W] by nearest sampling at the render's
    pixel centres (ids cannot be interpolated)."""
    Hs, Ws = seg.shape[1:]
    iy = np.clip(((np.arange(H) + 0.5) * Hs / H).astype(np.int64), 0, Hs - 1)
    ix = np.clip(((np.arange(W) + 0.5) * Ws / W).astype(np.int64), 0, Ws - 1)
    return seg[:, iy][:, :, ix]


def lift_segment_views(g, views, segs_for, feat_width=960, backend="gsplat", uncovered="skip", fill_for=None,
                       dino_for=None, chunk=GRAD_CHUNK, acc_device=None, log=print):
    """Lift region-level (LangSplat) CLIP features per level, plus optional DINO grids.

    segs_for(stem) → (seg [L, Hs, Ws] int, feats [n, C]) or None: per pixel of level l, the feature of the mask
    covering it. Pixels no mask covers are handled by `uncovered`:
      skip — they contribute nothing: f_i^l = Σ w·m_l·F_l / Σ w·m_l (LangSplat / Occam's LGS);
      fill — they take fill_for(stem), a teacher grid [gh, gw, C] upsampled like any teacher (our B2's choice);
             f_i^l = Σ w·F̃_l / Σ w over every pixel.
    Same all-pixel weighted mean as lift_views otherwise (gradient of Σ⟨render(f), F⟩, 32-channel chunks).
    Returns ({clip_s0.., dino}, weight [N] (all-pixel), stats); rows a level never covers are 0 in that level.
    """
    import torch
    if uncovered not in ("skip", "fill"):
        raise ValueError(f"uncovered must be skip or fill, not {uncovered!r}")
    if uncovered == "fill" and fill_for is None:
        raise ValueError("uncovered=fill needs fill_for (the pyramid teacher grids)")
    dev = g.means.device
    N = len(g)
    acc_device = acc_device or dev
    nl = len(LANGSPLAT_LEVELS)
    acc, wl = [None] * nl, torch.zeros(nl, N, device=acc_device)
    weight, dacc = torch.zeros(N, device=acc_device), None
    covered = []
    t0, n_pass, used = time.time(), 0, 0
    for k, v in enumerate(views):
        sf = segs_for(v.stem)
        if sf is None:
            log(f"  ! no LangSplat features for {v.stem}; skipped")
            continue
        seg, feats = sf
        used += 1
        W, H = render_size(v, feat_width)
        vm, K = viewmat_from_c2w(v.c2w.to(dev).float()), view_K(v, W, H).to(dev)
        ones = torch.ones(N, 1, device=dev, requires_grad=True)
        img, _ = render_features(g, ones, vm, K, W, H, backend)
        img.sum().backward()
        weight += ones.grad[:, 0].to(acc_device)
        n_pass += 1
        del ones, img
        ids_all = torch.as_tensor(seg_to_render(seg, W, H), device=dev).long()          # [L, H, W]
        Ft = torch.as_tensor(feats, dtype=torch.float32, device=dev)
        C = Ft.shape[1]
        fill = fill_for(v.stem) if uncovered == "fill" else None
        cov = []
        for li in range(nl):
            ids = ids_all[li]
            m = ids >= 0
            cov.append(float(m.float().mean()))
            if uncovered == "skip":
                one = torch.ones(N, 1, device=dev, requires_grad=True)
                img, _ = render_features(g, one, vm, K, W, H, backend)
                (img[..., 0] * m).sum().backward()
                wl[li] += one.grad[:, 0].to(acc_device)
                n_pass += 1
                del one, img
            if acc[li] is None:
                acc[li] = torch.zeros(N, C, device=acc_device)
            safe = ids.clamp_min(0)
            for c0 in range(0, C, chunk):
                c1 = min(C, c0 + chunk)
                F2D = Ft[safe, c0:c1]                                                    # [H, W, c]
                if uncovered == "skip":
                    F2D = F2D * m[..., None]
                else:
                    F2D = torch.where(m[..., None], F2D, upsample(fill[..., c0:c1], W, H, dev))
                f = torch.zeros(N, c1 - c0, device=dev, requires_grad=True)
                out, _ = render_features(g, f, vm, K, W, H, backend)
                (out * F2D).sum().backward()
                acc[li][:, c0:c1] += f.grad.to(acc_device)
                n_pass += 1
                del f, out, F2D
        covered.append(cov)
        if dino_for is not None:
            grid = dino_for(v.stem)
            if grid is not None:
                D = grid.shape[-1]
                if dacc is None:
                    dacc = torch.zeros(N, D, device=acc_device)
                for c0 in range(0, D, chunk):
                    c1 = min(D, c0 + chunk)
                    F2D = upsample(grid[..., c0:c1], W, H, dev)
                    f = torch.zeros(N, c1 - c0, device=dev, requires_grad=True)
                    out, _ = render_features(g, f, vm, K, W, H, backend)
                    (out * F2D).sum().backward()
                    dacc[:, c0:c1] += f.grad.to(acc_device)
                    n_pass += 1
                    del f, out, F2D
        if k == 0 or (k + 1) % 25 == 0 or k == len(views) - 1:
            log(f"  segment lift {k + 1}/{len(views)} views, {time.time() - t0:.0f} s")
    seen = weight > EPS
    feats = {}
    for li in range(nl):
        den = wl[li] if uncovered == "skip" else weight
        ok = den > EPS
        a = acc[li] / den.clamp_min(EPS)[:, None]
        a[~ok] = 0
        feats[f"clip_s{li}"] = a.float().cpu().numpy()
    if dacc is not None:
        a = dacc / weight.clamp_min(EPS)[:, None]
        a[~seen] = 0
        feats["dino"] = a.float().cpu().numpy()
    weight[~seen] = 0
    cov = np.asarray(covered) if covered else np.zeros((0, nl))
    stats = {"views": used, "passes": n_pass, "seconds": round(time.time() - t0, 1), "seen": int(seen.sum()), "n": N,
             "seen_frac": round(float(seen.float().mean()), 4), "render_width": int(feat_width), "uncovered": uncovered,
             "level_rows": [int((wl[li] > EPS).sum()) if uncovered == "skip" else int(seen.sum()) for li in range(nl)],
             "pixel_coverage": [round(float(x), 4) for x in cov.mean(0)] if len(cov) else None}
    return feats, weight.float().cpu().numpy(), stats


def lift_segments_run(run, teacher_dir, uncovered="skip", fill_dir=None, feat_width=960, backend="gsplat", log=print,
                      acc_budget_gb=2.5, limit=None):
    """lift_run for LangSplat teachers (teacher_dir/langsplat/<stem>_{s,f}.npy, teacher_dir/dino/<stem>.npy)."""
    from pathlib import Path
    import torch
    from .cameras import in_workspace, load_pipeline, train_views
    with in_workspace(run):
        _, pipeline, _, step = load_pipeline(run)
        model = pipeline.model
        cams, files, _ = train_views(pipeline)
        views = views_from_pipeline(model, cams, files, log=log)
        if limit:
            views = views[:limit]
        g = gaussians_from_model(model)
        del pipeline
    dev = g.means.device
    ls, dd = Path(teacher_dir) / "langsplat", Path(teacher_dir) / "dino"

    def segs_for(stem):
        ps, pf = ls / f"{stem}_s.npy", ls / f"{stem}_f.npy"
        if not (ps.exists() and pf.exists()):
            return None
        return np.load(ps)[list(LANGSPLAT_LEVELS)].astype(np.int64), np.load(pf).astype(np.float32)

    def dino_for(stem):
        p = dd / f"{stem}.npy"
        return np.load(p).astype(np.float32) if p.exists() else None

    def fill_for(stem):
        return np.load(Path(fill_dir) / "clip" / f"{stem}.npy").astype(np.float32)

    need_gb = len(g) * (len(LANGSPLAT_LEVELS) * 512 + 384 + 4) * 4 / 2 ** 30
    acc_device = dev if (dev.type == "cpu" or need_gb <= acc_budget_gb) else torch.device("cpu")
    log(f"  {len(g):,} Gaussians, {len(views)} training views, LangSplat levels s/m/l, uncovered pixels: {uncovered}, "
        f"render {feat_width} px wide, accumulators {need_gb:.2f} GB on {acc_device}")
    feats, w, stats = lift_segment_views(g, views, segs_for, feat_width, backend, uncovered,
                                         fill_for if uncovered == "fill" else None,
                                         dino_for if dd.is_dir() else None, acc_device=acc_device, log=log)
    stats.update(step=int(step), acc_device=str(acc_device))
    return feats, w, stats, g


def blend_weights(g, views, feat_width=960, backend="gsplat", log=print):
    """Total alpha-blending weight of every Gaussian over `views` (the lift's denominator; 0 = never
    seen). The FMGS bake uses it so both tables agree on which Gaussians the cameras saw."""
    import torch
    dev = g.means.device
    weight = torch.zeros(len(g), device=dev)
    for k, v in enumerate(views):
        W, H = render_size(v, feat_width)
        vm, K = viewmat_from_c2w(v.c2w.to(dev).float()), view_K(v, W, H).to(dev)
        ones = torch.ones(len(g), 1, device=dev, requires_grad=True)
        img, _ = render_features(g, ones, vm, K, W, H, backend)
        img.sum().backward()
        weight += ones.grad[:, 0]
        del ones, img
    weight[weight <= EPS] = 0
    return weight.float().cpu().numpy()


def normalise_rows(x):
    """Unit rows; zero rows stay zero."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.where(n > EPS, x / np.maximum(n, EPS), 0.0).astype(np.float32)


def views_from_pipeline(model, cameras, files, log=None):
    """View list for every training camera, with refined poses; SfM pose outliers are dropped
    (query.pose_inliers — none on backroom, 3 of 600 on GTN_lab_v1)."""
    from pathlib import Path
    from .cameras import optimized_c2w
    from .query import pose_inliers
    out = []
    for i in range(len(cameras)):
        c = cameras[i:i + 1]
        out.append(View(stem=Path(str(files[i])).stem, c2w=optimized_c2w(model, cameras, i).detach(),
                        fx=float(c.fx.reshape(-1)[0]), fy=float(c.fy.reshape(-1)[0]),
                        cx=float(c.cx.reshape(-1)[0]), cy=float(c.cy.reshape(-1)[0]),
                        width=int(c.width.reshape(-1)[0]), height=int(c.height.reshape(-1)[0])))
    keep = pose_inliers(np.array([v.c2w[:3, 3].cpu().numpy() for v in out]))
    if log and not keep.all():
        log(f"  dropped {int((~keep).sum())} training camera(s) as SfM pose outliers: "
            + ", ".join(v.stem for v, k in zip(out, keep) if not k))
    return [v for v, k in zip(out, keep) if k]


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
    from pathlib import Path
    import torch
    from .cameras import in_workspace, load_pipeline, train_views
    from .teachers import load_maps
    with in_workspace(run):
        _, pipeline, _, step = load_pipeline(run)
        model = pipeline.model
        cams, files, _ = train_views(pipeline)
        views = views_from_pipeline(model, cams, files, log=log)
        if limit:
            views = views[:limit]
        g = gaussians_from_model(model)
        del pipeline
    dev = g.means.device
    total_c = 0
    groups = sorted(p.name for p in Path(teacher_dir).glob("clip_s*") if p.is_dir())
    names = (["dino"] + groups) if groups else ["clip", "dino"]      # variant A: 'clip' follows from the groups
    for t in names:
        p = next(iter((teacher_dir / t).glob("*.npy")), None) if (teacher_dir / t).is_dir() else None
        if p is not None:
            total_c += int(np.load(p, mmap_mode="r").shape[-1])
    need_gb = len(g) * (total_c + 1) * 4 / 2 ** 30
    acc_device = dev if (dev.type == "cpu" or need_gb <= acc_budget_gb) else torch.device("cpu")
    log(f"  {len(g):,} Gaussians, {len(views)} training views, render {feat_width} px wide, "
        f"accumulators {need_gb:.2f} GB on {acc_device}")
    def maps_for(stem):
        m = load_maps(teacher_dir, stem)
        if groups:                         # the lift is linear: lifting the mean = the mean of the lifted groups
            m.pop("clip", None)
        return m
    feats, w, stats = lift_views(g, views, maps_for, feat_width, backend,
                                 acc_device=acc_device, log=log)
    stats.update(step=int(step), acc_device=str(acc_device))
    return feats, w, stats, g
