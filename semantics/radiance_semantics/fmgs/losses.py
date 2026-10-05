"""FMGS training losses, on rendered feature maps against the Phase 1 teacher maps.

    L = w_clip · Huber_δ(clip_render, clip_teacher) + w_dino · ‖dino_render − dino_teacher‖² + w_pa · L_pa

Defaults (FMGS): w_clip 0.2, δ 1.25, w_dino 0.8, w_pa 0.01. Each term is a mean over the valid pixels
(rendered alpha ≥ alpha_min — pixels the Gaussians actually cover) and channels.

Pixel alignment (our reading of FMGS's term; recorded in docs/SEMANTICS.md): the rendered CLIP map
should relate a pixel to its neighbours the way DINO does. For sampled pixels p and neighbours q in
a k × k window (dilation `dil`), with cos(·,·) on unit vectors,

    L_pa = mean |cos(clip_p, clip_q) − cos(dino_p, dino_q)|

DINO here is the TEACHER map (fixed), so the term only shapes CLIP; both sides are compared on the
same pixel pairs. Pairs whose neighbour falls outside the image or the valid mask are dropped.
"""

import torch
import torch.nn.functional as F


def masked_mean(x, mask):
    """Mean of x [H, W, C] over pixels where mask [H, W] is true (0 when none are)."""
    m = mask.to(x.dtype)[..., None]
    n = m.sum() * x.shape[-1]
    return (x * m).sum() / n.clamp_min(1.0)


def clip_loss(pred, target, mask, delta=1.25):
    return masked_mean(F.huber_loss(pred, target, reduction="none", delta=delta), mask)


def dino_loss(pred, target, mask):
    return masked_mean((pred - target) ** 2, mask)


def neighbour_offsets(k=3, dil=1):
    r = k // 2
    return [(dy * dil, dx * dil) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if (dy, dx) != (0, 0)]


def pixel_alignment(clip, dino, mask, samples=4096, k=3, dil=2, generator=None):
    """|cos_clip − cos_dino| over sampled pixels and their k×k neighbours (see the module docstring)."""
    H, W, _ = clip.shape
    valid = torch.nonzero(mask.reshape(-1), as_tuple=False)[:, 0]
    if len(valid) == 0:
        return clip.sum() * 0.0
    pick = valid[torch.randint(len(valid), (min(samples, len(valid)),), generator=generator,
                               device="cpu").to(valid.device)]
    py, px = pick // W, pick % W
    cn, dn = F.normalize(clip, dim=-1), F.normalize(dino, dim=-1)
    terms = []
    for dy, dx in neighbour_offsets(k, dil):
        qy, qx = py + dy, px + dx
        ok = (qy >= 0) & (qy < H) & (qx >= 0) & (qx < W)
        if not ok.any():
            continue
        ay, ax, by, bx = py[ok], px[ok], qy[ok], qx[ok]
        ok2 = mask[by, bx]
        ay, ax, by, bx = ay[ok2], ax[ok2], by[ok2], bx[ok2]
        if len(ay) == 0:
            continue
        sc = (cn[ay, ax] * cn[by, bx]).sum(-1)
        sd = (dn[ay, ax] * dn[by, bx]).sum(-1)
        terms.append((sc - sd).abs())
    if not terms:
        return clip.sum() * 0.0
    return torch.cat(terms).mean()


def total_loss(clip_pred, clip_tgt, dino_pred, dino_tgt, mask, w_clip=0.2, w_dino=0.8, w_pa=0.01, delta=1.25,
               pa_samples=4096, generator=None):
    """{"loss", "clip", "dino", "pa"}; DINO terms are skipped when there is no DINO prediction/target."""
    out = {"clip": clip_loss(clip_pred, clip_tgt, mask, delta)}
    loss = w_clip * out["clip"]
    if dino_pred is not None and dino_tgt is not None:
        out["dino"] = dino_loss(dino_pred, dino_tgt, mask)
        loss = loss + w_dino * out["dino"]
    if w_pa > 0 and dino_tgt is not None:
        out["pa"] = pixel_alignment(clip_pred, dino_tgt, mask, pa_samples, generator=generator)
        loss = loss + w_pa * out["pa"]
    out["loss"] = loss
    return out
