"""Bake the trained field into per-Gaussian rows: evaluate it at every centre, in .splat order.

The table the lift writes and the one FMGS writes are the same format (store.py), so the query
worker, the splat editor and the evaluation read either. Rows are unit vectors (cosine-ready);
`weight` marks Gaussians no training camera saw (0): the field extrapolates there, and the query
treats such rows as unseen exactly as it does for the lift.
"""

import numpy as np
import torch

from ..lift import normalise_rows


@torch.no_grad()
def bake(field, xyz, batch=65536, device=None, log=print, with_scales=False):
    """xyz [n, 3] (splat frame, any order) → (clip [n, 512], dino [n, 384]) float32 unit rows.
    Variant A (clip_groups > 1): 'clip' is the mean of the groups' outputs; with_scales also returns
    clip_scales [n, groups, 512] (unit rows per group), else None."""
    dev = torch.device(device) if device else next(field.parameters()).device
    x = torch.as_tensor(np.asarray(xyz, np.float32))
    G = field.cfg.clip_groups
    clip = np.zeros((len(x), field.cfg.clip_dim), np.float32)
    dino = np.zeros((len(x), field.cfg.dino_dim), np.float32)
    scales = np.zeros((len(x), G, field.cfg.clip_dim), np.float32) if G > 1 else None
    for i in range(0, len(x), batch):
        e = field.encode(x[i:i + batch].to(dev))
        dino[i:i + batch] = field.dino_head(e).float().cpu().numpy()
        if G > 1:
            for g in range(G):
                scales[i:i + batch, g] = field.clip_out(e, g).cpu().numpy()
            clip[i:i + batch] = scales[i:i + batch].mean(1)
        else:
            clip[i:i + batch] = field.clip_out(e).cpu().numpy()
    if scales is not None:
        scales = np.stack([normalise_rows(scales[:, g]) for g in range(G)], 1)
    out = (normalise_rows(clip), normalise_rows(dino))
    return out + (scales,) if with_scales else out
