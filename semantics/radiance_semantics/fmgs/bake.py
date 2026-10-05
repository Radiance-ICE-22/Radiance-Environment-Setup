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
def bake(field, xyz, batch=65536, device=None, log=print):
    """xyz [n, 3] (splat frame, any order) → (clip [n, 512], dino [n, 384]) float32 unit rows."""
    dev = torch.device(device) if device else next(field.parameters()).device
    x = torch.as_tensor(np.asarray(xyz, np.float32))
    clip = np.zeros((len(x), field.cfg.clip_dim), np.float32)
    dino = np.zeros((len(x), field.cfg.dino_dim), np.float32)
    for i in range(0, len(x), batch):
        out = field(x[i:i + batch].to(dev))
        clip[i:i + batch] = out["clip"].float().cpu().numpy()
        dino[i:i + batch] = out["dino"].float().cpu().numpy()
    return normalise_rows(clip), normalise_rows(dino)
