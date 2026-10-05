"""The per-Gaussian feature table both backends write, and the query worker / UI read.

    gsplats/workspace/<scene>/semantics/<run>/<backend>/
        clip.f16      float16 [n, 512]   unit rows (0 for never-seen Gaussians)
        dino.f16      float16 [n, 384]   unit rows (optional)
        weight.f32    float32 [n]        total blend weight over the training views (0 = unseen)
        geom.f32      float32 [n, 5]     x, y, z (splat frame), opacity, largest scale (m)
        pca_rgb.u8    uint8   [n, 3]     3 principal components of clip, for a feature-colour view
        index.json    shapes, run / checkpoint key, backend, teacher tag, order hash, metrics

Row i is record i of Galley's .splat for the same checkpoint: both come from
course_tools.splat_order(), and index.json carries a hash of that order so a mismatch is refused
instead of colouring the wrong Gaussians. Files are raw little-endian arrays (np.memmap-able);
the directory is written next to the old one and swapped in only when complete.
"""

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

VERSION = 1
GEOM_COLS = ["x", "y", "z", "opacity", "scale_max"]
_DTYPES = {"clip.f16": np.float16, "dino.f16": np.float16, "weight.f32": np.float32,
           "geom.f32": np.float32, "pca_rgb.u8": np.uint8}


def order_hash(order):
    return hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest()[:16]


def pca_rgb(x, sample=100_000, seed=0):
    """uint8 [n, 3]: projections on the top 3 principal components, 1st–99th percentile → 0–255."""
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n == 0:
        return np.zeros((0, 3), np.uint8)
    live = np.linalg.norm(x, axis=1) > 0
    idx = np.flatnonzero(live)
    if len(idx) < 3:
        return np.zeros((n, 3), np.uint8)
    if len(idx) > sample:
        idx = np.random.default_rng(seed).choice(idx, sample, replace=False)
    mu = x[idx].mean(0)
    _, _, vt = np.linalg.svd(x[idx] - mu, full_matrices=False)
    proj = (x - mu) @ vt[:3].T
    lo, hi = np.percentile(proj[live], [1, 99], axis=0)
    out = np.clip((proj - lo) / np.maximum(hi - lo, 1e-8), 0, 1)
    out[~live] = 0.5
    return np.round(out * 255).astype(np.uint8)


def write_table(dest, *, clip, weight, geom, order, meta, dino=None):
    """Write a complete table into `dest` (atomically replacing an old one). Arrays are already
    in .splat order. Returns the index dict."""
    dest = Path(dest)
    n = len(order)
    assert clip.shape[0] == n and weight.shape[0] == n and geom.shape == (n, len(GEOM_COLS)), \
        (clip.shape, weight.shape, geom.shape, n)
    tmp = dest.with_name(dest.name + ".writing")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    arrays = {"clip.f16": clip, "weight.f32": weight, "geom.f32": geom, "pca_rgb.u8": pca_rgb(clip)}
    if dino is not None:
        assert dino.shape[0] == n
        arrays["dino.f16"] = dino
    shapes = {}
    for name, arr in arrays.items():
        a = np.ascontiguousarray(np.asarray(arr).astype(_DTYPES[name]))
        a.tofile(tmp / name)
        shapes[name] = list(a.shape)
    index = {"version": VERSION, "n": int(n), "shapes": shapes, "geom_cols": GEOM_COLS,
             "order_sha": order_hash(order), "created": datetime.now().isoformat(timespec="seconds"), **meta}
    (tmp / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    old = dest.with_name(dest.name + ".old")
    if dest.exists():
        if old.exists():
            shutil.rmtree(old)
        os.replace(dest, old)
    os.replace(tmp, dest)
    if old.exists():
        shutil.rmtree(old)
    return index


@dataclass
class Table:
    path: Path
    index: dict
    clip: np.ndarray
    weight: np.ndarray
    geom: np.ndarray
    pca: np.ndarray
    dino: object = None

    @property
    def n(self):
        return self.index["n"]


def read_table(path, mmap=True):
    path = Path(path)
    index = json.loads((path / "index.json").read_text())

    def arr(name):
        p = path / name
        if name not in index["shapes"] or not p.exists():
            return None
        shape = tuple(index["shapes"][name])
        if mmap:
            return np.memmap(p, dtype=_DTYPES[name], mode="r", shape=shape)
        return np.fromfile(p, dtype=_DTYPES[name]).reshape(shape)

    return Table(path, index, arr("clip.f16"), arr("weight.f32"), arr("geom.f32"), arr("pca_rgb.u8"),
                 arr("dino.f16"))


def check_order(table, order):
    """Refuse a table whose rows do not match this .splat ordering."""
    h = order_hash(order)
    if table.index.get("order_sha") != h:
        raise ValueError(f"{table.path}: rows were written for a different .splat order "
                         f"({table.index.get('order_sha')} ≠ {h}) — re-run the export step")
