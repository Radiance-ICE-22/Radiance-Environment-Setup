"""2D teacher features per training image: a multi-scale CLIP pyramid and dense DINOv2 tokens.

CLIP (LERF's pyramid, averaged across scales as FMGS does): for each scale s (fraction of the
image's short side), square crops of side s·min(H, W) slide with a stride of half a crop; each
crop is resized to 224 px and embedded with OpenCLIP; the unit embeddings of a scale form a grid
over the crop centres. Every scale's grid is resampled (bilinear in crop-centre coordinates) onto
one COMMON grid whose cell (i, j) is centred at ((j + 0.5)·W/gw, (i + 0.5)·H/gh), and the scales
are averaged. Because the common grid tiles the image evenly, upsampling it to any render size
with F.interpolate(align_corners=False) puts every value back on its own pixel position.

DINOv2: the image is resized to a width that is a multiple of 14 (height keeps the aspect, also a
multiple of 14); the normalised patch tokens form the grid, which tiles the image the same way.

Output, per image stem (frame_00001 …), under gsplats/workspace/<scene>/semantics/teachers/<tag>/:
    clip/<stem>.npy   float16 [gh, gw, 512]
    dino/<stem>.npy   float16 [dh, dw, 384]
    meta.json         settings, tag, image size, grids, timing
<tag> hashes every setting that changes the numbers, so a different model or pyramid never mixes
with old files. Finished images are skipped on a re-run (resumable).
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import CLIP_DIM, CLIP_MODEL, CLIP_PRETRAINED, DINO_DIM, DINO_MODEL

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)
TEACHERS_VERSION = 1          # bump when the extraction code changes its output


@dataclass
class TeacherSettings:
    scales: list = field(default_factory=lambda: [round(float(s), 4) for s in np.linspace(0.05, 0.5, 7)])
    stride_frac: float = 0.5          # crop stride as a fraction of the crop side (LERF)
    cell_frac: float = 0.025          # common-grid cell as a fraction of the short side (= finest stride)
    min_tile: int = 16                # px; crops smaller than this are not worth embedding
    dino_width: int = 896             # px, multiple of 14 (64 patches across)
    batch: int = 256                  # CLIP crops per forward pass
    clip_model: str = CLIP_MODEL
    clip_pretrained: str = CLIP_PRETRAINED
    dino_model: str = DINO_MODEL
    teachers: tuple = ("clip", "dino")

    def tag(self):
        key = {k: v for k, v in asdict(self).items() if k not in ("batch",)}
        key["version"] = TEACHERS_VERSION
        key["teachers"] = list(self.teachers)
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


# ── geometry of the pyramid ──────────────────────────────────────────────────────

def crop_starts(size, tile, stride):
    """Start offsets so crops of `tile` px cover [0, size) with the given stride, the last one
    flush with the far edge."""
    if tile >= size:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def common_grid(H, W, cell_frac):
    cell = max(1.0, cell_frac * min(H, W))
    gh, gw = max(1, round(H / cell)), max(1, round(W / cell))
    return gh, gw


def resample(grid, centers_y, centers_x, H, W, gh, gw):
    """Bilinearly resample a grid [ny, nx, C] sampled at crop centres (px) onto the common grid
    [gh, gw, C] (cell centres at (i + 0.5)·H/gh). Positions outside the centre range clamp to
    the nearest edge value."""
    import torch
    import torch.nn.functional as F
    ny, nx = grid.shape[:2]
    qy = (np.arange(gh) + 0.5) * H / gh
    qx = (np.arange(gw) + 0.5) * W / gw
    fy = np.interp(qy, centers_y, np.arange(ny))        # fractional index into the scale grid
    fx = np.interp(qx, centers_x, np.arange(nx))
    uy = 2 * fy / (ny - 1) - 1 if ny > 1 else np.zeros_like(fy)
    ux = 2 * fx / (nx - 1) - 1 if nx > 1 else np.zeros_like(fx)
    gy, gx = np.meshgrid(uy, ux, indexing="ij")
    samp = torch.tensor(np.stack([gx, gy], -1)[None], dtype=grid.dtype, device=grid.device)  # [1, gh, gw, 2]
    out = F.grid_sample(grid.permute(2, 0, 1)[None], samp, mode="bilinear", align_corners=True,
                        padding_mode="border")
    return out[0].permute(1, 2, 0)                          # [gh, gw, C]


def clip_pyramid(encode, img, s):
    """img: [3, H, W] float in [0, 1] on the model's device. encode(batch [B, 3, 224, 224],
    normalised) → [B, D]. Returns [gh, gw, D] float32 (average of unit embeddings) and the
    number of crops embedded."""
    import torch
    import torch.nn.functional as F
    _, H, W = img.shape
    gh, gw = common_grid(H, W, s.cell_frac)
    mean = torch.tensor(CLIP_MEAN, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=img.device).view(1, 3, 1, 1)
    acc, used, n_crops = None, 0, 0
    for sc in s.scales:
        tile = int(round(sc * min(H, W)))
        if tile < s.min_tile:
            continue
        stride = max(1, int(round(tile * s.stride_frac)))
        ys, xs = crop_starts(H, tile, stride), crop_starts(W, tile, stride)
        pos = [(y, x) for y in ys for x in xs]
        embs = []
        for b in range(0, len(pos), s.batch):
            crops = torch.stack([img[:, y:y + tile, x:x + tile] for y, x in pos[b:b + s.batch]])
            crops = F.interpolate(crops, size=(224, 224), mode="bilinear", align_corners=False,
                                  antialias=tile > 224)
            with torch.no_grad():
                e = encode((crops - mean) / std).float()
            embs.append(e / e.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        grid = torch.cat(embs).reshape(len(ys), len(xs), -1)
        cy = np.asarray(ys, dtype=float) + tile / 2
        cx = np.asarray(xs, dtype=float) + tile / 2
        g = resample(grid, cy, cx, H, W, gh, gw)
        acc = g if acc is None else acc + g
        used += 1
        n_crops += len(pos)
    if acc is None:
        raise ValueError(f"image {W}x{H} is too small for every pyramid scale (min_tile {s.min_tile})")
    return acc / used, n_crops


def dino_dense(forward_tokens, img, s):
    """img: [3, H, W] in [0, 1]. forward_tokens(x [1, 3, h, w]) → patch tokens [1, h·w/196, D].
    Returns [h/14, w/14, D] float32."""
    import torch
    import torch.nn.functional as F
    _, H, W = img.shape
    w = max(14, int(round(s.dino_width / 14)) * 14)
    h = max(14, int(round(H * w / W / 14)) * 14)
    x = F.interpolate(img[None], size=(h, w), mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(IMNET_MEAN, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(IMNET_STD, device=img.device).view(1, 3, 1, 1)
    with torch.no_grad():
        t = forward_tokens((x - mean) / std).float()
    return t.reshape(h // 14, w // 14, -1)


# ── models and the extraction loop ───────────────────────────────────────────────

def default_encoders(device, s):
    """(clip_encode, dino_tokens) from the pinned models, fp16 autocast on CUDA."""
    import torch
    from .models import load_clip, load_dino
    out = {}
    if "clip" in s.teachers:
        clip, _, _ = load_clip(device)

        def clip_encode(x):
            with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
                return clip.encode_image(x)
        out["clip"] = clip_encode
    if "dino" in s.teachers:
        dino = load_dino(device)

        def dino_tokens(x):
            with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
                return dino.forward_features(x)["x_norm_patchtokens"]
        out["dino"] = dino_tokens
    return out


def frames_of(transforms_json):
    """(stem, path) of every frame in transforms.json, extraction order; paths relative to the scene."""
    tf = json.loads(Path(transforms_json).read_text())
    frames = sorted(tf["frames"], key=lambda f: f.get("file_path", ""))
    return [(Path(f["file_path"]).stem, f["file_path"]) for f in frames]


def load_image(path, device):
    import torch
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(im).permute(2, 0, 1).contiguous().to(device)


def _save(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.npy")
    np.save(tmp, arr.astype(np.float16))
    tmp.replace(path)


def extract(scene_dir, out_root, settings=None, device="cuda", encoders=None, log=print, limit=None):
    """Teacher features for every frame of <scene_dir>/transforms.json into out_root/<tag>/.
    Returns the meta dict (also written as meta.json). `encoders` overrides the models (tests)."""
    s = settings or TeacherSettings()
    tag = s.tag()
    out = Path(out_root) / tag
    out.mkdir(parents=True, exist_ok=True)
    frames = frames_of(Path(scene_dir) / "transforms.json")
    if limit:
        frames = frames[:limit]
    enc = encoders or default_encoders(device, s)
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta.update(tag=tag, settings=asdict(s), clip_dim=CLIP_DIM, dino_dim=DINO_DIM, n_frames=len(frames),
                images=meta.get("images", {}))
    meta["settings"]["teachers"] = list(s.teachers)
    t_all, done, skipped, crops_total = time.time(), 0, 0, 0
    for k, (stem, rel) in enumerate(frames):
        need = [t for t in s.teachers if not (out / t / f"{stem}.npy").exists()]
        if not need:
            skipped += 1
            continue
        t0 = time.time()
        img = load_image(Path(scene_dir) / rel, device)
        rec = meta["images"].get(stem, {})
        rec["size"] = [int(img.shape[2]), int(img.shape[1])]
        if "clip" in need:
            g, nc = clip_pyramid(enc["clip"], img, s)
            _save(out / "clip" / f"{stem}.npy", g.cpu().numpy())
            rec["clip_grid"], rec["crops"] = list(g.shape[:2]), nc
            crops_total += nc
        if "dino" in need:
            d = dino_dense(enc["dino"], img, s)
            _save(out / "dino" / f"{stem}.npy", d.cpu().numpy())
            rec["dino_grid"] = list(d.shape[:2])
        rec["seconds"] = round(time.time() - t0, 2)
        meta["images"][stem] = rec
        done += 1
        if done == 1 or done % 10 == 0 or k == len(frames) - 1:
            log(f"teachers {k + 1}/{len(frames)} {stem}: {rec['seconds']} s"
                + (f", {rec.get('crops')} CLIP crops" if "clip" in need else ""))
            meta_p.write_text(json.dumps(meta, indent=1) + "\n")
    size = sum(p.stat().st_size for p in out.rglob("*.npy"))
    meta["stats"] = {"done": done, "skipped": skipped, "seconds": round(time.time() - t_all, 1),
                     "crops": crops_total, "mb": round(size / 2 ** 20, 1)}
    meta_p.write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def load_maps(teacher_dir, stem):
    """{teacher: float32 array [gh, gw, C]} for one image stem."""
    out = {}
    for t in ("clip", "dino"):
        p = Path(teacher_dir) / t / f"{stem}.npy"
        if p.exists():
            out[t] = np.load(p).astype(np.float32)
    return out


def estimate_crops(H, W, s=None):
    """CLIP crops per image for these settings (the dominant cost)."""
    s = s or TeacherSettings()
    n = 0
    for sc in s.scales:
        tile = int(round(sc * min(H, W)))
        if tile < s.min_tile:
            continue
        stride = max(1, int(round(tile * s.stride_frac)))
        n += len(crop_starts(H, tile, stride)) * len(crop_starts(W, tile, stride))
    return n


def grid_bytes(H, W, s=None):
    s = s or TeacherSettings()
    gh, gw = common_grid(H, W, s.cell_frac)
    w = max(14, int(round(s.dino_width / 14)) * 14)
    h = max(14, int(round(H * w / W / 14)) * 14)
    return gh * gw * CLIP_DIM * 2 + (h // 14) * (w // 14) * DINO_DIM * 2


__all__ = ["TeacherSettings", "extract", "load_maps", "frames_of", "clip_pyramid", "dino_dense",
           "resample", "common_grid", "crop_starts", "estimate_crops", "grid_bytes"]
