"""FeatureField: Instant-NGP multi-resolution hash grid over the scene, two MLP heads.

Input: Gaussian centres in the splat (world) frame, normalised into [0, 1]³ by the scene box
(1st–99th percentile of the centres, padded 5 %; points outside are clamped onto it).
Encoding: L levels × F features, resolutions growing geometrically from `base` to `finest`, each
level a dense grid when it fits in 2^log2_table entries and a hashed one otherwise → L·F values.
Heads: CLIP (512) and DINO (384), each `layers` hidden layers of `hidden` ReLU units.

Defaults are the FMGS configuration (24 levels, 16 → 512, 2^20, 8 features; 192-d encoding).

Two implementations of each part (the encoding, the heads), chosen separately as "<enc>/<heads>":
  "tcnn"   tiny-cuda-nn HashGrid / CutlassMLP (fp16 compute) — fast, on the GPU
  "torch"  plain PyTorch (fp32) — CPU tests, and the fallback when tiny-cuda-nn fails on a GPU
"auto" probes each tiny-cuda-nn part on the device first (fmgs/diag.py, in a child process) and uses
PyTorch for a part that fails — on intellisense08 (RTX 2080, kitchen's tiny-cuda-nn build) the hash
grid kernel was rejected with "invalid configuration argument" on 5 Oct. Parameters of the two
implementations are not interchangeable: field.pt records which one wrote each part.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.utils.checkpoint
from torch import nn

PRIMES = (1, 2654435761, 805459861)


@dataclass
class FieldConfig:
    levels: int = 24
    features: int = 8
    log2_table: int = 20
    base: int = 16
    finest: int = 512
    hidden: int = 256
    layers: int = 2
    clip_dim: int = 512
    dino_dim: int = 384
    split: int = 1                 # the levels as this many consecutive grids (tiny-cuda-nn on sm_75: ≤ 96 dims per grid)

    @property
    def enc_dim(self):
        return self.levels * self.features

    @property
    def per_level_scale(self):
        return math.exp((math.log(self.finest) - math.log(self.base)) / max(1, self.levels - 1))

    def groups(self):
        """[(n_levels, base_resolution, per_level_scale)] per grid. split = 1 is the FMGS grid. With split
        = s the geometric ladder base → finest is cut into s runs of levels; each run starts at its
        level's resolution rounded to an integer (tiny-cuda-nn's base_resolution is an integer) and
        ends exactly where the ladder does, so only the run boundaries move (16 → 84, 98 → 512 for 24 levels)."""
        if self.split == 1:
            return [(self.levels, self.base, self.per_level_scale)]
        if self.levels % self.split or self.levels // self.split < 2:
            raise ValueError(f"split {self.split} must divide the {self.levels} levels into runs of ≥ 2")
        n, b, out = self.levels // self.split, self.per_level_scale, []
        for g in range(self.split):
            lo = round(self.base * b ** (g * n))
            hi = self.finest if g == self.split - 1 else self.base * b ** ((g + 1) * n - 1)
            out.append((n, lo, (hi / lo) ** (1 / (n - 1))))
        return out

    def level_scales(self):
        """tiny-cuda-nn's grid scale per level: base · s^k − 1 (resolution ⌈scale⌉ + 1)."""
        return [base * s ** k - 1.0 for n, base, s in self.groups() for k in range(n)]


def scene_box(means, pad=0.05):
    """(lo, hi) [3] tensors: 1st–99th percentile of the centres per axis, padded by `pad` of the span."""
    m = means.detach().float()
    if len(m) > 200_000:
        m = m[torch.randperm(len(m), generator=torch.Generator().manual_seed(0))[:200_000].to(m.device)]
    lo, hi = torch.quantile(m, 0.01, dim=0), torch.quantile(m, 0.99, dim=0)
    span = (hi - lo).clamp_min(1e-3)
    return lo - pad * span, hi + pad * span


def normalise(xyz, lo, hi):
    return ((xyz - lo) / (hi - lo)).clamp(0.0, 1.0)


class TorchHashGrid(nn.Module):
    """Instant-NGP hash encoding in PyTorch (trilinear, coherent prime hash), as tiny-cuda-nn's
    HashGrid computes it: level scale = base·b^l − 1, resolution = ⌈scale⌉ + 1, position x·scale + 0.5."""

    def __init__(self, cfg: FieldConfig):
        super().__init__()
        self.cfg = cfg
        self.scales, self.res, self.sizes, self.dense = [], [], [], []
        T = 2 ** cfg.log2_table
        tables = []
        for scale in cfg.level_scales():
            res = int(math.ceil(scale)) + 1
            dense = res ** 3 <= T
            size = min(res ** 3, T)
            self.scales.append(scale); self.res.append(res); self.sizes.append(size); self.dense.append(dense)
            tables.append(nn.Parameter(torch.empty(size, cfg.features).uniform_(-1e-4, 1e-4)))
        self.tables = nn.ParameterList(tables)

    def level(self, x, lvl, table):             # one level, the 8 corners at once: [N, 3] → [N, F]
        corners = torch.tensor([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)], device=x.device)
        pos = x * self.scales[lvl] + 0.5
        p0 = torch.floor(pos)
        frac = pos - p0
        idx = p0.long()[:, None, :] + corners                                   # [N, 8, 3]
        w = torch.where(corners.bool(), frac[:, None, :], 1 - frac[:, None, :]).prod(-1)   # [N, 8]
        if self.dense[lvl]:
            r = self.res[lvl]
            idx = idx.clamp(0, r - 1)
            flat = idx[..., 0] + r * (idx[..., 1] + r * idx[..., 2])
        else:
            flat = (idx[..., 0] * PRIMES[0]) ^ (idx[..., 1] * PRIMES[1]) ^ (idx[..., 2] * PRIMES[2])
        flat = flat % self.sizes[lvl]
        return torch.einsum("nc,ncf->nf", w, table[flat])

    def forward(self, x):                       # x in [0, 1]³, [N, 3] → [N, L·F]
        # With gradients, each level is checkpointed: backward recomputes its gather instead of
        # keeping [N, 8, F] per level (2 GB at 262k points × 24 levels), so the PyTorch fallback
        # costs about as much memory as tiny-cuda-nn.
        ckpt = torch.is_grad_enabled() and x.shape[0] > 4096
        out = []
        for lvl, table in enumerate(self.tables):
            if ckpt:
                out.append(torch.utils.checkpoint.checkpoint(self.level, x, lvl, table, use_reentrant=False))
            else:
                out.append(self.level(x, lvl, table))
        return torch.cat(out, -1)


def _mlp(n_in, n_out, hidden, layers):
    mods, d = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, n_out))
    return nn.Sequential(*mods)


class TcnnGrids(nn.Module):
    """Consecutive tiny-cuda-nn hash grids (FieldConfig.split > 1), concatenated in level order."""

    def __init__(self, grids):
        super().__init__()
        self.grids = nn.ModuleList(grids)

    def forward(self, x):
        return torch.cat([g(x) for g in self.grids], -1)


def tcnn_encoding(c: FieldConfig):
    import tinycudann as tcnn
    grids = [tcnn.Encoding(3, {"otype": "HashGrid", "n_levels": n, "n_features_per_level": c.features,
                               "log2_hashmap_size": c.log2_table, "base_resolution": base, "per_level_scale": s})
             for n, base, s in c.groups()]
    return grids[0] if len(grids) == 1 else TcnnGrids(grids)


def tcnn_head(c: FieldConfig, n_out, otype="CutlassMLP"):
    import tinycudann as tcnn
    return tcnn.Network(c.enc_dim, n_out, {"otype": otype, "activation": "ReLU", "output_activation": "None",
                                           "n_neurons": c.hidden, "n_hidden_layers": c.layers})


def split_impl(impl):
    """"tcnn" | "torch" | "<enc>/<heads>" → (enc, heads)."""
    parts = impl.split("/") if "/" in impl else [impl, impl]
    if len(parts) != 2 or any(p not in ("tcnn", "torch") for p in parts):
        raise ValueError(f"unknown field implementation {impl!r} (tcnn, torch or <enc>/<heads>; 'auto' is resolved first)")
    return parts[0], parts[1]


class FeatureField(nn.Module):
    def __init__(self, lo, hi, cfg: FieldConfig = None, impl="tcnn"):
        super().__init__()
        self.cfg = cfg or FieldConfig()
        enc, head = split_impl(impl)
        self.impl = f"{enc}/{head}"
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float32).clone())
        self.register_buffer("hi", torch.as_tensor(hi, dtype=torch.float32).clone())
        c = self.cfg
        self.encoding = tcnn_encoding(c) if enc == "tcnn" else TorchHashGrid(c)
        if head == "tcnn":
            self.clip_head, self.dino_head = tcnn_head(c, c.clip_dim), tcnn_head(c, c.dino_dim)
        else:
            self.clip_head = _mlp(c.enc_dim, c.clip_dim, c.hidden, c.layers)
            self.dino_head = _mlp(c.enc_dim, c.dino_dim, c.hidden, c.layers)

    def encode(self, xyz):
        """[N, 3] splat-frame points → [N, L·F] float32 encoding."""
        return self.encoding(normalise(xyz.float(), self.lo, self.hi)).float()

    def heads(self, enc):
        """[M, L·F] → {"clip": [M, 512], "dino": [M, 384]} float32."""
        return {"clip": self.clip_head(enc).float(), "dino": self.dino_head(enc).float()}

    def forward(self, xyz):
        return self.heads(self.encode(xyz))

    def save(self, path, **extra):
        torch.save({"state": self.state_dict(), "cfg": asdict(self.cfg), "impl": self.impl,
                    "lo": self.lo.cpu(), "hi": self.hi.cpu(), **extra}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        d = torch.load(path, map_location="cpu")
        f = cls(d["lo"], d["hi"], FieldConfig(**d["cfg"]), d["impl"]).to(device)
        f.load_state_dict(d["state"])
        return f, d
