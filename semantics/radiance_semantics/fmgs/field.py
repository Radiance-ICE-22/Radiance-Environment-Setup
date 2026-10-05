"""FeatureField: Instant-NGP multi-resolution hash grid over the scene, two MLP heads.

Input: Gaussian centres in the splat (world) frame, normalised into [0, 1]³ by the scene box
(1st–99th percentile of the centres, padded 5 %; points outside are clamped onto it).
Encoding: L levels × F features, resolutions growing geometrically from `base` to `finest`, each
level a dense grid when it fits in 2^log2_table entries and a hashed one otherwise → L·F values.
Heads: CLIP (512) and DINO (384), each `layers` hidden layers of `hidden` ReLU units.

Defaults are the FMGS configuration (24 levels, 16 → 512, 2^20, 8 features; 192-d encoding).

Two implementations of the same function:
  "tcnn"   tiny-cuda-nn HashGrid + CutlassMLP (fp16 compute) — the GPU path on the hosts
  "torch"  plain PyTorch (fp32) — CPU tests, and a fallback if tiny-cuda-nn misbehaves
Their parameters are not interchangeable: field.pt records which one wrote it.
"""

import math
from dataclasses import asdict, dataclass

import torch
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

    @property
    def enc_dim(self):
        return self.levels * self.features

    @property
    def per_level_scale(self):
        return math.exp((math.log(self.finest) - math.log(self.base)) / max(1, self.levels - 1))


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
        b = cfg.per_level_scale
        self.scales, self.res, self.sizes, self.dense = [], [], [], []
        T = 2 ** cfg.log2_table
        tables = []
        for lvl in range(cfg.levels):
            scale = cfg.base * b ** lvl - 1.0
            res = int(math.ceil(scale)) + 1
            dense = res ** 3 <= T
            size = min(res ** 3, T)
            self.scales.append(scale); self.res.append(res); self.sizes.append(size); self.dense.append(dense)
            tables.append(nn.Parameter(torch.empty(size, cfg.features).uniform_(-1e-4, 1e-4)))
        self.tables = nn.ParameterList(tables)

    def forward(self, x):                       # x in [0, 1]³, [N, 3] → [N, L·F]
        out = []
        corners = torch.tensor([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)], device=x.device)
        for lvl, table in enumerate(self.tables):
            pos = x * self.scales[lvl] + 0.5
            p0 = torch.floor(pos)
            frac = pos - p0
            p0 = p0.long()
            acc = 0
            for c in corners:
                idx = p0 + c
                w = torch.prod(torch.where(c.bool(), frac, 1 - frac), dim=-1, keepdim=True)
                if self.dense[lvl]:
                    r = self.res[lvl]
                    idx = idx.clamp(0, r - 1)
                    flat = idx[:, 0] + r * (idx[:, 1] + r * idx[:, 2])
                else:
                    flat = (idx[:, 0] * PRIMES[0]) ^ (idx[:, 1] * PRIMES[1]) ^ (idx[:, 2] * PRIMES[2])
                flat = flat % self.sizes[lvl]
                acc = acc + w * table[flat]
            out.append(acc)
        return torch.cat(out, -1)


def _mlp(n_in, n_out, hidden, layers):
    mods, d = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, n_out))
    return nn.Sequential(*mods)


class FeatureField(nn.Module):
    def __init__(self, lo, hi, cfg: FieldConfig = None, impl="tcnn"):
        super().__init__()
        self.cfg = cfg or FieldConfig()
        self.impl = impl
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float32).clone())
        self.register_buffer("hi", torch.as_tensor(hi, dtype=torch.float32).clone())
        c = self.cfg
        if impl == "tcnn":
            import tinycudann as tcnn
            self.encoding = tcnn.Encoding(3, {"otype": "HashGrid", "n_levels": c.levels, "n_features_per_level": c.features,
                                              "log2_hashmap_size": c.log2_table, "base_resolution": c.base,
                                              "per_level_scale": c.per_level_scale})
            net = {"otype": "CutlassMLP", "activation": "ReLU", "output_activation": "None",
                   "n_neurons": c.hidden, "n_hidden_layers": c.layers}
            self.clip_head = tcnn.Network(c.enc_dim, c.clip_dim, net)
            self.dino_head = tcnn.Network(c.enc_dim, c.dino_dim, net)
        elif impl == "torch":
            self.encoding = TorchHashGrid(c)
            self.clip_head = _mlp(c.enc_dim, c.clip_dim, c.hidden, c.layers)
            self.dino_head = _mlp(c.enc_dim, c.dino_dim, c.hidden, c.layers)
        else:
            raise ValueError(f"unknown field implementation {impl!r} (tcnn or torch)")

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
