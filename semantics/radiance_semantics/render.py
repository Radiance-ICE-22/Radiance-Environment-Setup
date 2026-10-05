"""Feature rendering on frozen Gaussians: gsplat (the call splatfacto makes) and a CPU reference.

Rendering is LINEAR in the per-Gaussian features for fixed geometry:

    image[p] = sum_i  w_i(p) * f_i,     w_i(p) = alpha_i(p) * prod_{j before i} (1 - alpha_j(p))

so for any 2D feature map F, the gradient of  L(f) = sum_p <image[p], F[p]>  with respect to
f_i is  sum_p w_i(p) F[p]  — the weighted sum the lift backend needs (Occam's LGS / LUDVIG).
probe.py checks exactly this on the installed gsplat.

`gsplat`    gsplat 1.0.0 `rasterization` with sh_degree=None (N-D features, up to 512 channels;
            other counts are padded to the next power of two), packed=False, classic mode —
            the same call as nerfstudio 1.1.4's splatfacto. Needs CUDA.
`reference` dense PyTorch alpha compositing using gsplat's own projection (gsplat.cuda.
            _torch_impl), for small scenes on the CPU. Only for tests: it has no tiling and no
            early stop, so it matches gsplat to within the T < 1e-4 cut-off, not bit for bit.
"""

from dataclasses import dataclass

import torch

ALPHA_MIN = 1.0 / 255.0     # gsplat skips weaker contributions
ALPHA_MAX = 0.999


@dataclass
class Gaussians:
    """Activated parameters: unit quaternions (w, x, y, z), linear scales, opacities in (0, 1)."""
    means: torch.Tensor       # [N, 3]
    quats: torch.Tensor       # [N, 4]
    scales: torch.Tensor      # [N, 3]
    opacities: torch.Tensor   # [N]

    def __len__(self):
        return self.means.shape[0]

    def to(self, device):
        return Gaussians(*(t.to(device) for t in (self.means, self.quats, self.scales, self.opacities)))

    @classmethod
    def from_splatfacto(cls, state):
        """From a checkpoint's pipeline state dict (raw parameters: log scales, opacity logits)."""
        def p(name):
            for k, v in state.items():
                if k.endswith(f"gauss_params.{name}") or k.endswith(f"_model.{name}"):
                    return v.float()
            raise KeyError(f"checkpoint has no '{name}' parameter (not a splatfacto model?)")
        q = p("quats")
        return cls(p("means"), q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12),
                   torch.exp(p("scales")), torch.sigmoid(p("opacities")).reshape(-1))


def render_features(g, features, viewmat, K, width, height, backend="gsplat"):
    """Render per-Gaussian features [N, C] → (image [H, W, C], alpha [H, W, 1]).

    viewmat: [4, 4] world→camera in OpenCV convention (x right, y down, z forward), as
    splatfacto's get_viewmat produces; K: [3, 3] pixel intrinsics for width × height.
    """
    if backend == "gsplat":
        from gsplat import rasterization
        img, alpha, _ = rasterization(
            means=g.means, quats=g.quats, scales=g.scales, opacities=g.opacities, colors=features,
            viewmats=viewmat[None], Ks=K[None], width=width, height=height, tile_size=16, packed=False,
            near_plane=0.01, far_plane=1e10, render_mode="RGB", sh_degree=None, sparse_grad=False,
            absgrad=False, rasterize_mode="classic")
        return img[0], alpha[0]
    if backend == "reference":
        return _render_reference(g, features, viewmat, K, width, height)
    raise ValueError(f"unknown backend {backend!r}")


def _render_reference(g, features, viewmat, K, width, height):
    from gsplat.cuda._torch_impl import _fully_fused_projection, _quat_scale_to_covar_preci
    covars, _ = _quat_scale_to_covar_preci(g.quats, g.scales, compute_preci=False, triu=False)
    radii, means2d, depths, conics, _ = _fully_fused_projection(
        g.means, covars, viewmat[None], K[None], width, height, eps2d=0.3, near_plane=0.01)
    radii, means2d, depths, conics = radii[0], means2d[0], depths[0], conics[0]

    order = torch.argsort(depths)                                     # front to back
    ys, xs = torch.meshgrid(torch.arange(height, dtype=features.dtype),
                            torch.arange(width, dtype=features.dtype), indexing="ij")
    pix = torch.stack([xs.reshape(-1) + 0.5, ys.reshape(-1) + 0.5], -1)   # [P, 2] pixel centres
    d = pix[:, None, :] - means2d[order][None]                         # [P, N, 2]
    a, b, c = conics[order].unbind(-1)
    sigma = 0.5 * (a * d[..., 0] ** 2 + c * d[..., 1] ** 2) + b * d[..., 0] * d[..., 1]
    alpha = (g.opacities[order][None] * torch.exp(-sigma)).clamp(max=ALPHA_MAX)
    keep = (sigma >= 0) & (alpha >= ALPHA_MIN) & (radii[order] > 0)[None]
    alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], 1), 1)
    w = alpha * trans                                                  # [P, N] blend weights
    img = w @ features[order]                                          # [P, C]
    return img.reshape(height, width, -1), w.sum(1).reshape(height, width, 1)


def viewmat_from_c2w(c2w):
    """nerfstudio camera-to-world [3, 4] (OpenGL: y up, z back) → gsplat world→camera [4, 4]
    (OpenCV), identical to nerfstudio.models.splatfacto.get_viewmat for one camera."""
    R = c2w[:3, :3] * torch.tensor([1.0, -1.0, -1.0], dtype=c2w.dtype, device=c2w.device)
    T = c2w[:3, 3:4]
    vm = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
    vm[:3, :3] = R.T
    vm[:3, 3:4] = -R.T @ T
    return vm


def intrinsics(fx, fy, cx, cy, device="cpu"):
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device)
