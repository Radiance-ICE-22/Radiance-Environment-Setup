"""The probe's identities on the CPU reference renderer (gsplat's own projection, dense
compositing). On the hosts the same checks run against gsplat's CUDA rasterizer."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")

from radiance_semantics import probe, render  # noqa: E402
from radiance_semantics.render import Gaussians, viewmat_from_c2w  # noqa: E402


def small_scene(n=60, W=32, H=24, seed=1):
    g = probe.synthetic(n, seed, "cpu")
    g.means[:, 2] = g.means[:, 2].clamp(2.0, 6.0)
    g.scales.mul_(8.0)                                    # big enough to cover pixels at 32x24
    vm, K = probe.synthetic_camera(W, H, "cpu")
    return g, vm, K, W, H


def test_identities_hold_on_reference():
    g, vm, K, W, H = small_scene()
    r = probe.check(g, vm, K, W, H, channels=8, backend="reference", seed=3)
    assert r["ok"], r
    assert r["visible"] > 0 and r["linearity_rel"] < 1e-5 and r["finite_diff_rel_max"] < 1e-5


def test_check_catches_a_nonlinear_renderer(monkeypatch):
    real = render.render_features

    def squared(g, f, *a, **k):                           # a renderer that is NOT linear in f
        img, alpha = real(g, f, *a, **k)
        return img * img, alpha

    monkeypatch.setattr(render, "render_features", squared)
    g, vm, K, W, H = small_scene()
    r = probe.check(g, vm, K, W, H, channels=4, backend="reference", seed=3)
    assert not r["ok"] and r["linearity_rel"] > 1e-2


class FakeGsplat:
    """Stands in for gsplat 1.0.0's rasterization on the CPU: a fixed linear renderer that, like
    the real CUDA backward kernel, refuses more than 32 channels when gradients are needed."""

    def __init__(self, n, W, H, seed=0):
        gen = torch.Generator().manual_seed(seed)
        w = torch.rand(H * W, n, generator=gen) * (torch.rand(H * W, n, generator=gen) < 0.2)
        self.w = w / w.sum(1, keepdim=True).clamp_min(1.0) * 0.9         # blend weights, alpha ≤ 0.9
        self.W, self.H, self.calls = W, H, []

    def __call__(self, means, quats, scales, opacities, colors, viewmats, Ks, width, height, **kw):
        self.calls.append((colors.shape[-1], colors.requires_grad))
        if colors.requires_grad and colors.shape[-1] > 32:
            raise RuntimeError(f"Unsupported number of channels: {colors.shape[-1]}")
        img = (self.w @ colors).reshape(1, height, width, -1)
        return img, self.w.sum(1).reshape(1, height, width, 1), {}


def test_gsplat_path_chunks_gradients_to_32(monkeypatch):
    import gsplat
    g, vm, K, W, H = small_scene()
    fake = FakeGsplat(len(g), W, H)
    monkeypatch.setattr(gsplat, "rasterization", fake)
    f = torch.zeros(len(g), 96, requires_grad=True)
    F2D = torch.randn(H, W, 96)
    img, _ = render.render_features(g, f, vm, K, W, H, backend="gsplat")
    (img * F2D).sum().backward()
    assert all(c <= 32 for c, rg in fake.calls if rg)                    # three 32-channel chunks
    assert torch.allclose(f.grad, fake.w.T @ F2D.reshape(-1, 96), atol=1e-5)
    with torch.no_grad():                                                # no grad: one 96-wide call
        fake.calls.clear()
        render.render_features(g, torch.ones(len(g), 96), vm, K, W, H, backend="gsplat")
    assert fake.calls == [(96, False)]
    r = probe.check(g, vm, K, W, H, channels=32, backend="gsplat", seed=2)   # 2C = 64 with grad
    assert r["ok"], r
    assert r["grad_chunk_rel"] < 1e-6


def test_weights_reproduce_alpha():
    g, vm, K, W, H = small_scene()
    ones = torch.ones(len(g), 1)
    img, alpha = render.render_features(g, ones, vm, K, W, H, backend="reference")
    assert torch.allclose(img, alpha, atol=1e-6)
    assert float(alpha.max()) <= 1.0 + 1e-6


def test_viewmat_matches_splatfacto():
    splatfacto = pytest.importorskip("nerfstudio.models.splatfacto")
    c2w = torch.tensor([[0.0, -1.0, 0.0, 1.0], [0.6, 0.0, -0.8, 2.0], [0.8, 0.0, 0.6, -0.5]])
    ours = viewmat_from_c2w(c2w)
    theirs = splatfacto.get_viewmat(c2w[None])[0]
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_from_splatfacto_activates_parameters():
    st = {"_model.gauss_params.means": torch.zeros(2, 3), "_model.gauss_params.scales": torch.log(torch.full((2, 3), 0.5)),
          "_model.gauss_params.quats": torch.tensor([[2.0, 0, 0, 0], [0, 0, 3.0, 0]]),
          "_model.gauss_params.opacities": torch.zeros(2, 1), "_model.gauss_params.features_dc": torch.zeros(2, 3)}
    g = Gaussians.from_splatfacto(st)
    assert torch.allclose(g.scales, torch.full((2, 3), 0.5))
    assert torch.allclose(g.quats.norm(dim=1), torch.ones(2))
    assert g.opacities.shape == (2,) and torch.allclose(g.opacities, torch.full((2,), 0.5))


def test_lift_passes_and_res_parsing():
    assert probe.lift_passes(32) == 16 + 12 + 1
    assert probe.lift_passes(64) == 8 + 6 + 1
    assert probe.lift_passes(128) == 4 + 3 + 1
    assert probe.parse_res("480x270, 960X540".replace(" ", "")) == [(480, 270), (960, 540)]


def test_cli_reference_backend_runs_on_cpu(capsys):
    rc = probe.main(["--backend", "reference", "--device", "cpu", "--n", "40", "--res", "24x16",
                     "--channels", "4", "--reps", "1"])
    out = capsys.readouterr().out
    assert "GALLEY_JSON" in out
    assert rc in (0, 1) and math.isfinite(0.0)
