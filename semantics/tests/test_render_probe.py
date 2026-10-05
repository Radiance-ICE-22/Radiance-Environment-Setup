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
    assert probe.lift_passes(64) == 8 + 6 + 1
    assert probe.lift_passes(128) == 4 + 3 + 1
    assert probe.parse_res("480x270, 960X540".replace(" ", "")) == [(480, 270), (960, 540)]


def test_cli_reference_backend_runs_on_cpu(capsys):
    rc = probe.main(["--backend", "reference", "--device", "cpu", "--n", "40", "--res", "24x16",
                     "--channels", "4", "--reps", "1"])
    out = capsys.readouterr().out
    assert "GALLEY_JSON" in out
    assert rc in (0, 1) and math.isfinite(0.0)
