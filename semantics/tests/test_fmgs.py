"""Phase 4 (FMGS backend), CPU: losses by hand, the hash grid, subset/frustum selection, and a real
training run on a synthetic scene whose teacher maps are rendered from known per-Gaussian
features (the field must learn them), with the reference renderer and the PyTorch field."""
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")

from radiance_semantics import lift as L  # noqa: E402
from radiance_semantics.fmgs import losses as LS  # noqa: E402
from radiance_semantics.fmgs import train as T  # noqa: E402
from radiance_semantics.fmgs.bake import bake  # noqa: E402
from radiance_semantics.fmgs.field import FeatureField, FieldConfig, TorchHashGrid, normalise, scene_box, split_impl  # noqa: E402
from radiance_semantics.render import Gaussians, render_features, viewmat_from_c2w  # noqa: E402

SMALL = {"levels": 4, "features": 2, "base": 4, "finest": 32, "hidden": 32, "layers": 1, "clip_dim": 6, "dino_dim": 4}


# ── losses ────────────────────────────────────────────────────────────────────
def test_clip_huber_and_dino_l2_by_hand():
    pred = torch.tensor([[[0.0, 3.0]], [[1.0, 1.0]]])          # [H=2, W=1, C=2]
    tgt = torch.tensor([[[1.0, 0.0]], [[0.0, 0.0]]])
    mask = torch.tensor([[True], [False]])
    # pixel 0 only: |0−1| = 1 ≤ δ → 0.5·1² = 0.5 ; |3−0| = 3 > δ → δ(3 − δ/2) = 1.25·2.375
    exp = (0.5 + 1.25 * (3 - 0.625)) / 2
    assert math.isclose(float(LS.clip_loss(pred, tgt, mask, 1.25)), exp, rel_tol=1e-6)
    assert math.isclose(float(LS.dino_loss(pred, tgt, mask)), (1 + 9) / 2, rel_tol=1e-6)
    assert float(LS.clip_loss(pred, tgt, torch.zeros(2, 1, dtype=torch.bool))) == 0.0


def test_pixel_alignment_zero_when_structure_matches_and_positive_otherwise():
    g = torch.Generator().manual_seed(0)
    H, W = 12, 12
    labels = (torch.arange(W)[None, :].expand(H, W) >= 6).long()     # two regions
    basis_c, basis_d = torch.eye(5)[:2], torch.eye(3)[:2]
    clip, dino = basis_c[labels], basis_d[labels]                    # same partition, orthogonal regions
    mask = torch.ones(H, W, dtype=torch.bool)
    assert float(LS.pixel_alignment(clip, dino, mask, 500, k=3, dil=1, generator=g)) < 1e-6
    flat = torch.ones(H, W, 5)                                       # CLIP says all the same, DINO says two
    assert float(LS.pixel_alignment(flat, dino, mask, 500, k=3, dil=1, generator=g)) > 0.01
    out = LS.total_loss(clip, clip, None, None, mask)                # no DINO: clip term only
    assert set(out) == {"clip", "loss"} and float(out["loss"]) == 0.0


# ── field ─────────────────────────────────────────────────────────────────────
def test_scene_box_and_normalise_clamp():
    m = torch.tensor([[0.0, 0, 0], [10.0, 2, 4]]).repeat(50, 1)
    lo, hi = scene_box(m, pad=0.0)
    x = normalise(torch.tensor([[-5.0, 1, 2], [20.0, 1, 2]]), lo, hi)
    assert torch.all(x >= 0) and torch.all(x <= 1)
    assert torch.allclose(x[:, 1:], torch.tensor([[0.5, 0.5], [0.5, 0.5]]), atol=1e-5)


def test_torch_hash_grid_dense_and_hashed_levels_interpolate():
    cfg = FieldConfig(levels=3, features=2, log2_table=6, base=4, finest=16)
    grid = TorchHashGrid(cfg)
    assert grid.dense == [True, False, False]                       # 4³ fits in 2^6; 8³, 16³ do not
    with torch.no_grad():
        for t in grid.tables:
            t.copy_(torch.randn_like(t))
    x = torch.rand(10, 3)
    y = grid(x)
    assert y.shape == (10, cfg.enc_dim) and torch.all(torch.isfinite(y))
    # trilinear: midpoint between two lattice-aligned points equals the average on the coarsest level
    s = grid.scales[0]
    a = torch.tensor([[(1 - 0.5) / s, 0.3, 0.3]]); b = torch.tensor([[(2 - 0.5) / s, 0.3, 0.3]])   # lattice x = 1, 2
    mid = (a + b) / 2
    lvl0 = lambda p: grid(p)[:, :2]                                  # noqa: E731
    assert torch.allclose(lvl0(mid), (lvl0(a) + lvl0(b)) / 2, atol=1e-5)


def test_field_save_load_round_trip(tmp_path):
    f = FeatureField(torch.zeros(3), torch.ones(3), FieldConfig(**SMALL), impl="torch")
    x = torch.rand(7, 3)
    f.save(tmp_path / "f.pt", note=1)
    f2, d = FeatureField.load(tmp_path / "f.pt")
    assert d["impl"] == "torch/torch" and d["note"] == 1 and f2.impl == "torch/torch"
    assert torch.allclose(f(x)["clip"], f2(x)["clip"]) and f2(x)["dino"].shape == (7, 4)
    with pytest.raises(ValueError):
        FeatureField(torch.zeros(3), torch.ones(3), impl="bogus")


def test_impl_split_and_cpu_resolve():
    from radiance_semantics.fmgs.diag import resolve
    assert split_impl("tcnn") == ("tcnn", "tcnn") and split_impl("torch") == ("torch", "torch")
    assert split_impl("torch/tcnn") == ("torch", "tcnn")
    for bad in ("tcnn/", "cuda", "torch/torch/torch"):
        with pytest.raises(ValueError):
            split_impl(bad)
    assert resolve({}, "cpu", log=lambda m: None) == ("torch/torch", {}, {})


def test_split_grid_keeps_the_ladder():
    one, two = FieldConfig(), FieldConfig(split=2)
    assert two.enc_dim == one.enc_dim == 192 and len(two.level_scales()) == 24
    (n0, b0, s0), (n1, b1, s1) = two.groups()
    assert (n0, b0, n1, b1) == (12, 16, 12, 98) and abs(s0 - one.per_level_scale) < 1e-12
    a, b = one.level_scales(), two.level_scales()
    assert a[:12] == pytest.approx(b[:12]) and b[-1] == pytest.approx(a[-1])        # same finest: 511
    assert max(abs(x - y) / (x + 1) for x, y in zip(a, b)) < 0.01                    # boundaries move < 1 %
    g = TorchHashGrid(FieldConfig(**{**SMALL, "split": 2}))
    assert g(torch.rand(10, 3)).shape == (10, SMALL["levels"] * SMALL["features"])
    with pytest.raises(ValueError):
        FieldConfig(levels=24, split=5).groups()


def test_hash_grid_checkpointed_gradients_match():
    torch.manual_seed(0)
    grid = TorchHashGrid(FieldConfig(**SMALL))
    x = torch.rand(5000, 3)                      # > 4096: the checkpointed path
    w = torch.randn(grid(x[:1]).shape[1])
    grads = []
    for ckpt in (True, False):
        grid.zero_grad()
        if ckpt:
            y = grid(x)
        else:
            y = torch.cat([grid.level(x, l, t) for l, t in enumerate(grid.tables)], -1)
        (y @ w).square().mean().backward()
        grads.append([t.grad.clone() for t in grid.tables])
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(*grads))


# ── selection ─────────────────────────────────────────────────────────────────
def test_trainable_subset_is_the_most_opaque():
    g = Gaussians(torch.zeros(5, 3), torch.tensor([[1.0, 0, 0, 0]]).repeat(5, 1), torch.ones(5, 3),
                  torch.tensor([0.1, 0.9, 0.5, 0.95, 0.2]))
    assert T.trainable_subset(g, 0.4).tolist() == [1, 3]
    assert T.trainable_subset(g, 1.0).tolist() == [0, 1, 2, 3, 4]


def test_in_view_frustum():
    vm = torch.eye(4)
    K = torch.tensor([[10.0, 0, 5], [0, 10.0, 5], [0, 0, 1]])
    m = torch.tensor([[0.0, 0, 1], [0.0, 0, -1], [10.0, 0, 1], [0.04, 0, 1]])
    assert T.in_view(m, vm, K, 10, 10).tolist() == [True, False, False, True]


# ── training on a synthetic scene ─────────────────────────────────────────────
def gt_features(means, C, phase):
    x = means
    return torch.stack([torch.sin(1.3 * x[:, i % 3] + phase + i) for i in range(C)], 1)


def scene_and_teachers(n=150, k=4):
    from radiance_semantics import probe
    g = probe.synthetic(n, 3, "cpu")
    g.means[:, 2] = g.means[:, 2].clamp(2.0, 6.0)
    g.scales.mul_(8.0)
    g.opacities.clamp_(min=0.6)
    views = []
    for i in range(k):
        c2w = torch.eye(4)[:3].clone()
        c2w[:, 1:3] *= -1
        c2w[0, 3] = 0.3 * (i - 1.5)
        views.append(L.View(stem=f"f{i}", c2w=c2w, fx=40.0, fy=40.0, cx=24.0, cy=16.0, width=48, height=32))
    gt = {"clip": gt_features(g.means, 6, 0.0), "dino": gt_features(g.means, 4, 1.0)}
    maps = {}
    for v in views:          # teacher maps: the true features rendered (normalised) at a coarse grid
        W, H = 16, 11
        vm, K = viewmat_from_c2w(v.c2w), L.view_K(v, W, H)
        out = {}
        for name, f in gt.items():
            img, a = render_features(g, f, vm, K, W, H, "reference")
            out[name] = (img / a.clamp_min(1e-6)).numpy()
        maps[v.stem] = out
    return g, views, maps


def cfg(**kw):
    base = dict(steps=60, feat_width=24, variant="faithful", impl="torch", log2_table=8, subset_frac=1.0, lr=3e-2,
                lr_final=1e-2, alpha_min=0.3, pa_samples=64, ckpt_every=20, log_every=20, field_cfg=SMALL)
    base.update(kw)
    return T.TrainConfig(**base)


@pytest.mark.parametrize("variant", ["faithful", "blite"])
def test_training_lowers_the_loss_and_leaves_gaussians_untouched(tmp_path, variant):
    g, views, maps = scene_and_teachers()
    before = T.gauss_checksum(g)
    logs = []
    stats, field = T.train(g, views, lambda s: maps.get(s, {}), cfg(variant=variant), tmp_path, "cpu", "reference",
                           log=logs.append)
    assert stats["loss_last"] < 0.7 * stats["loss_first"], stats
    assert stats["gauss_unchanged"] and stats["gauss_checksum_after"] == before == T.gauss_checksum(g)
    assert stats["variant"] == variant and stats["fallback"] is None
    assert (tmp_path / "field.pt").exists() and json.loads((tmp_path / "train.json").read_text())["steps"] == 60
    assert sorted(p.name for p in (tmp_path / "ckpt").glob("*.pt")) == ["step-000040.pt", "step-000060.pt"]
    assert any("step 60/60" in m for m in logs)
    clip, dino = bake(field, g.means.numpy())
    assert clip.shape == (len(g), 6) and dino.shape == (len(g), 4)
    assert np.allclose(np.linalg.norm(clip, axis=1), 1, atol=1e-4)


def test_resume_from_checkpoint(tmp_path):
    g, views, maps = scene_and_teachers()
    T.train(g, views, lambda s: maps[s], cfg(steps=40), tmp_path, "cpu", "reference", log=lambda m: None)
    (tmp_path / "field.pt").unlink()
    logs = []
    stats, _ = T.train(g, views, lambda s: maps[s], cfg(steps=60), tmp_path, "cpu", "reference", log=logs.append)
    assert stats["resumed_from"] == 40 and any("resumed from step-000040.pt" in m for m in logs)
    logs.clear()                                                    # other settings: not resumed
    stats, _ = T.train(g, views, lambda s: maps[s], cfg(steps=60, lr=1e-2), tmp_path, "cpu", "reference", log=logs.append)
    assert stats["resumed_from"] == 0 and any("other settings" in m for m in logs)


def test_out_of_memory_walks_the_ladder(tmp_path, monkeypatch):
    g, views, maps = scene_and_teachers()
    seen = []
    real = T._train_once

    def fake(g_, views_, teachers, c, *a, **k):
        seen.append((c.log2_table, c.subsample, c.variant))
        if len(seen) < 3:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(g_, views_, teachers, c, *a, **k)
    monkeypatch.setattr(T, "_train_once", fake)
    logs = []
    stats, _ = T.train(g, views, lambda s: maps[s], cfg(variant="auto", steps=10, log2_table=20), tmp_path, "cpu",
                       "reference", log=logs.append)
    assert seen == [(20, 1.0, "faithful"), (19, 1.0, "faithful"), (19, 0.5, "faithful")]
    assert stats["fallback"] == {"level": 2, "name": "half the visible Gaussians per step"}
    assert sum("fallback" in m for m in logs) == 2


def test_blend_weights_match_the_lift():
    g, views, maps = scene_and_teachers()
    w = L.blend_weights(g, views, 24, "reference", log=lambda m: None)
    _, w_lift, _ = L.lift_views(g, views, lambda s: {"clip": maps[s]["clip"]}, 24, "reference", log=lambda m: None)
    assert np.allclose(w, w_lift, rtol=1e-5, atol=1e-7)
