import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")

from radiance_semantics import lift as L  # noqa: E402
from radiance_semantics import probe  # noqa: E402
from radiance_semantics.store import check_order, order_hash, pca_rgb, read_table, write_table  # noqa: E402


def scene(n=80, seed=4):
    g = probe.synthetic(n, seed, "cpu")
    g.means[:, 2] = g.means[:, 2].clamp(2.0, 5.0)
    g.scales.mul_(10.0)
    return g


def views(k=2):
    out = []
    for i in range(k):
        c2w = torch.eye(4)[:3].clone()
        c2w[:, 1:3] *= -1               # OpenGL camera looking along +z of the world (as the synthetic camera)
        c2w[0, 3] = 0.2 * i
        out.append(L.View(stem=f"f{i}", c2w=c2w, fx=40.0, fy=40.0, cx=24.0, cy=16.0, width=48, height=32))
    return out


def test_constant_map_lifts_to_the_constant():
    g = scene()
    const = np.tile(np.array([0.3, -1.0, 2.0, 0.5, 4.0], np.float32), (4, 6, 1))
    feats, w, st = L.lift_views(g, views(), lambda stem: {"clip": const}, feat_width=24, backend="reference",
                                chunk=2, log=lambda m: None)            # chunk 2 < 5 channels: chunked path
    seen = w > 0
    assert st["seen"] == int(seen.sum()) > 0 and st["passes"] == 2 * (1 + 3)
    assert np.allclose(feats["clip"][seen], const[0, 0], atol=1e-4)
    assert np.all(feats["clip"][~seen] == 0)
    assert int((w > 0).sum()) == st["seen"]                    # unseen ⇔ weight exactly 0


def test_two_region_map_separates_left_and_right():
    g = scene(200, seed=7)
    grid = np.zeros((4, 8, 2), np.float32)
    grid[:, :4, 0] = 1.0                 # left half: channel 0
    grid[:, 4:, 1] = 1.0                 # right half: channel 1
    v = views(1)
    feats, w, _ = L.lift_views(g, v, lambda stem: {"clip": grid}, feat_width=48, backend="reference",
                               log=lambda m: None)
    from radiance_semantics.render import viewmat_from_c2w
    vm = viewmat_from_c2w(v[0].c2w)
    cam = (vm[:3, :3] @ g.means.T + vm[:3, 3:4]).T
    u = 40.0 * cam[:, 0] / cam[:, 2] + 24.0
    sure = (w > w.max() * 0.2) & ((u < 16) | (u > 32)).numpy()
    left = sure & (u < 16).numpy()
    right = sure & (u > 32).numpy()
    assert left.any() and right.any()
    assert (feats["clip"][left, 0] > feats["clip"][left, 1]).mean() > 0.9
    assert (feats["clip"][right, 1] > feats["clip"][right, 0]).mean() > 0.9


def test_view_K_upsample_and_normalise():
    v = views(1)[0]
    K = L.view_K(v, 24, 16)
    assert torch.allclose(K, torch.tensor([[20.0, 0, 12.0], [0, 20.0, 8.0], [0, 0, 1.0]]))
    assert L.render_size(v, 24) == (24, 16)
    grid = np.arange(2 * 3, dtype=np.float32).reshape(2, 3, 1)
    up = L.upsample(grid, 6, 4, "cpu")[..., 0].numpy()
    assert up.shape == (4, 6) and up[0, 0] == 0.0 and up[-1, -1] == 5.0      # edges clamp to the edge cells
    assert np.all(np.diff(up, axis=1) >= 0) and np.all(np.diff(up, axis=0) >= 0)
    x = L.normalise_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(x, [[0.6, 0.8], [0, 0]])


def test_table_round_trip_and_order_guard(tmp_path):
    n = 50
    rng = np.random.default_rng(0)
    clip = L.normalise_rows(rng.normal(size=(n, 512)))
    clip[3] = 0
    dino = L.normalise_rows(rng.normal(size=(n, 384)))
    order = rng.permutation(200)[:n]
    geom = rng.normal(size=(n, 5)).astype(np.float32)
    w = rng.random(n).astype(np.float32)
    d = tmp_path / "lift"
    idx = write_table(d, clip=clip, dino=dino, weight=w, geom=geom, order=order, meta={"key": "k1", "backend": "lift"})
    assert idx["n"] == n and idx["order_sha"] == order_hash(order)
    t = read_table(d)
    assert t.clip.shape == (n, 512) and t.clip.dtype == np.float16 and np.allclose(t.clip, clip, atol=2e-3)
    assert t.dino.shape == (n, 384) and np.allclose(t.weight, w) and np.allclose(t.geom, geom)
    assert t.pca.shape == (n, 3) and tuple(t.pca[3]) == (128, 128, 128)            # unseen row: mid grey
    check_order(t, order)
    with pytest.raises(ValueError, match="different .splat order"):
        check_order(t, order[::-1])
    write_table(d, clip=clip[:10], weight=w[:10], geom=geom[:10], order=order[:10], meta={"key": "k2"})
    t2 = read_table(d)
    assert t2.n == 10 and t2.index["key"] == "k2" and t2.dino is None
    assert not (tmp_path / "lift.writing").exists() and not (tmp_path / "lift.old").exists()
    assert json.loads((d / "index.json").read_text())["shapes"]["clip.f16"] == [10, 512]


def test_pca_rgb_spreads_values():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 16)).astype(np.float32)
    rgb = pca_rgb(x)
    assert rgb.shape == (500, 3) and rgb.dtype == np.uint8 and rgb.min() == 0 and rgb.max() == 255
