import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from radiance_semantics import annotations as A
from radiance_semantics import query as Q
from radiance_semantics.lift import normalise_rows

pytest.importorskip("scipy")


def test_relevancy_matches_lerf_formula():
    rng = np.random.default_rng(0)
    f = normalise_rows(rng.normal(size=(20, 8)))
    q = normalise_rows(rng.normal(size=(1, 8)))[0]
    negs = normalise_rows(rng.normal(size=(3, 8)))
    w = np.ones(20)
    w[5] = 0
    rel = Q.relevancy(f.astype(np.float16), w, q, negs, chunk=7)
    sq = f @ q
    sn = f @ negs.T
    soft = np.exp(10 * sq)[:, None] / (np.exp(10 * sq)[:, None] + np.exp(10 * sn))
    exp = soft.min(1)
    exp[5] = 0
    assert np.allclose(rel, exp, atol=2e-3)                 # fp16 input


def test_voxel_labels_split_two_blobs():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.05, size=(100, 3))
    b = rng.normal(0, 0.05, size=(100, 3)) + [1.0, 0, 0]
    lab, k, vox = Q.voxel_labels(np.vstack([a, b]), voxel=0.1)
    assert k >= 2 and len(set(lab[:100])) == 1 and len(set(lab[100:])) == 1 and lab[0] != lab[150]


class FakeEncoder:
    """Text → fixed unit vectors: the query vector is e0, negatives e1..e4."""

    def encode(self, texts):
        out = np.zeros((len(texts), 8), np.float32)
        for i in range(len(texts)):
            out[i, i] = 1.0
        return out


class Tab:
    def __init__(self, clip, weight, geom):
        self.clip, self.weight, self.geom = clip, weight, geom
        self.n = len(clip)


def synthetic_table():
    """Two objects that match the query (a big one at x=2, a small one at x=-2) and background."""
    rng = np.random.default_rng(2)
    big = rng.normal(0, 0.08, (300, 3)) + [2.0, 0.0, 0.5]
    small = rng.normal(0, 0.05, (60, 3)) + [-2.0, 1.0, 0.5]
    bg = rng.uniform(-3, 3, (2000, 3))
    pts = np.vstack([big, small, bg])
    clip = np.zeros((len(pts), 8), np.float32)
    clip[:360, 0] = 1.0                                    # matches the query (e0)
    clip[360:, 1] = 1.0                                    # looks like "object"
    clip = normalise_rows(clip + rng.normal(0, 0.05, clip.shape))
    geom = np.hstack([pts, np.full((len(pts), 1), 0.9), np.full((len(pts), 1), 0.02)]).astype(np.float32)
    return Tab(clip.astype(np.float16), np.ones(len(pts), np.float32), geom)


def cams():
    """Cameras on a ring at z=1.5 looking at the origin."""
    c2w = []
    for t in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        pos = np.array([4 * np.cos(t), 4 * np.sin(t), 1.5])
        back = pos / np.linalg.norm(pos)                  # OpenGL: camera z axis points away from the target
        right = np.cross([0, 0, 1.0], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        c2w.append(np.column_stack([right, up, back, pos]))
    return np.array(c2w)


def test_run_query_ranks_big_object_first_with_feasible_approach():
    tab = synthetic_table()
    sparse = np.random.default_rng(3).uniform(-3, 3, (3000, 3)) * [1, 1, 0] + [0, 0, -0.5]   # floor-ish points
    scene = Q.Scene(cam_c2w=cams(), sparse_xyz=sparse)
    s = Q.Settings(standoff=1.0, margin=0.5)
    res = Q.run_query(tab, scene, "big thing", FakeEncoder(), s)
    c = res["candidates"]
    assert len(c) == 2 and c[0]["n"] > c[1]["n"]
    assert np.allclose(c[0]["centroid"], [2.0, 0.0, -0.5], atol=0.05)          # course frame: y, z flipped
    assert np.allclose(c[1]["centroid"], [-2.0, -1.0, -0.5], atol=0.05)
    a = Q.to_splat(np.array(c[0]["approach"]))
    lo, hi = scene.camera_box(s.margin)
    assert np.all(a >= lo - 1e-9) and np.all(a <= hi + 1e-9)
    horiz = np.linalg.norm(a[:2] - np.array(c[0]["centroid_splat"])[:2])
    assert 0.3 < horiz <= 1.0 + 1e-6 and abs(a[2] - 1.5) < 1e-9     # towards the cameras, at their height
    assert np.allclose(lo[2], 1.5) and np.allclose(hi[2], 1.5)       # flat capture: z falls back to the cameras
    assert c[0]["gap"] is not None and res["margin"] > 0.5 and not res["ambiguous"]
    sc = Q.score_against(res, [2.1, 0.05, -0.45])
    assert sc["hit"] and sc["rank_of_hit"] == 1
    miss = Q.score_against(res, [-2.0, -1.0, -0.5])
    assert not miss["hit"] and miss["rank_of_hit"] == 2


def test_unseen_and_low_opacity_are_ignored():
    tab = synthetic_table()
    tab.weight[:300] = 0                                   # the big object was never seen
    res = Q.run_query(tab, Q.Scene(cam_c2w=cams()), "x", FakeEncoder())
    assert len(res["candidates"]) == 1 and res["candidates"][0]["gap"] is None
    tab.geom[:, 3] = 0.05
    assert Q.run_query(tab, Q.Scene(cam_c2w=cams()), "x", FakeEncoder())["candidates"] == []


def test_relative_threshold_drops_a_moderately_relevant_wall():
    """A large wall at relevancy ~0.6 and a small object at ~0.8: with a fixed threshold (alpha 0) the
    wall's summed score wins; relative to the query's peak (alpha 0.5, τ ≈ 0.675) only the object is
    left — the "whiteboard vs white walls" case from backroom (5 Oct)."""
    rng = np.random.default_rng(6)
    wall = np.column_stack([rng.uniform(-3, 3, 5000), np.full(5000, 2.5), rng.uniform(0, 2.5, 5000)])
    obj = rng.normal(0, 0.08, (200, 3)) + [0.0, 0.0, 1.0]
    pts = np.vstack([wall, obj])

    def feat(a):                                           # rel = sigmoid(10·a) with e0 = query, e5 unrelated
        v = np.zeros(8)
        v[0], v[5] = a, np.sqrt(1 - a * a)
        return v
    clip = np.array([feat(0.0405)] * len(wall) + [feat(0.1386)] * len(obj), np.float32)
    geom = np.hstack([pts, np.full((len(pts), 1), 0.9), np.full((len(pts), 1), 0.02)]).astype(np.float32)
    tab = Tab(clip, np.ones(len(pts), np.float32), geom)
    sc = Q.Scene(cam_c2w=cams())
    fixed = Q.run_query(tab, sc, "x", FakeEncoder(), Q.Settings(rel_alpha=0.0))
    assert fixed["candidates"][0]["large"] and fixed["candidates"][0]["n"] > 1000      # the wall wins
    rel = Q.run_query(tab, sc, "x", FakeEncoder(), Q.Settings(rel_alpha=0.5))
    assert abs(rel["peak"] - 0.8) < 0.01 and abs(rel["tau"] - 0.675) < 0.01
    assert len(rel["candidates"]) == 1 and not rel["candidates"][0]["large"]
    assert np.allclose(rel["candidates"][0]["centroid"], [0.0, 0.0, -1.0], atol=0.05)


def test_annotations_init_set_ready(tmp_path):
    p = tmp_path / "semantics" / "queries.json"
    d = A.init(A.load(p), "backroom")
    assert [q["text"] for q in d["queries"]] == ["red tool chest", "shop vacuum", "green foam mats",
                                                 "garden cart", "whiteboard"]
    A.set_position(d, "shop vacuum", (1, -2, -0.5))
    A.save(p, d)
    done, missing = A.ready(A.load(p), "phase1_gate")
    assert [q["text"] for q in done] == ["shop vacuum"] and len(missing) == 4
    A.init(d, "backroom")
    assert len(d["queries"]) == 5                          # init is idempotent
    assert json.loads(p.read_text())["frame"].startswith("course")


def test_splat_order_matches_the_previous_inline_selection():
    """course_tools.splat_order must reproduce exactly the order cmd_splat used before the refactor,
    or semantic rows and Galley's cached .splat records would disagree."""
    figs = Path(__file__).resolve().parents[2] / "figs"
    if not (figs / "course_tools.py").exists():
        pytest.skip("figs/ not next to semantics/")
    sys.path.insert(0, str(figs))
    spec = importlib.util.spec_from_file_location("course_tools", figs / "course_tools.py")
    ct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct)
    rng = np.random.default_rng(5)
    n = 5000
    means = rng.normal(size=(n, 3))
    means[7] = np.nan
    scales = rng.normal(-4, 1, size=(n, 3))
    opac = rng.normal(0, 2, size=(n, 1))
    alpha = 1.0 / (1.0 + np.exp(-opac.reshape(-1)))       # the pre-refactor code, verbatim
    keep = (alpha >= 0.05) & np.isfinite(means).all(1) & np.isfinite(scales).all(1)
    idx = np.nonzero(keep)[0]
    importance = np.exp(scales[idx].sum(1)) * alpha[idx]
    old = idx[np.argsort(-importance)][:1000]
    assert np.array_equal(ct.splat_order(means, scales, opac, 0.05, 1000), old)
    assert 7 not in ct.splat_order(means, scales, opac)


def test_pose_outliers_are_dropped_from_the_camera_box():
    """GTN_lab_v1: 3 of 600 cameras registered 100–219 m away from an 11 × 8 m lab."""
    rng = np.random.default_rng(0)
    P = rng.uniform([-5, -4, 0], [5, 4, 1.5], (200, 3))
    assert Q.pose_inliers(P).all()                                  # a clean capture keeps every camera
    P[[3, 50]] = [[168.0, 120.0, -60.0], [-150.0, 2.0, 1.0]]
    keep = Q.pose_inliers(P)
    assert (~keep).sum() == 2 and not keep[3] and not keep[50]
    c2w = np.tile(np.eye(4), (len(P), 1, 1))
    c2w[:, :3, 3] = P
    sc = Q.Scene(cam_c2w=c2w)
    lo, hi = sc.camera_box(0.5)
    assert sc.dropped_cameras == 2 and hi[0] < 5 and lo[0] > -5


def test_dino_split_separates_two_touching_objects():
    rng = np.random.default_rng(1)
    a, b = np.eye(8)[0], np.eye(8)[1]
    X = np.concatenate([a + 0.05 * rng.standard_normal((40, 8)), b + 0.05 * rng.standard_normal((30, 8))])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    g = Q.dino_split(X, 15, 0.5)
    assert g is not None and sorted([g.sum(), (~g).sum()]) == [30, 40] and len(set(g[:40])) == 1
    same = a + 0.05 * rng.standard_normal((70, 8))
    assert Q.dino_split(same / np.linalg.norm(same, axis=1, keepdims=True), 15, 0.5) is None   # one object: no split
    assert Q.dino_split(X[:20], 15, 0.5) is None                                              # too few rows


def test_diffusion_spreads_relevancy_within_dino_similar_neighbours_only():
    rel = np.array([0.9, 0.5, 0.5, 0.0], np.float32)
    idx = np.array([[1, 2], [0, 2], [0, 1], [0, 1]], np.int32)
    w = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]], np.float32)   # 1 is like 0; 2 is unlike both
    r = Q.diffuse(rel, idx, w, 0.5, 1)
    assert r[1] > 0.5 and np.isclose(r[2], 0.5) and r[3] == 0.0                  # unseen rows (rel 0) stay 0
    assert np.isclose(r[0], 0.5 * 0.9 + 0.5 * 0.5)
