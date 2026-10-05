"""figs/semantic_worker.py's request handling and JSON-lines loop, on a real (tiny) table."""
import base64
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
FIGS = Path(__file__).resolve().parents[2] / "figs"
if not (FIGS / "semantic_worker.py").exists():
    pytest.skip("figs/ not next to semantics/", allow_module_level=True)
sys.path.insert(0, str(FIGS))
spec = importlib.util.spec_from_file_location("semantic_worker", FIGS / "semantic_worker.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)

from radiance_semantics.lift import normalise_rows  # noqa: E402
from radiance_semantics.store import write_table  # noqa: E402


class Enc:
    """'match' → e0, anything else → e1 (so the table's first 300 rows match the query)."""

    def encode(self, texts):
        out = np.zeros((len(texts), 8), np.float32)
        for i, t in enumerate(texts):
            out[i, 0 if t == "match" else 1 + i % 7] = 1.0
        return out


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "figs"
    (r / "figs_env.sh").parent.mkdir(parents=True)
    (r / "figs_env.sh").write_text("#\n")
    run = r / "SousVide" / "gsplats" / "workspace" / "outputs" / "room" / "splatfacto" / "R1"
    (run / "nerfstudio_models").mkdir(parents=True)
    (run / "config.yml").write_text("x")
    (run / "nerfstudio_models" / "step-000000010.ckpt").write_bytes(b"ck")
    sd = r / "SousVide" / "gsplats" / "workspace" / "room"
    sd.mkdir(parents=True)
    frames = []
    for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        pos = np.array([3 * np.cos(t), 3 * np.sin(t), 1.2])
        back = pos / np.linalg.norm(pos)
        right = np.cross([0, 0, 1.0], back)
        right /= np.linalg.norm(right)
        M = np.eye(4)
        M[:3, :3] = np.column_stack([right, np.cross(back, right), back])
        M[:3, 3] = pos
        frames.append({"file_path": f"images/f{len(frames)}.png", "transform_matrix": M.tolist()})
    (sd / "transforms.json").write_text(json.dumps({"frames": frames}))
    rng = np.random.default_rng(0)
    pts = np.vstack([rng.normal(0, 0.08, (300, 3)) + [0.5, 0, 0.6], rng.uniform(-2, 2, (700, 3))])
    clip = np.zeros((1000, 8), np.float32)
    clip[:300, 0] = 1
    clip[300:, 1] = 1
    clip = normalise_rows(clip + rng.normal(0, 0.02, clip.shape))
    geom = np.hstack([pts, np.full((1000, 1), 0.9), np.full((1000, 1), 0.02)]).astype(np.float32)
    from radiance_semantics.paths import find_scene_run
    key = find_scene_run(r, "room").key
    write_table(sd / "semantics" / "R1" / "lift", clip=clip, weight=np.ones(1000, np.float32), geom=geom,
                order=np.arange(1000), meta={"key": key, "backend": "lift", "run": "R1"})
    return r


def test_query_labels_and_errors(root):
    w = W.Worker(root, encoder=Enc())
    out = w.query({"scene": "room", "backend": "lift", "text": "match", "relevancy": True,
                   "settings": {"standoff": 0.8, "negatives": ["a", "b"]}})
    c = out["result"]["candidates"][0]
    assert np.allclose(c["centroid"], [0.5, 0.0, -0.6], atol=0.05) and out["stale"] is False
    rel = np.frombuffer(base64.b64decode(out["relevancy_b64"]), np.uint8)
    assert rel.shape == (1000,) and rel[:300].mean() > 200 and rel[300:].mean() < 128
    lab = w.labels({"scene": "room", "backend": "lift", "index": 5, "labels": ["other", "match"]})
    assert lab["scores"][0][0] == "match" and lab["seen"]
    for bad in ({"scene": "room", "text": ""}, {"scene": "room", "text": "x", "settings": {"nope": 1}},
                {"scene": "room", "backend": "nerf", "text": "x"}):
        with pytest.raises(W.BadRequest):
            w.query(bad)
    with pytest.raises(W.NoTable):
        w.query({"scene": "room", "backend": "fmgs", "text": "x"})
    with pytest.raises(W.NoScene):
        w.query({"scene": "elsewhere", "text": "x"})
    with pytest.raises(W.BadRequest):
        w.labels({"scene": "room", "index": 5000, "labels": ["a"]})


def test_reload_on_new_table_and_stale(root):
    w = W.Worker(root, encoder=Enc())
    w.query({"scene": "room", "text": "match"})
    idx = root / "SousVide" / "gsplats" / "workspace" / "room" / "semantics" / "R1" / "lift" / "index.json"
    ix = json.loads(idx.read_text())
    ix["key"] = "older-key"
    idx.write_text(json.dumps(ix))
    import os
    os.utime(idx, (1e9, 1e9))
    out = w.query({"scene": "room", "text": "match"})
    assert out["stale"] is True and out["table"]["key"] == "older-key"


def test_serve_loop_protocol(root):
    w = W.Worker(root, encoder=Enc())
    inp = io.StringIO("\n".join([
        json.dumps({"id": 1, "op": "ping"}),
        "not json",
        json.dumps({"id": 3, "op": "query", "scene": "room", "text": "match"}),
        json.dumps({"id": 4, "op": "query", "scene": "room", "backend": "fmgs", "text": "x"}),
        json.dumps({"id": 5, "op": "explode"}),
    ]) + "\n")
    out = io.StringIO()
    W.serve(w, inp, out, idle=0, log=io.StringIO())
    lines = [json.loads(x) for x in out.getvalue().splitlines()]
    assert lines[0]["ready"] and lines[1]["id"] == 1 and lines[1]["ok"]
    assert lines[2]["ok"] is False and lines[2]["code"] == "bad_request"
    assert lines[3]["id"] == 3 and lines[3]["ok"] and lines[3]["result"]["candidates"]
    assert lines[4]["code"] == "no_table" and lines[5]["code"] == "bad_request"
