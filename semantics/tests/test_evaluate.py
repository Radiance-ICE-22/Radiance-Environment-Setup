"""Phase 5 evaluation: multi-instance scoring, failure kinds, aggregation, AUROC, CSV and LaTeX writers."""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radiance_semantics import evaluate as E  # noqa: E402


def cand(c, half=0.2, rank=1):
    c = np.asarray(c, float)
    return {"rank": rank, "centroid": c.tolist(), "box": {"lo": (c - half).tolist(), "hi": (c + half).tolist()}}


def test_a_hit_on_any_instance_counts():
    res = {"candidates": [cand([5, 0, 0]), cand([0, 0, 0], rank=2)]}
    sc = E.score_instances(res, [[0, 0, 0], [5.1, 0, 0]])
    assert sc["hit"] and sc["instance"] == 1 and abs(sc["error"] - 0.1) < 1e-9 and sc["rank_of_hit"] == 1
    miss = E.score_instances({"candidates": [cand([3, 0, 0])]}, [[0, 0, 0], [6, 0, 0]])
    assert not miss["hit"] and miss["error"] == 3.0
    assert E.score_instances({"candidates": []}, [[0, 0, 0]])["error"] is None


def test_failure_kinds():
    assert E.failure_kind({"hit": True, "error": 0.1}) is None
    assert E.failure_kind({"hit": False, "error": None}) == "no candidate"
    assert E.failure_kind({"hit": False, "error": 1.2}) == "bad position"
    assert E.failure_kind({"hit": False, "error": 4.0}) == "wrong object"


def test_auroc():
    assert E.auroc([0.9, 0.8], [0.6, 0.5]) == 1.0
    assert E.auroc([0.5], [0.5]) == 0.5
    assert E.auroc([0.6, 0.4], [0.5]) == 0.5
    assert E.auroc([], [0.5]) is None


def recs():
    return [
        {"text": "a", "negative": False, "hit": True, "error": 0.2, "ambiguous": False, "gap_ok": True, "peak": 0.80, "n_candidates": 2, "ms": 700},
        {"text": "b", "negative": False, "hit": True, "error": 0.4, "ambiguous": True, "gap_ok": False, "peak": 0.70, "n_candidates": 1, "ms": 800},
        {"text": "c", "negative": False, "hit": False, "error": 3.0, "ambiguous": False, "gap_ok": True, "peak": 0.65, "n_candidates": 3, "ms": 750},
        {"text": "d", "negative": False, "hit": False, "error": None, "ambiguous": False, "gap_ok": None, "peak": 0.60, "n_candidates": 0, "ms": 760},
        {"text": "n1", "negative": True, "peak": 0.62, "n_candidates": 1, "ambiguous": False, "gap_ok": None, "ms": 740},
        {"text": "n2", "negative": True, "peak": 0.58, "n_candidates": 0, "ambiguous": False, "gap_ok": None, "ms": 720},
    ]


def test_aggregate():
    m = E.aggregate(recs())
    assert m["n_pos"] == 4 and m["n_neg"] == 2 and m["hits"] == 2 and m["hit_rate"] == 0.5
    assert m["error_median_m"] == 0.4                     # over the three with a candidate
    assert m["ambiguous_rate"] == 0.25 and m["feasible_rate"] == 0.5 and m["neg_fp_rate"] == 0.5
    assert m["neg_auroc"] == round((2 + 2 + 2 + 0.0 + 1) / 8, 3)   # 0.8,0.7,0.65 beat both; 0.60 beats 0.58 only
    assert m["failures"] == {"wrong object": 1, "no candidate": 1} and m["query_ms_median"] == 745


def test_csv_and_latex(tmp_path):
    rows = [{"scene": "backroom", "variant": "L-C", **E.aggregate(recs()), "build_min": 3.0, "peak_vram_gb": 7.4, "table_mb": 924.0},
            {"scene": "GTN_lab_v1", "variant": "F-CD", **E.aggregate(recs()), "build_min": None}]
    E.write_csv(rows, tmp_path / "v.csv")
    got = list(csv.reader(open(tmp_path / "v.csv")))
    assert got[0][:3] == ["scene", "variant", "hit_rate"] and got[1][1] == "L-C" and len(got) == 3
    E.write_latex(rows, tmp_path / "v.tex")
    tex = (tmp_path / "v.tex").read_text()
    assert r"\toprule" in tex and "GTN\\_lab\\_v1" in tex and tex.count(r"\midrule") == 2 and "--" in tex


def test_build_cost_of_a_resumed_fmgs_run(tmp_path):
    class T:
        path = tmp_path
        index = {"metrics": {"fmgs": {"steps": 4200, "it_per_s": 2.0, "seconds": 600.0, "peak_vram_mib_device": 7680}}}
    (tmp_path / "clip.f16").write_bytes(b"\0" * 2 ** 20)
    c = E.build_cost(T)
    assert c["build_min"] == 35.0 and c["peak_vram_gb"] == 7.5 and c["table_mb"] == 1.0


def test_parse_variant():
    assert E.parse_variant("L-C") == ("lift", {}, "")
    assert E.parse_variant("L-CD@480") == ("lift", {"dino_diffuse": True, "dino_split": True}, "_w480")
    import pytest
    assert E.parse_variant("F-CD@ms") == ("fmgs", {}, "_ms")
    for bad in ("X", "L-C@Wide!", "L-C@" + "a" * 17):
        with pytest.raises(ValueError):
            E.parse_variant(bad)


def test_eval_dir_keeps_other_sets_out_of_the_phase5_path(tmp_path):
    from types import SimpleNamespace
    import pytest
    run = SimpleNamespace(semantics_dir=tmp_path, run="r1")
    assert E.eval_dir(run) == E.eval_dir(run, "phase5") == tmp_path / "r1" / "eval"
    assert E.eval_dir(run, "phase1_gate") == tmp_path / "r1" / "eval" / "phase1_gate"
    for bad in ("", "..", "a/b", "../x"):
        with pytest.raises(ValueError):
            E.eval_dir(run, bad)


def test_run_variants_on_another_set_leaves_phase5_files_alone(tmp_path, monkeypatch):
    """--set phase1_gate writes eval/phase1_gate/<variant>.json; eval/<variant>.json and --report are untouched."""
    import json
    from types import SimpleNamespace
    from radiance_semantics import annotations, paths, query, store
    run = SimpleNamespace(semantics_dir=tmp_path, run="r1", key="k1", scene_dir=tmp_path,
                          backend_dir=lambda b, s="": tmp_path / "r1" / f"{b}{s}")
    (tmp_path / "r1" / "lift").mkdir(parents=True)
    (tmp_path / "r1" / "lift" / "index.json").write_text("{}")
    phase5 = tmp_path / "r1" / "eval" / "L-C.json"
    phase5.parent.mkdir(parents=True)
    phase5.write_text(json.dumps({"set": "phase5", "metrics": {"hits": 22}}))
    qs = [{"text": "chair", "set": "phase1_gate", "position": [0, 0, 0]},
          {"text": "sofa", "set": "phase5", "position": [1, 1, 1]}]
    monkeypatch.setattr(paths, "find_scene_run", lambda root, scene: run)
    monkeypatch.setattr(annotations, "path_for", lambda r: tmp_path / "queries.json")
    monkeypatch.setattr(annotations, "load", lambda p: {"queries": qs})
    monkeypatch.setattr(store, "read_table", lambda d: SimpleNamespace(index={"key": "k1"}))
    monkeypatch.setattr(query, "TextEncoder", lambda: SimpleNamespace(_key=(None, None)))
    monkeypatch.setattr(E, "load_scene", lambda d: None)
    seen = []
    monkeypatch.setattr(E, "evaluate", lambda t, sc, q, enc, s, log: seen.extend(x["text"] for x in q) or recs())
    monkeypatch.setattr(E, "build_cost", lambda t, m: {})
    E.run_variants(tmp_path, "backroom", ["L-C"], "phase1_gate", log=lambda _: None)
    assert seen == ["chair"]
    out = json.loads((tmp_path / "r1" / "eval" / "phase1_gate" / "L-C.json").read_text())
    assert out["set"] == "phase1_gate"
    assert json.loads(phase5.read_text())["metrics"]["hits"] == 22                 # the Phase 5 result survives
    assert E.report_rows(tmp_path, ["backroom"], ["L-C"])[0]["hits"] == 22
    assert E.report_rows(tmp_path, ["backroom"], ["L-C"], "phase1_gate")[0]["hits"] == out["metrics"]["hits"]
