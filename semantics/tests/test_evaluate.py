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
