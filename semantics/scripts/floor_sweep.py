#!/usr/bin/env python3
"""Relevancy-floor sweep over several lift tables (CPU): hits and absent-object false positives per floor, so
teacher variants can be compared at equal false positives instead of at the one floor (0.55) fixed on ViT-B/16.

Relevancy is computed once per query and table (memoised); only the threshold changes between floors.
A table's own calibrated floor (variant C2) is ignored here (table_floor=False), so every floor is absolute.

Usage (kitchen env, from semantics/):
  CUDA_VISIBLE_DEVICES= python scripts/floor_sweep.py --out floor_sweep.json \
      --tables Old=lift_pyr A=lift_ms A-flat=lift_ms:flat B2=lift_sam2 B2-flat=lift_sam2:flat C=lift_clipl
"""
import argparse
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import radiance_semantics.query as Q  # noqa: E402
from radiance_semantics.annotations import load, path_for  # noqa: E402
from radiance_semantics.evaluate import aggregate, evaluate, load_scene  # noqa: E402
from radiance_semantics.paths import find_scene_run  # noqa: E402
from radiance_semantics.store import read_table  # noqa: E402

FLOORS = [0.46, 0.48, 0.50, 0.51, 0.52, 0.525, 0.53, 0.54, 0.55, 0.575, 0.60]
_rel, _cache = Q.relevancy, {}


def _cached(feats, weight, q, negs, *a, **k):
    ai = feats.__array_interface__ if hasattr(feats, "__array_interface__") else np.asarray(feats).__array_interface__
    key = (ai["data"][0], feats.shape, ai.get("strides"), np.asarray(q).tobytes())
    if key not in _cache:
        _cache[key] = _rel(feats, weight, q, negs, *a, **k)
    return _cache[key]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=str(Path.home() / "Radiance/figs"))
    ap.add_argument("--scenes", nargs="+", default=["backroom", "flightroom"])
    ap.add_argument("--set", default="phase5")
    ap.add_argument("--tables", nargs="+", required=True, help="NAME=table_dir[:flat] (flat = no per-query scale choice)")
    ap.add_argument("--floors", nargs="+", type=float, default=FLOORS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    Q.relevancy = _cached
    has_tf = any(f.name == "table_floor" for f in fields(Q.Settings))
    out = []
    for scene in a.scenes:
        run = find_scene_run(Path(a.project_root), scene)
        qs = [q for q in load(path_for(run))["queries"]
              if q.get("set") == a.set and (q.get("negative") or q.get("position") is not None)]
        sc = load_scene(run.scene_dir)
        for spec in a.tables:
            name, _, tdir = spec.partition("=")
            tdir, _, flag = tdir.partition(":")
            _cache.clear()
            t = read_table(run.semantics_dir / run.run / tdir)
            enc = Q.TextEncoder.for_table(t)
            over = {"scale_select": False} if flag == "flat" else {}
            if has_tf:
                over["table_floor"] = False
            for f in a.floors:
                recs = evaluate(t, sc, qs, enc, replace(Q.Settings(), threshold=f, **over), log=lambda _: None)
                m = aggregate(recs)
                out.append({"scene": scene, "teachers": name, "table": tdir, "flat": flag == "flat", "floor": f,
                            **{k: m[k] for k in ("hits", "n_pos", "neg_fp_rate", "neg_auroc", "error_median_m")},
                            "hit": [r["text"] for r in recs if r.get("hit")],
                            "neg_with_candidate": [r["text"] for r in recs if r["negative"] and r["n_candidates"]]})
                print(f"{scene} {name} floor {f}: hits {m['hits']}/{m['n_pos']}, negatives with a candidate "
                      f"{out[-1]['neg_with_candidate']}", flush=True)
                Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
