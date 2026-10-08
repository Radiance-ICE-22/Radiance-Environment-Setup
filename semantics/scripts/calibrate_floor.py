#!/usr/bin/env python3
"""Calibrate the relevancy floor for a table built with another CLIP model (variant C: ViT-L/14).

The query keeps Gaussians whose LERF relevancy clears a floor (0.55), fixed on ViT-B/16. Relevancy is a sigmoid of
similarity differences, and a different CLIP puts similarities on a different scale, so the same number means
something else. This maps the floor by quantiles, without using any evaluation query:

  for each word of a NEUTRAL vocabulary (everyday objects that appear in no Phase 5 or development query) and each
  scene: q = the fraction of the reference table's Gaussians (ViT-B/16) whose relevancy is below 0.55, and the
  floor for the target table = the target model's relevancy at that same quantile q.
  Words whose reference relevancy never reaches the floor (q ≈ 1) only probe the target's maximum, so the floor is
  mapped on the POOLED relevancies of all words per scene (one quantile over every word × Gaussian), and the mean
  over scenes is written into each target table's index.json ("query": {"threshold": …}). The per-word median is
  printed for comparison.

Usage (CPU, kitchen env, from semantics/):
  python scripts/calibrate_floor.py --ref lift_pyr --target lift_clipl --write lift_clipl fmgs_clipl fmgs_c_clipl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radiance_semantics.paths import find_scene_run  # noqa: E402
from radiance_semantics.query import NEGATIVES, Settings, TextEncoder, relevancy  # noqa: E402
from radiance_semantics.store import read_table  # noqa: E402

# Everyday objects absent from every Phase 5 query (incl. synonyms and negatives) and from the dev set.
VOCAB = ["laptop", "printer", "book", "scissors", "screwdriver", "hammer", "cable", "shelf", "door", "window",
         "ceiling light", "desk", "extension cord", "shoe", "jacket", "plastic bag", "paper", "pen", "telephone",
         "ladder", "drone", "fire extinguisher", "sign", "pipe", "light switch", "power outlet", "glove", "helmet",
         "battery", "headphones"]


def table_dir(root, scene, name):
    run = find_scene_run(root, scene)
    return run.semantics_dir / run.run / name


def floor_at_quantiles(ref, tgt, enc_ref, enc_tgt, words, base_floor):
    """([(word, q, target floor)], pooled floor) for one scene."""
    out, pool_r, pool_t = [], [], []
    keep = (np.asarray(ref.geom[:, 3]) >= Settings.min_opacity) & (np.asarray(ref.weight) > 0)
    for w in words:
        Er = enc_ref.encode([w] + NEGATIVES)
        Et = enc_tgt.encode([w] + NEGATIVES)
        rr = relevancy(ref.clip, ref.weight, Er[0], Er[1:])[keep]
        rt = relevancy(tgt.clip, tgt.weight, Et[0], Et[1:])[keep]
        q = float((rr < base_floor).mean())
        out.append((w, q, float(np.quantile(rt, q))))
        pool_r.append(rr); pool_t.append(rt)
    pr, pt = np.concatenate(pool_r), np.concatenate(pool_t)
    return out, float(np.quantile(pt, float((pr < base_floor).mean())))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=str(Path.home() / "Radiance/figs"))
    ap.add_argument("--scenes", nargs="+", default=["backroom", "flightroom"])
    ap.add_argument("--ref", default="lift_pyr", help="reference table (the model the floor was set on)")
    ap.add_argument("--target", default="lift_clipl", help="table of the model to calibrate")
    ap.add_argument("--write", nargs="*", default=[], help="tables (of the target model) that get the floor")
    a = ap.parse_args(argv)
    root = Path(a.project_root)
    rows, pooled = [], []
    for scene in a.scenes:
        ref, tgt = read_table(table_dir(root, scene, a.ref)), read_table(table_dir(root, scene, a.target))
        er, et = TextEncoder.for_table(ref), TextEncoder.for_table(tgt)
        per_word, pf = floor_at_quantiles(ref, tgt, er, et, VOCAB, Settings.threshold)
        pooled.append(pf)
        for w, q, f in per_word:
            rows.append((scene, w, q, f))
            print(f"  {scene:<10} {w:<18} q {q:.4f}  floor {f:.4f}", flush=True)
        print(f"  {scene}: pooled floor {pf:.4f}")
    floors = np.array([r[3] for r in rows])
    floor = float(np.mean(pooled))
    print(f"calibrated floor (pooled, mean over scenes): {floor:.4f}  [per scene {', '.join(f'{p:.4f}' for p in pooled)}]; "
          f"per-word median {np.median(floors):.4f} (IQR {np.percentile(floors, 25):.4f}–{np.percentile(floors, 75):.4f}); "
          f"reference floor {Settings.threshold}")
    for scene in a.scenes:
        for name in a.write:
            p = table_dir(root, scene, name) / "index.json"
            ix = json.loads(p.read_text())
            ix["query"] = {"threshold": round(floor, 4), "calibrated": {
                "method": "pooled quantile map of the reference floor over a neutral vocabulary", "reference": a.ref,
                "per_scene": [round(p, 4) for p in pooled],
                "reference_floor": Settings.threshold, "words": len(VOCAB), "scenes": a.scenes,
                "iqr": [round(float(np.percentile(floors, 25)), 4), round(float(np.percentile(floors, 75)), 4)]}}
            p.write_text(json.dumps(ix, indent=2) + "\n")
            print(f"  wrote floor {floor:.4f} into {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
