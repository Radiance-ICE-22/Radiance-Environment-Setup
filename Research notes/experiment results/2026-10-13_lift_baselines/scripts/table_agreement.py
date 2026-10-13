"""Per-Gaussian agreement between lift tables of one scene (kitchen env, CPU).

    python table_agreement.py --out ../raw/table_agreement.json

For each scene and pair (reference = our lift, other = LUDVIG / Occam's on the same teachers): seen counts,
and over Gaussians seen by both, the cosine between the unit rows (CLIP, DINO, and for A each scale group),
summarised as quantiles plus a histogram on [0, 1] (rows are unit length, so cosine = dot product).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "Radiance/Radiance-Environment-Setup/semantics"))
from radiance_semantics.paths import find_scene_run          # noqa: E402
from radiance_semantics.store import read_table              # noqa: E402

PAIRS = [("lift_pyr", "lift_ludvigpyr"), ("lift_pyr", "lift_occampyr"),
         ("lift_ms", "lift_ludvigms"), ("lift_ms", "lift_occamms"), ("lift_ludvigpyr", "lift_occampyr")]
BINS = np.linspace(0, 1, 41)
Q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


def cos_rows(a, b, idx, chunk=200_000):
    out = np.empty(len(idx), np.float32)
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        x, y = np.asarray(a[j], np.float32), np.asarray(b[j], np.float32)
        out[i:i + chunk] = (x * y).sum(-1)
    return out


def summary(c):
    h, _ = np.histogram(np.clip(c, 0, 1), BINS)
    return {"n": int(len(c)), "mean": float(c.mean()), "quantiles": {str(q): float(np.quantile(c, q)) for q in Q},
            "frac_above_0.99": float((c > 0.99).mean()), "frac_above_0.9": float((c > 0.9).mean()),
            "hist_edges": BINS.tolist(), "hist": h.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", nargs="+", default=["backroom", "flightroom"])
    a = ap.parse_args()
    res = {}
    for sc in a.scenes:
        run = find_scene_run(Path.home() / "Radiance/figs", sc)
        base = run.semantics_dir / run.run
        res[sc] = {}
        for ref, oth in PAIRS:
            A, B = read_table(base / ref), read_table(base / oth)
            sa, sb = np.asarray(A.weight) > 0, np.asarray(B.weight) > 0
            idx = np.nonzero(sa & sb)[0]
            r = {"n": int(A.n), "seen_ref": int(sa.sum()), "seen_other": int(sb.sum()), "seen_both": int(len(idx)),
                 "clip": summary(cos_rows(A.clip, B.clip, idx))}
            if A.dino is not None and B.dino is not None:
                r["dino"] = summary(cos_rows(A.dino, B.dino, idx))
            if A.clip_scales is not None and B.clip_scales is not None:
                for g in range(A.clip_scales.shape[1]):
                    r[f"clip_s{g}"] = summary(cos_rows(A.clip_scales[:, g], B.clip_scales[:, g], idx))
            res[sc][f"{ref}__{oth}"] = r
            print(sc, ref, oth, {k: round(v["quantiles"]["0.5"], 4) for k, v in r.items() if isinstance(v, dict)},
                  "seen", r["seen_ref"], r["seen_other"], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
