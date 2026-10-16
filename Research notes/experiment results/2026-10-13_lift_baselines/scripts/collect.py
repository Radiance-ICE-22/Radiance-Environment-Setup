"""Copy the raw results of the 13 Oct lift-baseline runs into ../raw and derive the CSVs the report plots.

    python3 collect.py            (any python3 with the standard library; reads the host's semantics dirs)

raw/eval/<scene>/<variant>.json      evaluator output, Phase 5 set (frozen 6 Oct, backroom amended 7 Oct)
raw/eval_dev/<scene>/<variant>.json  evaluator output, development set (phase1_gate, 5 backroom queries)
raw/lift/<scene>/<table>.json        lift stats (time, peak VRAM, seen Gaussians) from each table's index
raw/floor_sweep.json, raw/table_agreement.json   written by floor_sweep.py / table_agreement.py
raw/logs/*.log                       the run logs
csv/*.csv                            derived tables (one row per method × teachers × query mode × scene)
"""
import csv
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW, CSV = ROOT / "raw", ROOT / "csv"
WS = Path.home() / "Radiance/figs/SousVide/gsplats/workspace"
RUNS = {"backroom": "2026-09-26_190524", "flightroom": "2026-09-29_163855"}
LOGS = Path.home() / "Radiance/third_party/logs"

# (method, teachers, eval variant suffix, table dir)
METHODS = [
    ("Ours", "Old", "@pyr", "lift_pyr"), ("LUDVIG", "Old", "@ludvigpyr", "lift_ludvigpyr"),
    ("Occam", "Old", "@occampyr", "lift_occampyr"),
    ("Ours", "A", "", "lift"), ("LUDVIG", "A", "@ludvigms", "lift_ludvigms"), ("Occam", "A", "@occamms", "lift_occamms"),
]
MODES = ["L-C", "L-CD"]


def sem(scene):
    return WS / scene / "semantics" / RUNS[scene]


def write_csv(name, rows):
    CSV.mkdir(parents=True, exist_ok=True)
    with open(CSV / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    # ── copy raw files ──
    for scene in RUNS:
        for m, t, suf, table in METHODS:
            for mode in MODES:
                src = sem(scene) / "eval" / f"{mode}{suf}.json"
                dst = RAW / "eval" / scene / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            idx = json.loads((sem(scene) / table / "index.json").read_text())
            meta = idx.get("meta", idx)
            raw_stats = sem(scene) / f"{table}_raw" / "lift.json"
            rec = {"table": table, "method": m, "teachers": t, "rows": idx.get("n"),
                   "table_mb": round(sum(p.stat().st_size for p in (sem(scene) / table).iterdir()) / 2**20, 1),
                   "lift": meta.get("metrics", {}).get("lift"),
                   "raw_lift_json": json.loads(raw_stats.read_text()) if raw_stats.exists() else None}
            d = RAW / "lift" / scene
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{table}.json").write_text(json.dumps(rec, indent=1) + "\n")
    dev = sem("backroom") / "eval_dev_baselines_2026-10-13"
    if dev.exists():
        (RAW / "eval_dev" / "backroom").mkdir(parents=True, exist_ok=True)
        for p in dev.glob("*.json"):
            shutil.copy2(p, RAW / "eval_dev" / "backroom" / p.name)
    (RAW / "logs").mkdir(parents=True, exist_ok=True)
    for name in ["ludvig_lifts.log", "occam_lifts.log", "eval_baselines.log", "install_ludvig.log",
                 "install_occamlgs.log"]:
        if (LOGS / name).exists():
            shutil.copy2(LOGS / name, RAW / "logs" / name)

    # ── derived CSVs ──
    summary, combined, per_query, cost = [], [], {}, []
    for m, t, suf, table in METHODS:
        for mode in MODES:
            tot = {"method": m, "teachers": t, "mode": mode, "hits": 0, "n_pos": 0, "neg_fp": 0, "n_neg": 0}
            for scene in RUNS:
                d = json.loads((RAW / "eval" / scene / f"{mode}{suf}.json").read_text())
                mt = d["metrics"]
                summary.append({"scene": scene, "method": m, "teachers": t, "mode": mode, "hits": mt["hits"],
                                "n_pos": mt["n_pos"], "hit_rate": mt["hit_rate"], "err_med_m": mt["error_median_m"],
                                "err_p90_m": mt["error_p90_m"], "ambiguous": mt["ambiguous_rate"],
                                "feasible": mt["feasible_rate"], "neg_fp": round(mt["neg_fp_rate"] * mt["n_neg"]),
                                "n_neg": mt["n_neg"], "auroc": mt["neg_auroc"], "query_ms": mt["query_ms_median"],
                                "no_candidate": mt["failures"].get("no candidate", 0),
                                "wrong_object": mt["failures"].get("wrong object", 0),
                                "bad_position": mt["failures"].get("bad position", 0)})
                tot["hits"] += mt["hits"]; tot["n_pos"] += mt["n_pos"]
                tot["neg_fp"] += round(mt["neg_fp_rate"] * mt["n_neg"]); tot["n_neg"] += mt["n_neg"]
                for r in d["records"]:
                    if r["negative"]:
                        continue
                    k = (scene, r["text"])
                    per_query.setdefault(k, {"scene": scene, "query": r["text"]})
                    per_query[k][f"{m}-{t}-{mode}"] = int(bool(r["hit"]))
                    per_query[k][f"{m}-{t}-{mode}_peak"] = r["peak"]
            combined.append(tot)
        for scene in RUNS:
            rec = json.loads((RAW / "lift" / scene / f"{table}.json").read_text())
            L = rec["lift"] or {}
            cost.append({"scene": scene, "method": m, "teachers": t, "lift_s": L.get("seconds"),
                         "peak_vram_mib": L.get("peak_vram_mib"), "seen": L.get("seen"), "rows": rec["rows"],
                         "seen_frac": round(L.get("seen", 0) / rec["rows"], 4), "views": L.get("views"),
                         "table_mb": rec["table_mb"]})
    write_csv("summary.csv", summary)
    write_csv("combined.csv", combined)
    write_csv("per_query.csv", list(per_query.values()))
    write_csv("lift_cost.csv", cost)

    devdir = RAW / "eval_dev" / "backroom"
    if devdir.exists():
        dev = []
        for p in sorted(devdir.glob("*.json")):
            d = json.loads(p.read_text())
            recs = [r for r in d["records"] if not r["negative"]]
            dev.append({"variant": d["variant"], "hits": f'{d["metrics"]["hits"]}/{d["metrics"]["n_pos"]}',
                        "err_med_m": d["metrics"]["error_median_m"],
                        "peaks": " ".join(f'{r["peak"]:.3f}' for r in recs)})
        write_csv("dev.csv", dev)

    fs = RAW / "floor_sweep.json"
    if fs.exists():
        sweep = json.loads(fs.read_text())
        write_csv("floor_sweep.csv", sweep_rows(sweep))
    ag = RAW / "table_agreement.json"
    if ag.exists():
        a = json.loads(ag.read_text())
        rows, hist = [], []
        for scene, pairs in a.items():
            for pair, r in pairs.items():
                for feat in ("clip", "dino", "clip_s0", "clip_s1", "clip_s2"):
                    if feat not in r:
                        continue
                    s = r[feat]
                    rows.append({"scene": scene, "pair": pair, "feature": feat, "seen_ref": r["seen_ref"],
                                 "seen_other": r["seen_other"], "seen_both": r["seen_both"], "mean": round(s["mean"], 5),
                                 **{f"q{q}": round(v, 5) for q, v in s["quantiles"].items()},
                                 "frac_gt_0.99": round(s["frac_above_0.99"], 5), "frac_gt_0.9": round(s["frac_above_0.9"], 5)})
                    if feat in ("clip", "dino"):
                        tot = sum(s["hist"])
                        for lo, c in zip(s["hist_edges"][:-1], s["hist"]):
                            hist.append({"scene": scene, "pair": pair, "feature": feat, "bin_lo": round(lo, 3),
                                         "frac": round(c / tot, 6)})
        write_csv("table_agreement.csv", rows)
        write_csv("table_agreement_hist.csv", hist)
    print("raw →", RAW, "\ncsv →", CSV)


def sweep_rows(sweep):
    """floor_sweep.json (a list of per scene × teachers × floor records) → rows summed over both scenes."""
    acc = {}
    for r in sweep:
        k = (r["teachers"], r["floor"])
        a = acc.setdefault(k, {"table": r["teachers"], "floor": r["floor"], "hits": 0, "n_pos": 0, "neg_fp": 0, "n_neg": 0})
        a["hits"] += r["hits"]; a["n_pos"] += r["n_pos"]
        a["neg_fp"] += len(r["neg_with_candidate"]); a["n_neg"] += 3
    return sorted(acc.values(), key=lambda r: (r["table"], r["floor"]))


if __name__ == "__main__":
    main()
