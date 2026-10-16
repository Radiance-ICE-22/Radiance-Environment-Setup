"""Copy the raw results of the 13-14 Oct full-pipeline runs into ../raw and derive the report's CSVs.

    python3 collect.py

Compared configurations (all scored by our unchanged query + evaluator on the Phase 5 sets):
  Ours-Old, Ours-A          our lift, scale-averaged / multi-scale CLIP teachers           (lift_pyr, lift)
  B2                        our lift, SAM ViT-B region teachers with pyramid fall-back      (lift_sam2, 8 Oct)
  LUDVIG-ctrl               LUDVIG's lift on our Old teachers                               (lift_ludvigpyr, 13 Oct)
  LUDVIG-full               LUDVIG's pipeline: ViT-g DINOv2 PCA-40 + its CLIP pyramid; L-CL = its diffusion
  Occam-ctrl                Occam's lift on our A teachers                                  (lift_occamms, 13 Oct)
  Occam-full                Occam's pipeline: LangSplat SAM ViT-H features, levels s/m/l    (lift_occamfull)
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
TP = Path.home() / "Radiance/third_party"
PREV = ROOT.parent / "2026-10-13_lift_baselines" / "raw"

# (config, table, [(mode, eval variant file stem)])
CONFIGS = [
    ("Ours-Old", "lift_pyr", [("L-C", "L-C@pyr"), ("L-CD", "L-CD@pyr")]),
    ("Ours-A", "lift", [("L-C", "L-C"), ("L-CD", "L-CD")]),
    ("B2", "lift_sam2", [("L-C", "L-C@sam2"), ("L-CD", "L-CD@sam2")]),
    ("LUDVIG-ctrl", "lift_ludvigpyr", [("L-C", "L-C@ludvigpyr"), ("L-CD", "L-CD@ludvigpyr")]),
    ("LUDVIG-full", "lift_ludvigfull", [("L-C", "L-C@ludvigfull"), ("L-CD", "L-CD@ludvigfull"),
                                        ("L-CL", "L-CL@ludvigfull")]),
    ("Occam-ctrl", "lift_occamms", [("L-C", "L-C@occamms"), ("L-CD", "L-CD@occamms")]),
    ("Occam-full", "lift_occamfull", [("L-C", "L-C@occamfull")]),
]


def sem(scene):
    return WS / scene / "semantics" / RUNS[scene]


def write_csv(name, rows):
    CSV.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(CSV / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    summary, combined, perq, cost = [], [], {}, []
    for cfg, table, modes in CONFIGS:
        for mode, stem in modes:
            tot = {"config": cfg, "mode": mode, "hits": 0, "n_pos": 0, "neg_fp": 0, "n_neg": 0}
            for scene in RUNS:
                src = sem(scene) / "eval" / f"{stem}.json"
                dst = RAW / "eval" / scene / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                d = json.loads(src.read_text())
                assert d["set"] == "phase5", (src, d["set"])
                m = d["metrics"]
                fp = round(m["neg_fp_rate"] * m["n_neg"])
                summary.append({"scene": scene, "config": cfg, "mode": mode, "hits": m["hits"], "n_pos": m["n_pos"],
                                "err_med_m": m["error_median_m"], "err_p90_m": m["error_p90_m"], "neg_fp": fp,
                                "n_neg": m["n_neg"], "auroc": m["neg_auroc"], "feasible": m["feasible_rate"],
                                "query_ms": m["query_ms_median"],
                                "no_candidate": m["failures"].get("no candidate", 0),
                                "wrong_object": m["failures"].get("wrong object", 0),
                                "bad_position": m["failures"].get("bad position", 0)})
                tot["hits"] += m["hits"]; tot["n_pos"] += m["n_pos"]; tot["neg_fp"] += fp; tot["n_neg"] += m["n_neg"]
                for r in d["records"]:
                    if r["negative"]:
                        continue
                    k = (scene, r["text"])
                    perq.setdefault(k, {"scene": scene, "query": r["text"]})
                    perq[k][f"{cfg}-{mode}"] = int(bool(r["hit"]))
                    perq[k][f"{cfg}-{mode}_peak"] = r["peak"]
            combined.append(tot)
        for scene in RUNS:                                             # lift statistics
            idx = json.loads((sem(scene) / table / "index.json").read_text())
            meta = idx.get("meta", idx)
            L = meta.get("metrics", {}).get("lift") or {}
            d = RAW / "lift" / scene
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{table}.json").write_text(json.dumps({"config": cfg, "table": table, "rows": idx.get("n"),
                                                          "lift": L}, indent=1) + "\n")
            feat_s = None
            if cfg == "Occam-full":
                lm = json.loads((TP / "langsplat_features" / scene / "meta.json").read_text())
                shutil.copy2(TP / "langsplat_features" / scene / "meta.json", d / "langsplat_meta.json")
                feat_s = lm["seconds"]
            if cfg == "LUDVIG-full":
                feat_s = L.get("dino_seconds", 0) + L.get("clip_seconds", 0)
            cost.append({"scene": scene, "config": cfg, "lift_s": L.get("seconds"), "features_s": feat_s,
                         "diffusion_s": L.get("diffusion_seconds"), "seen": L.get("seen"), "rows": idx.get("n"),
                         "seen_frac": round((L.get("seen") or 0) / idx.get("n"), 4),
                         "table_mb": round(sum(p.stat().st_size for p in (sem(scene) / table).iterdir()) / 2**20, 1)})
    for scene in RUNS:                                                 # development set (new eval/<set>/ layout)
        dd = sem(scene) / "eval" / "phase1_gate"
        if dd.exists():
            for p in dd.glob("*full.json"):
                (RAW / "eval_dev" / scene).mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, RAW / "eval_dev" / scene / p.name)
    (RAW / "logs").mkdir(parents=True, exist_ok=True)
    for name in ["full_pipelines.log", "occam_full.log", "ludvig_full_eval.log", "full_eval.log", "test_langsplat_nms.log"]:
        if (TP / "logs" / name).exists():
            shutil.copy2(TP / "logs" / name, RAW / "logs" / name)
    write_csv("summary.csv", summary)
    write_csv("combined.csv", combined)
    write_csv("per_query.csv", list(perq.values()))
    write_csv("cost.csv", cost)

    dev = []
    for scene in RUNS:
        for p in sorted((RAW / "eval_dev" / scene).glob("*.json")) if (RAW / "eval_dev" / scene).exists() else []:
            d = json.loads(p.read_text())
            recs = [r for r in d["records"] if not r["negative"]]
            dev.append({"variant": d["variant"], "hits": f'{d["metrics"]["hits"]}/{d["metrics"]["n_pos"]}',
                        "err_med_m": d["metrics"]["error_median_m"],
                        "missed": "; ".join(f'{r["text"]} ({r["failure"]})' for r in recs if not r["hit"]) or "-"})
    if dev:
        write_csv("dev.csv", dev)

    sweep = json.loads((PREV / "floor_sweep.json").read_text())       # Old / A curves from 13 Oct
    sweep = [r for r in sweep if r["teachers"] in ("Old", "A")]
    if (RAW / "floor_sweep.json").exists():
        sweep += json.loads((RAW / "floor_sweep.json").read_text())
    acc = {}
    for r in sweep:
        k = (r["teachers"], r["floor"])
        a = acc.setdefault(k, {"table": r["teachers"], "floor": r["floor"], "hits": 0, "n_pos": 0, "neg_fp": 0})
        a["hits"] += r["hits"]; a["n_pos"] += r["n_pos"]; a["neg_fp"] += len(r["neg_with_candidate"])
    write_csv("floor_sweep.csv", sorted(acc.values(), key=lambda r: (r["table"], r["floor"])))
    print("raw →", RAW, "\ncsv →", CSV)


if __name__ == "__main__":
    main()
