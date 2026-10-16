"""Raw results of the 16 Oct region-teacher runs (semantic/new: --clip-mode langsplat, --uncovered skip|fill) and the
CSVs / LaTeX tables the report reads.   python3 collect.py
"""
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW, CSV, TAB = ROOT / "raw", ROOT / "csv", ROOT / "report" / "tables"
WS = Path.home() / "Radiance/figs/SousVide/gsplats/workspace"
RUNS = {"backroom": "2026-09-26_190524", "flightroom": "2026-09-29_163855"}
PREV = ROOT.parent
# config, table, [(mode, eval stem)]
CFG = [("Ours A (default)", "lift", [("L-C", "L-C"), ("L-CD", "L-CD")]),
       ("Ours B2 (SAM ViT-B, area levels, fill)", "lift_sam2", [("L-C", "L-C@sam2"), ("L-CD", "L-CD@sam2")]),
       ("Occam's full chain (centre-pixel lift)", "lift_occamfull", [("L-C", "L-C@occamfull")]),
       ("Ours + LangSplat teachers, skip", "lift_ls", [("L-C", "L-C@ls"), ("L-CD", "L-CD@ls")]),
       ("Ours + LangSplat teachers, fill", "lift_lsfill", [("L-C", "L-C@lsfill"), ("L-CD", "L-CD@lsfill")])]
SHORT = {"lift": "A", "lift_sam2": "B2", "lift_occamfull": "Occam-full", "lift_ls": "skip", "lift_lsfill": "fill"}


def sem(scene):
    return WS / scene / "semantics" / RUNS[scene]


def write_csv(name, rows):
    CSV.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(CSV / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def esc(s):
    return s.replace("&", r"\&").replace("_", r"\_")


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    rows, perq, summ = [], {}, []
    for cfg, table, modes in CFG:
        for mode, stem in modes:
            tot = [0, 0]
            cells = [cfg if mode == "L-C" else "", mode]
            per = []
            for scene in RUNS:
                src = sem(scene) / "eval" / f"{stem}.json"
                (RAW / "eval" / scene).mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, RAW / "eval" / scene / src.name)
                d = json.loads(src.read_text())
                assert d["set"] == "phase5"
                m = d["metrics"]
                fp = round(m["neg_fp_rate"] * m["n_neg"])
                tot[0] += m["hits"]; tot[1] += fp
                rows.append({"scene": scene, "config": SHORT[table], "mode": mode, "hits": m["hits"], "n_pos": m["n_pos"],
                             "err_med_m": m["error_median_m"], "neg_fp": fp, "auroc": m["neg_auroc"],
                             "failures": json.dumps(m["failures"])})
                per += [f'{m["hits"]}/{m["n_pos"]}', f'{m["error_median_m"]:.2f}', f'{m["neg_auroc"]:.2f}']
                if mode == "L-C":
                    for r in d["records"]:
                        if not r["negative"]:
                            perq.setdefault((scene, r["text"]), {"scene": scene, "query": r["text"]})[SHORT[table]] = \
                                (int(bool(r["hit"])), r["peak"], r.get("failure"))
            bold = table == "lift_ls" and mode == "L-C"
            cells += [(rf"\textbf{{{tot[0]}}}" if bold else str(tot[0])) + "/29", f"{tot[1]}/6"] + per
            summ.append(" & ".join(cells) + r" \\")
        summ.append(r"\addlinespace[2pt]")
    write_csv("summary.csv", rows)
    (TAB / "summary.tex").write_text("\n".join(summ[:-1]) + "\n")

    order = ["A", "B2", "Occam-full", "fill", "skip"]
    lines, out = [], []
    for (scene, q), r in perq.items():
        hits = [r[c][0] for c in order]
        out.append({"scene": scene, "query": q, **{c: r[c][0] for c in order}, **{c + "_peak": r[c][1] for c in order}})
        if len(set(hits)) > 1:
            lines.append(" & ".join([f"{esc(q)} ({scene[0]})"] + [(r"\hit" if r[c][0] else r"\miss") + f"\\,{r[c][1]:.3f}"
                                                                 for c in order]) + r" \\")
    write_csv("per_query.csv", out)
    (TAB / "per_query_diff.tex").write_text("\n".join(lines) + "\n")
    tot = [sum(o[c] for o in out) for c in order]
    (TAB / "per_query_total.tex").write_text("Total (of 29) & " + " & ".join(map(str, tot)) + r" \\" + "\n")

    for scene in RUNS:                                                    # dev set
        dd = sem(scene) / "eval" / "phase1_gate"
        for p in sorted(dd.glob("L-C@ls*.json")) if dd.exists() else []:
            (RAW / "eval_dev" / scene).mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, RAW / "eval_dev" / scene / p.name)

    cost = []                                                             # lift statistics
    for scene in RUNS:
        for t in ("lift_ls", "lift_lsfill", "lift_occamfull", "lift"):
            idx = json.loads((sem(scene) / t / "index.json").read_text())
            L = idx.get("meta", idx).get("metrics", {}).get("lift") or {}
            cost.append({"scene": scene, "table": t, "seconds": L.get("seconds"), "passes": L.get("passes"),
                         "seen": L.get("seen"), "level_rows": L.get("level_rows"), "pixel_coverage": L.get("pixel_coverage")})
            (RAW / "lift" / scene).mkdir(parents=True, exist_ok=True)
            (RAW / "lift" / scene / f"{t}.json").write_text(json.dumps(L, indent=1) + "\n")
    write_csv("cost.csv", cost)

    sweep = json.loads((RAW / "floor_sweep.json").read_text())
    sweep += [r for r in json.loads((PREV / "2026-10-13_lift_baselines/raw/floor_sweep.json").read_text())
              if r["teachers"] in ("Old", "A")]
    sweep += [r for r in json.loads((PREV / "2026-10-14_full_pipelines/raw/floor_sweep.json").read_text())
              if r["teachers"] in ("Occam-full", "Occam-full-flat")]
    acc = defaultdict(lambda: [0, 0])
    for r in sweep:
        a = acc[(r["teachers"], r["floor"])]
        a[0] += r["hits"]; a[1] += len(r["neg_with_candidate"])
    names = ["A", "Occam-full", "Occam-full-flat", "LS-skip", "LS-skip-flat", "LS-fill", "LS-fill-flat"]
    floors = sorted({fl for _, fl in acc})
    write_csv("floor_sweep.csv", [{"table": t, "floor": fl, "hits": acc[(t, fl)][0], "neg_fp": acc[(t, fl)][1]}
                                  for t in names for fl in floors])
    (ROOT / "report" / "data").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "report" / "data" / "sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["floor"] + [n.replace("-", "") + "_h" for n in names] + [n.replace("-", "") + "_fp" for n in names])
        for fl in floors:
            w.writerow([fl] + [acc[(n, fl)][0] for n in names] + [acc[(n, fl)][1] for n in names])
    lab = {"A": "Ours A (default)", "Occam-full": "Occam's full chain", "Occam-full-flat": "\\quad no level choice",
           "LS-skip": "Ours + LangSplat, skip", "LS-skip-flat": "\\quad no level choice",
           "LS-fill": "Ours + LangSplat, fill", "LS-fill-flat": "\\quad no level choice"}
    env = []
    for n in names:
        best = {}
        for fl in floors:
            h, fp = acc[(n, fl)]
            for k in range(fp, 7):
                best[k] = max(best.get(k, 0), h)
        env.append(" & ".join([lab[n]] + [str(best.get(k, "--")) for k in range(1, 6)] + [str(acc[(n, 0.55)][0])]) + r" \\")
    (TAB / "envelope.tex").write_text("\n".join(env) + "\n")
    (RAW / "logs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path.home() / "Radiance/third_party/logs/ls_teacher.log", RAW / "logs" / "ls_teacher.log")
    print("→", ROOT)


if __name__ == "__main__":
    main()
