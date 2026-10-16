"""Tables and plot data for the Occam's-full-chain report (after collect.py and occam_analysis.py).

    python3 make_occam_tables.py     → ../report/occam/tables/*.tex, ../report/occam/data/*.csv
"""
import csv
import json
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW, CSV = ROOT / "raw", ROOT / "csv"
OUT = ROOT / "report" / "occam"
TAB, DATA = OUT / "tables", OUT / "data"
RUNS = ("backroom", "flightroom")
# configuration → eval file stem, label
CFG = [("Occam-full", "L-C@occamfull", "Occam full"), ("Occam-ctrl", "L-C@occamms", "Occam lift, A"),
       ("Ours-A", "L-C", "Ours A"), ("Ours-Old", "L-C@pyr", "Ours Old"), ("B2", "L-C@sam2", "B2")]
LEVEL = {0: "s", 1: "m", 2: "l"}
GROUP = {0: "S", 1: "M", 2: "L"}


def esc(s):
    return s.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def write(name, text):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / name).write_text(text)


def wide(name, header, rows):
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def evals():
    E = {}
    for scene in RUNS:
        for cfg, stem, _ in CFG:
            d = json.loads((RAW / "eval" / scene / f"{stem}.json").read_text())
            E[(scene, cfg)] = d
    return E


def main():
    E = evals()
    L = json.loads((RAW / "occam" / "levels.json").read_text())

    # headline table: per scene metrics for each configuration
    lines = []
    for cfg, stem, lab in CFG:
        cells = [lab]
        tot_h = tot_fp = 0
        for scene in RUNS:
            m = E[(scene, cfg)]["metrics"]
            fp = round(m["neg_fp_rate"] * m["n_neg"])
            tot_h += m["hits"]; tot_fp += fp
            cells += [f'{m["hits"]}/{m["n_pos"]}', f'{m["error_median_m"]:.2f}', f'{m["error_p90_m"]:.2f}', f"{fp}/3",
                      f'{m["neg_auroc"]:.2f}']
        cells.insert(1, (rf"\textbf{{{tot_h}}}" if cfg == "Occam-full" else str(tot_h)) + "/29")
        cells.insert(2, f"{tot_fp}/6")
        lines.append(" & ".join(cells) + r" \\")
    write("headline.tex", "\n".join(lines) + "\n")

    # per-query: hit/miss + peak for Occam-full, Occam-ctrl, Ours-A, Ours-Old; Occam-full level + error
    lines = []
    for scene in RUNS:
        lines.append(r"\midrule" + "\n" + rf"\multicolumn{{8}}{{l}}{{\textit{{{scene}}}}}\\")
        recs = {cfg: {r["text"]: r for r in E[(scene, cfg)]["records"]} for cfg, _, _ in CFG}
        lev = {r["text"]: r for r in L[scene]["Occam-full"]}
        for text, r in recs["Occam-full"].items():
            if r["negative"]:
                continue
            row = [esc(text)]
            hits = []
            for cfg in ("Occam-full", "Occam-ctrl", "Ours-A", "Ours-Old"):
                x = recs[cfg][text]
                hits.append(bool(x["hit"]))
                mark = r"\hit" if x["hit"] else r"\miss"
                row.append(rf"{mark}\,{x['peak']:.2f}")
            lv = lev[text]["scale_group"]
            row.append(LEVEL[lv] if lv is not None else "--")
            row.append("--" if r["error"] is None else f'{r["error"]:.2f}')
            row.append(esc(r["failure"] or ""))
            if len(set(hits)) > 1:
                row[0] = rf"\textbf{{{row[0]}}}"
            lines.append(" & ".join(row) + r" \\")
    write("per_query.tex", "\n".join(lines) + "\n")

    # absent objects
    lines = []
    for scene in RUNS:
        recs = {cfg: {r["text"]: r for r in E[(scene, cfg)]["records"]} for cfg, _, _ in CFG}
        for text, r in recs["Occam-full"].items():
            if not r["negative"]:
                continue
            row = [f"{esc(text)} ({scene})"]
            for cfg in ("Occam-full", "Occam-ctrl", "Ours-A", "Ours-Old"):
                x = recs[cfg][text]
                row.append(("fires" if x["n_candidates"] else "--") + f"\\,{x['peak']:.2f}")
            lines.append(" & ".join(row) + r" \\")
    write("absent.tex", "\n".join(lines) + "\n")

    # level choice counts
    lines = []
    for cfg, lab in (("Occam-full", "Occam full (SAM levels s / m / l)"), ("Occam-ctrl", "Occam lift, A (scale groups S / M / L)"),
                     ("Ours-A", "Ours A (scale groups S / M / L)")):
        cells = [lab]
        for scene in RUNS:
            c = Counter(r["scale_group"] for r in L[scene][cfg] if not r["negative"])
            cells += [str(c.get(0, 0)), str(c.get(1, 0)), str(c.get(2, 0))]
        lines.append(" & ".join(cells) + r" \\")
    write("levels.tex", "\n".join(lines) + "\n")

    # peaks per level (positives vs absent), Occam full
    lines = []
    for scene in RUNS:
        pos = [r for r in L[scene]["Occam-full"] if not r["negative"]]
        neg = [r for r in L[scene]["Occam-full"] if r["negative"]]
        cells = [scene]
        for g in range(3):
            cells += [f'{st.median(r["scale_peaks"][g] for r in pos):.3f}', f'{max(r["scale_peaks"][g] for r in neg):.3f}']
        lines.append(" & ".join(cells) + r" \\")
    write("level_peaks.tex", "\n".join(lines) + "\n")

    # masks per level and coverage
    M = list(csv.DictReader(open(RAW / "occam" / "masks.csv")))
    lines = []
    for scene in RUNS:
        R = [r for r in M if r["scene"] == scene]
        for lv in ("default", "s", "m", "l"):
            c = sorted(int(r[f"masks_{lv}"]) for r in R)
            v = sorted(float(r[f"cover_{lv}"]) for r in R)
            used = "" if lv == "default" else r"\checkmark"
            lines.append(" & ".join([scene if lv == "default" else "", lv, used, str(int(st.median(c))),
                                     f"{c[len(c) // 10]}--{c[9 * len(c) // 10]}", f"{100 * st.median(v):.1f}",
                                     f"{100 * v[0]:.1f}"]) + r" \\")
        lines.append(r"\addlinespace[2pt]")
    write("masks.tex", "\n".join(lines[:-1]) + "\n")

    # timing
    T = list(csv.DictReader(open(RAW / "occam" / "timing.csv")))
    TL = json.loads((RAW / "occam" / "table_levels.json").read_text())
    lines = []
    for scene in RUNS:
        s = sorted(float(r["seconds"]) for r in T if r["scene"] == scene)
        res = next(r for r in M if r["scene"] == scene)
        lift = json.loads((RAW / "lift" / scene / "lift_occamfull.json").read_text())["lift"]
        lines.append(" & ".join([scene, f'{res["W"]}$\\times${res["H"]}', str(len(s)), f"{st.median(s):.0f}",
                                 f"{s[9 * len(s) // 10]:.0f}", f"{sum(s) / 3600:.1f}", f'{lift["seconds"]:.0f}',
                                 f'{100 * TL[scene]["seen"] / TL[scene]["rows"]:.1f}']) + r" \\")
    write("timing.tex", "\n".join(lines) + "\n")

    # plot data: per-image seconds histogram, masks per level boxes
    bins = list(range(30, 121, 5))
    def hist(vals):
        h = [0] * (len(bins) - 1)
        for v in vals:
            for i in range(len(bins) - 1):
                if bins[i] <= v < bins[i + 1]:
                    h[i] += 1
        return h
    hb = hist([float(r["seconds"]) for r in T if r["scene"] == "backroom"])
    hf = hist([float(r["seconds"]) for r in T if r["scene"] == "flightroom"])
    wide("timing_hist.csv", ["lo", "backroom", "flightroom"], [[bins[i] + 2.5, hb[i], hf[i]] for i in range(len(hb))])

    # peaks scatter: A peak vs Occam-full peak for positives (hit status of Occam full)
    rows = []
    for scene in RUNS:
        a = {r["text"]: r for r in E[(scene, "Ours-A")]["records"]}
        for r in E[(scene, "Occam-full")]["records"]:
            rows.append([a[r["text"]]["peak"], r["peak"], int(r["negative"]), int(bool(r.get("hit"))), scene])
    wide("peaks.csv", ["a", "occam", "neg", "hit", "scene"], rows)
    wide("peaks_pos.csv", ["a", "occam"], [[x[0], x[1]] for x in rows if not x[2]])
    wide("peaks_neg.csv", ["a", "occam"], [[x[0], x[1]] for x in rows if x[2]])

    # sweep from the shared floor_sweep.csv (Occam full, flat, A, Old)
    F = list(csv.DictReader(open(CSV / "floor_sweep.csv")))
    tabs = ["Old", "A", "Occam-full", "Occam-full-flat"]
    floors = sorted({float(r["floor"]) for r in F})
    def fv(t, fl, k):
        return next(int(r[k]) for r in F if r["table"] == t and float(r["floor"]) == fl)
    wide("sweep.csv", ["floor"] + [t.replace("-", "") + "_hits" for t in tabs] + [t.replace("-", "") + "_fp" for t in tabs],
         [[fl] + [fv(t, fl, "hits") for t in tabs] + [fv(t, fl, "neg_fp") for t in tabs] for fl in floors])
    print("→", OUT)


if __name__ == "__main__":
    main()
