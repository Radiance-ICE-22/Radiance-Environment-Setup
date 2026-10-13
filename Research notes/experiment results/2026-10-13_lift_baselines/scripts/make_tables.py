"""Typeset the report's tables from ../csv into ../report/tables/*.tex (run after collect.py).

    python3 make_tables.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV, OUT = ROOT / "csv", ROOT / "report" / "tables"
METHOD_ORDER = [("Ours", "Old"), ("LUDVIG", "Old"), ("Occam", "Old"), ("Ours", "A"), ("LUDVIG", "A"), ("Occam", "A")]
NAME = {"Ours": "Ours (gsplat grad.)", "LUDVIG": "LUDVIG (original)", "Occam": "Occam's LGS (original)"}


def rows(name):
    with open(CSV / name) as f:
        return list(csv.DictReader(f))


def tex_escape(s):
    return s.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body)


def summary_table():
    S = rows("summary.csv")
    lines = []
    for mode in ("L-C", "L-CD"):
        lines.append(r"\midrule" + "\n" + rf"\multicolumn{{11}}{{l}}{{\textbf{{{mode}}}"
                     + (r" (CLIP relevancy only)" if mode == "L-C" else r" (+ DINO diffusion and split at query time)")
                     + r"}\\")
        for m, t in METHOD_ORDER:
            cells = [NAME[m], t]
            tot_h = tot_n = 0
            for scene in ("backroom", "flightroom"):
                r = next(x for x in S if x["scene"] == scene and x["method"] == m and x["teachers"] == t and x["mode"] == mode)
                tot_h += int(r["hits"]); tot_n += int(r["n_pos"])
                cells += [f'{r["hits"]}/{r["n_pos"]}', f'{float(r["err_med_m"]):.2f}', f'{r["neg_fp"]}/{r["n_neg"]}',
                          f'{float(r["auroc"]):.2f}']
            cells.insert(2, rf"\textbf{{{tot_h}}}/{tot_n}")
            lines.append(" & ".join(cells) + r" \\")
    write("summary.tex", "\n".join(lines) + "\n")


def per_query_table():
    P = rows("per_query.csv")
    cols = [(m, t, "L-C") for m, t in METHOD_ORDER]
    lines = []
    scene = None
    for r in P:
        if r["scene"] != scene:
            scene = r["scene"]
            lines.append(r"\midrule" + "\n" + rf"\multicolumn{{7}}{{l}}{{\textit{{{scene}}}}}\\")
        marks = []
        vals = [int(r[f"{m}-{t}-{mode}"]) for m, t, mode in cols]
        for v in vals:
            marks.append(r"\hit" if v else r"\miss")
        name = tex_escape(r["query"])
        if len(set(vals)) > 1:
            name = rf"\textbf{{{name}}}"
        lines.append(f"{name} & " + " & ".join(marks) + r" \\")
    tot = [sum(int(r[f"{m}-{t}-{mode}"]) for r in P) for m, t, mode in cols]
    lines.append(r"\midrule" + "\n" + "Total (of 29) & " + " & ".join(str(x) for x in tot) + r" \\")
    write("per_query.tex", "\n".join(lines) + "\n")


def cost_table():
    C = rows("lift_cost.csv")
    lines = []
    for m, t in METHOD_ORDER:
        cells = [NAME[m], t]
        for scene in ("backroom", "flightroom"):
            r = next(x for x in C if x["scene"] == scene and x["method"] == m and x["teachers"] == t)
            cells += [f'{float(r["lift_s"]):.0f}', f'{int(r["peak_vram_mib"]) / 1024:.1f}',
                      f'{100 * float(r["seen_frac"]):.1f}']
        lines.append(" & ".join(cells) + r" \\")
    write("cost.tex", "\n".join(lines) + "\n")


def agreement_table():
    A = rows("table_agreement.csv")
    label = {"lift_pyr__lift_ludvigpyr": "Ours vs LUDVIG (Old)", "lift_pyr__lift_occampyr": "Ours vs Occam (Old)",
             "lift_ms__lift_ludvigms": "Ours vs LUDVIG (A)", "lift_ms__lift_occamms": "Ours vs Occam (A)"}
    lines = []
    for pair, lab in label.items():
        for feat in ("clip", "dino"):
            cells = [lab if feat == "clip" else "", feat.upper()]
            for scene in ("backroom", "flightroom"):
                r = next(x for x in A if x["scene"] == scene and x["pair"] == pair and x["feature"] == feat)
                cells += [f'{float(r["q0.5"]):.4f}', f'{float(r["q0.01"]):.3f}', f'{100 * float(r["frac_gt_0.99"]):.1f}']
            lines.append(" & ".join(cells) + r" \\")
    write("agreement.tex", "\n".join(lines) + "\n")


def dev_table():
    p = CSV / "dev.csv"
    if not p.exists():
        return
    D = rows("dev.csv")
    label = {"L-C@pyr": "Ours, Old", "L-C@ludvigpyr": "LUDVIG, Old", "L-C@occampyr": "Occam's, Old",
             "L-C": "Ours, A", "L-C@ludvigms": "LUDVIG, A", "L-C@occamms": "Occam's, A"}
    D = sorted(D, key=lambda r: list(label).index(r["variant"]))
    lines = [" & ".join([label[r["variant"]], r["hits"], f'{float(r["err_med_m"]):.2f}', r["peaks"]]) + r" \\"
             for r in D]
    write("dev.tex", "\n".join(lines) + "\n")


def write_wide(name, header, data):
    d = ROOT / "report" / "data"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)


def plot_data():
    """Wide CSVs for pgfplots (one row per x value)."""
    Cb = rows("combined.csv")
    def tot(m, t, mode):
        return next(int(r["hits"]) for r in Cb if r["method"] == m and r["teachers"] == t and r["mode"] == mode)
    write_wide("hits_bars.csv", ["method", "OldLC", "ALC", "OldLCD", "ALCD"],
               [[m, tot(m, "Old", "L-C"), tot(m, "A", "L-C"), tot(m, "Old", "L-CD"), tot(m, "A", "L-CD")]
                for m in ("Ours", "LUDVIG", "Occam")])
    S = rows("summary.csv")
    def sc_hits(scene, m, t):
        return next(int(r["hits"]) for r in S if r["scene"] == scene and r["method"] == m and r["teachers"] == t
                    and r["mode"] == "L-C")
    write_wide("hits_scene.csv", ["config", "backroom", "flightroom"],
               [[f"{m}-{t}", sc_hits("backroom", m, t), sc_hits("flightroom", m, t)] for m, t in METHOD_ORDER])
    C = rows("lift_cost.csv")
    def c(scene, m, t, k):
        return next(r[k] for r in C if r["scene"] == scene and r["method"] == m and r["teachers"] == t)
    write_wide("cost.csv", ["config", "backroom_s", "flightroom_s", "backroom_seen", "flightroom_seen"],
               [[f"{m}-{t}", c("backroom", m, t, "lift_s"), c("flightroom", m, t, "lift_s"),
                 float(c("backroom", m, t, "seen_frac")) * 100, float(c("flightroom", m, t, "seen_frac")) * 100]
                for m, t in METHOD_ORDER])
    H = rows("table_agreement_hist.csv")
    keys = [("backroom", "lift_pyr__lift_occampyr", "clip"), ("backroom", "lift_pyr__lift_occampyr", "dino"),
            ("flightroom", "lift_pyr__lift_occampyr", "clip"), ("flightroom", "lift_pyr__lift_occampyr", "dino")]
    bins = sorted({float(r["bin_lo"]) for r in H})
    def frac(k, b):
        return next(float(r["frac"]) for r in H if (r["scene"], r["pair"], r["feature"]) == k and float(r["bin_lo"]) == b)
    write_wide("agreement_hist.csv", ["bin", "br_clip", "br_dino", "fr_clip", "fr_dino"],
               [[b + 0.0125] + [max(frac(k, b), 1e-6) for k in keys] for b in bins])
    p = CSV / "floor_sweep.csv"
    if p.exists():
        F = rows("floor_sweep.csv")
        tables = ["Old", "A", "LUDVIG-Old", "LUDVIG-A", "Occam-Old", "Occam-A"]
        floors = sorted({float(r["floor"]) for r in F})
        def fv(t, fl, k):
            return next(int(r[k]) for r in F if r["table"] == t and float(r["floor"]) == fl)
        hdr = ["floor"] + [f"{t}_hits" for t in tables] + [f"{t}_fp" for t in tables]
        write_wide("floor_sweep.csv", [h.replace("-", "") for h in hdr],
                   [[fl] + [fv(t, fl, "hits") for t in tables] + [fv(t, fl, "neg_fp") for t in tables] for fl in floors])


if __name__ == "__main__":
    plot_data()
    summary_table()
    per_query_table()
    cost_table()
    agreement_table()
    dev_table()
    print("tables →", OUT)
