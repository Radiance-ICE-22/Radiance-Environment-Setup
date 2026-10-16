"""Typeset the report's tables and plot data from ../csv (run after collect.py).

    python3 make_tables.py        → ../report/tables/*.tex, ../report/data/*.csv
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV, TAB, DATA = ROOT / "csv", ROOT / "report" / "tables", ROOT / "report" / "data"
NAME = {"Ours-Old": "Ours, Old teachers", "Ours-A": "Ours, A teachers (default)", "B2": "Ours, B2 (SAM ViT-B regions)",
        "LUDVIG-ctrl": "LUDVIG lift, Old teachers", "LUDVIG-full": "LUDVIG full pipeline",
        "Occam-ctrl": "Occam's lift, A teachers", "Occam-full": "Occam's full pipeline"}
ORDER = ["Ours-Old", "Ours-A", "B2", "LUDVIG-ctrl", "LUDVIG-full", "Occam-ctrl", "Occam-full"]


def rows(name):
    with open(CSV / name) as f:
        return list(csv.DictReader(f))


def esc(s):
    return s.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def write(d, name, text):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text)


def wide(name, header, data):
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)


def main():
    S = rows("summary.csv")
    lines = []
    for cfg in ORDER:
        for mode in ("L-C", "L-CD", "L-CL"):
            rs = {r["scene"]: r for r in S if r["config"] == cfg and r["mode"] == mode}
            if not rs:
                continue
            b, f = rs["backroom"], rs["flightroom"]
            tot = int(b["hits"]) + int(f["hits"])
            fp = int(b["neg_fp"]) + int(f["neg_fp"])
            bold = cfg == "Occam-full"
            cells = [NAME[cfg] if mode == "L-C" else "", mode,
                     (rf"\textbf{{{tot}}}" if bold else str(tot)) + "/29", f"{fp}/6",
                     f'{b["hits"]}/13', f'{float(b["err_med_m"]):.2f}', f'{float(b["auroc"]):.2f}',
                     f'{f["hits"]}/16', f'{float(f["err_med_m"]):.2f}', f'{float(f["auroc"]):.2f}']
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[2pt]")
    write(TAB, "summary.tex", "\n".join(lines[:-1]) + "\n")

    P = rows("per_query.csv")
    cols = [("Ours-A", "L-C"), ("B2", "L-C"), ("LUDVIG-full", "L-C"), ("LUDVIG-full", "L-CL"), ("Occam-ctrl", "L-C"),
            ("Occam-full", "L-C")]
    lines, scene = [], None
    for r in P:
        if r["scene"] != scene:
            scene = r["scene"]
            lines.append(r"\midrule" + "\n" + rf"\multicolumn{{{len(cols) + 1}}}{{l}}{{\textit{{{scene}}}}}\\")
        v = [int(r[f"{c}-{m}"]) for c, m in cols]
        name = esc(r["query"])
        if len(set(v)) > 1:
            name = rf"\textbf{{{name}}}"
        lines.append(name + " & " + " & ".join(r"\hit" if x else r"\miss" for x in v) + r" \\")
    tot = [sum(int(r[f"{c}-{m}"]) for r in P) for c, m in cols]
    lines.append(r"\midrule" + "\nTotal (of 29) & " + " & ".join(map(str, tot)) + r" \\")
    write(TAB, "per_query.tex", "\n".join(lines) + "\n")

    C = rows("cost.csv")
    lines = []
    for cfg in ["Ours-Old", "Ours-A", "B2", "LUDVIG-full", "Occam-full"]:
        cells = [NAME[cfg]]
        for scene in ("backroom", "flightroom"):
            r = next(x for x in C if x["scene"] == scene and x["config"] == cfg)
            feat = r["features_s"]
            if cfg == "LUDVIG-full":
                cells += [f'{float(feat) / 60:.0f} (incl.)', f'{float(r["diffusion_s"]) / 60:.1f}']
            elif cfg == "Occam-full":
                cells += [f'{float(feat) / 3600:.1f} h + {float(r["lift_s"]):.0f} s', "--"]
            else:
                cells += [f'{float(r["lift_s"]) / 60:.0f}', "--"]
            cells += [f'{100 * float(r["seen_frac"]):.1f}', f'{float(r["table_mb"]) / 1024:.1f}']
        lines.append(" & ".join(cells) + r" \\")
    write(TAB, "cost.tex", "\n".join(lines) + "\n")

    D = rows("dev.csv")
    lab = {"L-C@ludvigfull": "LUDVIG full, L-C", "L-CD@ludvigfull": "LUDVIG full, L-CD",
           "L-CL@ludvigfull": "LUDVIG full, L-CL (diffusion)", "L-C@occamfull": "Occam's full, L-C"}
    write(TAB, "dev.tex", "\n".join(" & ".join([lab.get(r["variant"], esc(r["variant"])), r["hits"],
                                                  f'{float(r["err_med_m"]):.2f}', esc(r["missed"])]) + r" \\"
                                        for r in D) + "\n")

    Cb = rows("combined.csv")
    wide("bars.csv", ["config", "LC", "LCD", "LCL"],
         [[c, *[next((r["hits"] for r in Cb if r["config"] == c and r["mode"] == m), "nan") for m in ("L-C", "L-CD", "L-CL")]]
          for c in ORDER])

    F = rows("floor_sweep.csv")
    tables = ["Old", "A", "LUDVIG-full", "Occam-full", "Occam-full-flat"]
    floors = sorted({float(r["floor"]) for r in F})
    def fv(t, fl, k):
        return next(int(r[k]) for r in F if r["table"] == t and float(r["floor"]) == fl)
    wide("floor_sweep.csv", ["floor"] + [t.replace("-", "") + "_hits" for t in tables] + [t.replace("-", "") + "_fp" for t in tables],
         [[fl] + [fv(t, fl, "hits") for t in tables] + [fv(t, fl, "neg_fp") for t in tables] for fl in floors])
    env = []
    for t in tables:
        best = {}
        for fl in floors:
            fp, h = fv(t, fl, "neg_fp"), fv(t, fl, "hits")
            for k in range(fp, 7):
                best[k] = max(best.get(k, 0), h)
        env.append((t, [best.get(k) for k in range(1, 6)]))
    lab = {"Old": "Ours, Old", "A": "Ours, A", "LUDVIG-full": "LUDVIG full", "Occam-full": "Occam's full",
           "Occam-full-flat": "Occam's full, no level choice"}
    write(TAB, "envelope.tex", "\n".join(" & ".join([lab[t]] + [("--" if v is None else (rf"\textbf{{{v}}}" if v == max(
        e[1][i] or 0 for e in env) else str(v))) for i, v in enumerate(vals)]) + r" \\" for t, vals in env) + "\n")
    print("tables →", TAB, "| data →", DATA)


if __name__ == "__main__":
    main()
