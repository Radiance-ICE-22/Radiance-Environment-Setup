# Lift baselines: original LUDVIG and Occam's LGS vs our lift (13 Oct 2026, intellisense08)

**Report:** [report/lift_baselines_report.pdf](report/lift_baselines_report.pdf)

## Result in brief
| Lift | Teachers | Hits L-C (of 29) | Hits L-CD | Lift time backroom / flightroom |
| --- | --- | --- | --- | --- |
| Ours (gsplat gradients) | Old / A | 20 / 22 | 20 / 22 | 181 s / 262 s · 1064 s / 1364 s |
| LUDVIG (original code) | Old / A | 20 / 22 | 20 / 22 | 369 s / 633 s · 784 s / 1352 s |
| Occam's LGS (original code) | Old / A | 21 / 22 | 19 / 20 | 28 s / 38 s · 124 s / 74 s |

- LUDVIG's lift **is** ours: per-Gaussian cosine 1.0000 on every row of both scenes, identical outcomes on every query.
- Occam's centre-pixel lift: CLIP cosine median 0.997, DINO 0.98; leaves 4.8 % / 7.3 % of Gaussians unseen; 6.5–18×
  faster than ours; ±1–2 queries either way, not significant. Floor sweep: same 24/29 ceiling for all.
- The lift is not the bottleneck; teachers and query are.

## Contents
- `raw/eval/<scene>/<variant>.json` — Phase 5 evaluator output (frozen sets, backroom amended 7 Oct). Variants:
  `L-C`/`L-CD` = our A table (`lift`), `@pyr` = our Old (`lift_pyr`), `@ludvigpyr`/`@ludvigms` = LUDVIG Old/A,
  `@occampyr`/`@occamms` = Occam's Old/A.
- `raw/eval_dev/backroom/*.json` — development set (`phase1_gate`, 5 queries).
- `raw/lift/<scene>/<table>.json` — lift time, peak VRAM, seen Gaussians, table size.
- `raw/floor_sweep.json` — floors 0.46–0.60, per scene × table (`semantics/scripts/floor_sweep.py`).
- `raw/table_agreement.json` — per-Gaussian cosine between tables (`scripts/table_agreement.py`).
- `raw/logs/` — install, lift, evaluation logs.
- `csv/` — `summary`, `combined`, `per_query`, `lift_cost`, `floor_sweep`, `table_agreement(_hist)`, `dev`.

## Regenerate
```bash
python3 scripts/collect.py        # host semantics dirs → raw/ + csv/
python3 scripts/make_tables.py    # csv/ → report/tables/*.tex, report/data/*.csv
cd report && pdflatex lift_baselines_report.tex && pdflatex lift_baselines_report.tex
```
The runs themselves: code on branches `semantic/LUDVIG` and `semantic/LGS` (`semantics/baselines/`), report §8.
Tables on the host: `SousVide/gsplats/workspace/<scene>/semantics/<run>/lift_{ludvig,occam}{pyr,ms}` (+ `_raw`).
