# Experiment results

One folder per experiment, named `<date>_<topic>/`, each with
- `raw/` — the untouched outputs of the runs (evaluator JSONs, lift statistics, sweeps, logs),
- `csv/` — tables derived from `raw/` by the experiment's `scripts/collect.py`,
- `scripts/` — what produced `raw/` → `csv/` → report tables,
- `report/` — the LaTeX report (`pdflatex` twice inside `report/`) and its PDF.

| Experiment | What | Report |
| --- | --- | --- |
| [2026-10-13_lift_baselines](2026-10-13_lift_baselines/) | The original LUDVIG and Occam's LGS code against our lift, controlled (same Gaussians, cameras, teachers, query, evaluator) | [lift_baselines_report.pdf](2026-10-13_lift_baselines/report/lift_baselines_report.pdf) |
