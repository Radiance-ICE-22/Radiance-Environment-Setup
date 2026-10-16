# Experiment results

One folder per experiment, named `<date>_<topic>/`, each with
- `raw/` — the untouched outputs of the runs (evaluator JSONs, lift statistics, sweeps, logs),
- `csv/` — tables derived from `raw/` by the experiment's `scripts/collect.py`,
- `scripts/` — what produced `raw/` → `csv/` → report tables,
- `report/` — the LaTeX report (`pdflatex` twice inside `report/`) and its PDF.

| Experiment | What | Report |
| --- | --- | --- |
| [2026-10-13_lift_baselines](2026-10-13_lift_baselines/) | The original LUDVIG and Occam's LGS code against our lift, controlled (same Gaussians, cameras, teachers, query, evaluator) | [lift_baselines_report.pdf](2026-10-13_lift_baselines/report/lift_baselines_report.pdf) |
| [2026-10-14_full_pipelines](2026-10-14_full_pipelines/) | The papers' whole pipelines: LUDVIG (DINOv2 ViT-g + graph diffusion), Occam's LGS on LangSplat SAM ViT-H features — Occam's full reaches 25/29 at 1/6 false positives | [full_pipelines_report.pdf](2026-10-14_full_pipelines/report/full_pipelines_report.pdf) |
| [2026-10-16_langsplat_teacher](2026-10-16_langsplat_teacher/) | LangSplat region features as our own teacher mode (`--clip-mode langsplat`) and the skip-vs-fill test: 24/29 at 1/6 FP; skip vs fill differ by one query | [langsplat_teacher_report.pdf](2026-10-16_langsplat_teacher/report/langsplat_teacher_report.pdf) |
