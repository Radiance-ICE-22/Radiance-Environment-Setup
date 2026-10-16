# Region-level teachers in our pipeline: `--clip-mode langsplat`, skip vs fill (16 Oct 2026)

Branch `semantic/new`. **Report:** [report/langsplat_teacher_report.pdf](report/langsplat_teacher_report.pdf)

| Configuration (Phase 5, 29 objects / 6 absent) | Hits @0.55 | Absent firing | Best hits at 1 FP |
| --- | --- | --- | --- |
| Ours A (default) | 22 | 2 | 19 |
| Ours B2 (SAM ViT-B regions, fill) | 21 | 2 | — |
| Occam's full chain (LangSplat features, centre-pixel lift) | 25 | 1 | 25 |
| **Ours + LangSplat teachers, skip** | **24** | **1** | 24 (25 flat) |
| Ours + LangSplat teachers, fill | 23 | 1 | 24 |

- LangSplat's region features are now a teacher mode of our pipeline (cache tag `4f4bdb1a08`), lifted with our all-pixel
  lift: 24/29, AUROC 0.92 / 1.00, dev 5/5; ±1 from Occam's chain (water jug cluster offset).
- Skip vs fill: one query (camera tripod), skip better; so B2's 21 comes from its other choices (ViT-B, no crop layer,
  stricter SAM thresholds, area levels, no LangSplat NMS).
- Build: `semantic_pipeline.py --scene <s> --backend lift --clip-mode langsplat [--uncovered skip|fill] [--table-suffix _ls]`.

Contents: `raw/` (evaluator JSONs, dev set, lift stats with per-level coverage, floor sweep, log), `csv/`,
`scripts/collect.py` (→ csv + report tables).
