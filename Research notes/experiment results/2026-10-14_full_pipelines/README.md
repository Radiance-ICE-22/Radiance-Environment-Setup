# Full pipelines: LUDVIG (DINOv2 ViT-g + graph diffusion) and Occam's LGS (LangSplat SAM ViT-H features)

Runs 13–14 Oct 2026 on intellisense08 (RTX 2080). **Report:** [report/full_pipelines_report.pdf](report/full_pipelines_report.pdf)

## Result in brief (Phase 5 sets, 29 positives / 6 absent objects, floor 0.55)
| Configuration | Hits | Absent firing | Best hits at 1 false positive (floor sweep) | Features + lift per scene |
| --- | --- | --- | --- | --- |
| Ours, A teachers (default) | 22 | 2 | 19 | 18–23 min lift (+ 31–53 min teachers) |
| Ours, B2 (SAM ViT-B regions) | 21 | 2 | — | 15–17 min (+ 41–47 min teachers) |
| LUDVIG full (L-C / L-CL with its diffusion) | 19 / 18 | 1 | 19 | 29–37 min + 2–4 min diffusion |
| **Occam's full (LangSplat SAM ViT-H, 3 levels)** | **25** | **1** | **25** (floors 0.52–0.55) | 5.1 h / 4.7 h + 30 s |

- Occam's full pipeline is the best configuration measured: backroom 12/13, flightroom 13/16, AUROC 0.92 / 1.00; gains
  swivel chair, camera tripod, hose reel over A, loses nothing. Without per-query level choice: 23 at 1 FP.
- LUDVIG's full pipeline does not improve on our lift; its diffusion cannot recover objects below the floor.
- The 2D features decide the outcome, not the lift (cf. 2026-10-13_lift_baselines).

## Contents
- `raw/eval/<scene>/*.json` (Phase 5), `raw/eval_dev/backroom/*.json` (development set), `raw/lift/<scene>/*.json`
  (lift statistics, LangSplat meta), `raw/floor_sweep.json`, `raw/logs/` (runs; `test_langsplat_nms.log` = the
  vectorised-NMS equivalence check).
- `csv/` — summary, combined, per_query, cost, dev, floor_sweep.
- `scripts/collect.py`, `scripts/make_tables.py` — raw → csv → report tables.

Code: `semantics/baselines/` on `semantic/LUDVIG` (ludvig_full.py, ludvig_diffuse.py, import_diffused.py; query
variant L-CL) and `semantic/LGS` (langsplat_features.py, occam_lift.py --langsplat). LangSplat feature cache (15 GB):
`~/Radiance/third_party/langsplat_features/`.
