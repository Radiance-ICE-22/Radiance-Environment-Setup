# Handover — state of the project on 12 Oct 2026

For the next session (Claude Code on intellisense08, or a human). Read this, then `CLAUDE.md` (rules), then only the
sections of the docs you need. Everything below was true at commit `02b4654` (main = origin/main, working tree clean).

## 1. Where things stand

| Area | State | Where the detail is |
| --- | --- | --- |
| FiGS install + `figs_pipeline.py` | Working on intellisense08 (`~/Radiance/figs`) | `docs/FiGS_pipeline_script_guide.md`, `LAB_MACHINE.md` |
| Galley console | Running (tmux `galley`, port 8800, idle). All features through the splat editor and the Phase 5 evaluation tile | `docs/GALLEY_UI.md` |
| SV-Net | Pipeline runs through Galley; **the student (Maverick, cohort `p4_smoke`) does not fly the course**. Next: watch its video, then cohort `p4_beta` (`data_beta`) | `GALLEY_UI.md` §6 |
| Semantics P0–P4 | Done, all gates passed | `docs/SEMANTICS.md` §1–11 |
| Semantics P5 (evaluation) | Done, plus the teacher study (A, B/B2, C/C2) and a floor sweep. Only open item: the browser check of View ▸ Compare | `SEMANTICS.md` §12–16 |
| Semantics P6 (instruction → course → SV-Net) | Not started | `docs/SEMANTICS_PLAN.md` |

**Current default** (`--backend lift`): multi-scale CLIP teachers, variant A (`--clip-mode scales`). FMGS keeps the
scale-averaged pyramid.

**The key finding from the last session.** A beats Old at the default floor of 0.55 (22/29 vs 20/29), but that lead is a
floor effect. A takes the maximum over three scale groups, which raises every peak by about 0.045, including the peaks of
objects that aren't in the scene.

At an equal number of false detections:

| Teachers | Hits /29 | Absent objects firing (of 6) | Floor |
| --- | --- | --- | --- |
| B2 flat (B2 without level choice) | 24 | 2 | 0.46–0.48 |
| Old | 23 | 2 | 0.51 |
| B2 | 23 | 2 | 0.51 |
| A | 22 | 2 | 0.55 |

These floors were read off the evaluation set itself, so the numbers are optimistic. No difference between designs is
statistically significant on 29 queries. FMGS's "no candidate" misses come from high-relevancy Gaussians scattered over
the room, not from the floor.

All of this is in `docs/report/semantic_embedding_report.pdf` §9–10 and `SEMANTICS.md` §16.

## 2. Decisions waiting for Suhan (don't decide alone)

1. **Default lift.** Keep A, or switch to Old/B2-flat with a floor calibrated per table (no evaluation queries)? The
   report recommends calibrating first, then deciding. A costs about 5× the lift time and 2.7× the table size, and its
   queries take 2.0–2.7 s.
2. **Next semantics work.** The report's ranked proposals (§10, table "Proposals in suggested order"):
   1. Calibrate the floor per table and per scoring rule; verify the absent-object set and enlarge it.
   2. B2 flat as a lift option, fused with Old/A at query time.
   3. A ViT-L/14 crop re-ranker on the top-5 candidates.
   4. FMGS: bake only supervised Gaussians, plus smoothing at query time.
   5. Prompt ensembles, and size/height priors.

   The full A×B×C teacher combinations were judged not worth running: the ceiling is 24/29.
3. **Merging B2 / C2.** Both live only on branches (`sem/variant-b-sam` 83e0cc0, `sem/variant-c-clipl` 7ac2cb0, both
   pushed). Neither is merged.
4. **Disk cleanup.** B2's teacher maps take 37 GB (`teachers/d869f497f5` on both scenes). The suffixed variant tables
   and their `*_train` directories take a further ~100 GB. Ask before deleting anything; keep the SAM mask cache
   (`sam_masks_vitb_p32`, 32 MB). Disk: 323 GB free of 1.6 TB.
5. **Ground truth.** "ceiling fan" (backroom) and "recycling bin" (flightroom) fire for every design at every floor.
   They are probably look-alikes that really are in the rooms, so the absent-object set needs checking.
6. **Phase 6 choices still open:** a rule-based vs LLM instruction parser; whether sparse-point clearance is good
   enough for the collision metric.
7. **Own scene.** The intellisense lab footage failed alignment (no ArUco marker visible), so it needs a re-shoot
   with the marker. The GTN_lab_v1 video is kept in `~/Radiance/video_captures/`, but its reconstruction was folded.

## 3. What is on the host

**Repo and worktrees**
- `~/Radiance/Radiance-Environment-Setup` is `main`.
- `~/Radiance/rese-b` is a worktree of `sem/variant-b-sam`; `~/Radiance/rese-c` is a worktree of `sem/variant-c-clipl`.
  Remove them with `git worktree remove` once the branches are settled.

**Tables per scene** in `SousVide/gsplats/workspace/<scene>/semantics/<run>/`
(backroom `2026-09-26_190524`, flightroom `2026-09-29_163855`):

| Table | Contents |
| --- | --- |
| `lift` | official, A |
| `lift_pyr` | Old |
| `lift_ms` | A |
| `lift_sam2` | B2 |
| `lift_clipl` | C, with the C2 floor in `index.json` |
| `lift_sam` | B |
| `lift_w480`, `lift_w720` | feature-width sweep |
| `fmgs`, `fmgs_c` | Old FMGS |
| `fmgs*_{ms,sam,sam2,clipl}` | FMGS per variant |

**Teacher caches** in `<scene>/semantics/teachers/<tag>`:

| Tag | Teachers |
| --- | --- |
| `d7a30d7264` | Old |
| `9c437cce68` | A |
| `96588ef160` | B |
| `d869f497f5` | B2 |
| `f38313b31f` | C (ViT-L/14) |

**Evaluations** in `<run>/`:
- `eval/` holds the current per-variant JSONs (`L-C@ms`, `@sam2`, `@clipl`, `L-Cflat@sam2`, …).
- `eval_amended_pyramid_2026-10-07/` is Old, `eval_frozen_2026-10-06/` the original frozen set, and
  `eval_clipl_uncalibrated/` is C.

**Logs and runner scripts in `~/Radiance`**
- `variant_{a,b,c,b2}.log`, `eval_c2.log`, `floor_sweep.{log,json}`, `run_variant*.sh`.
- `run_variant.sh` was restored after being overwritten while it ran: never edit a script while it is running.
- These scripts are scratch: they are not in the repo.

**Other**
- Node 22 is at `~/Radiance/tools/node` (`export PATH=~/Radiance/tools/node/bin:$PATH`) for frontend builds.
- The frozen kitchen constraints are in `~/Radiance/kitchen_constraints_2026-10-06.txt`.

## 4. Documents (all in the repo)

| File | What |
| --- | --- |
| `docs/SEMANTICS.md` | Authoritative semantics status and every measurement (§16 is the latest) |
| `docs/PROJECT_CHRONICLE.md` → `docs/report/project_chronicle.pdf` | The whole project in order, with every problem and fix. Rebuild with `docs/report/chronicle/build.sh` |
| `docs/report/semantic_embedding_report.pdf` | Full semantic pipeline and the Old / A / B2 / C2 comparison, with analysis and proposals |
| `docs/report/semantics_primer.pdf` | Prerequisites: splats, CLIP, DINOv2, SAM, the lift, FMGS |
| `docs/report/teacher_variants.pdf`, `semantics_pipeline.pdf`, `backend_implementation.pdf` | Earlier reports (6–8 Oct) |
| `docs/phase5_results/teacher_variants_final_2026-10-08/` | `variants.csv`, `per_query.csv`, `floor_sweep.json` |
| `docs/phase5_queries/` | Frozen query sets (f38225b); amended backroom set (7 Oct) |

## 5. Useful commands

```bash
source ~/Radiance/figs/figs_env.sh && cd ~/Radiance/Radiance-Environment-Setup
nvidia-smi; curl -s localhost:8800/api/health                          # before any GPU work
(cd semantics && CUDA_VISIBLE_DEVICES= python -m pytest -q tests)      # semantics tests (CPU)
(cd ui/backend && .venv/bin/python -m pytest -q)                       # Galley tests
python figs/semantic_query.py --scene backroom --backend lift "red tool chest"
python -m radiance_semantics.evaluate --scene flightroom --variants L-C L-C@sam2 L-Cflat@sam2   # from semantics/
CUDA_VISIBLE_DEVICES= python semantics/scripts/floor_sweep.py --out /tmp/fs.json \
    --tables Old=lift_pyr A=lift_ms B2=lift_sam2 B2-flat=lift_sam2:flat          # CPU, a few minutes
python figs/semantic_pipeline.py --scene backroom --backend lift --table-suffix _x --clip-mode sam2   # GPU, tmux
```

## 6. Working rules that bit us (see CLAUDE.md for all)

- **Pushing:** ask Suhan before every push. Commit with `git -c user.name="Platinum-Saber"
  -c user.email="sansikasuhan5@gmail.com"`, because the shared account has no git identity.
- **GPU:** one job at a time. Long runs go in tmux with a log file. Don't stop Galley while a job runs.
- **Kitchen env:** never install anything into kitchen without `--no-deps`, the constraints file and a torch re-check;
  ask first.
- **Shared account:** stay inside `~/Radiance` and `~/.local/share/galley`.
- **Evaluation:** settings must not be tuned on the evaluation sets. Report any post-hoc sweep as post-hoc.
