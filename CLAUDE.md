# CLAUDE.md — Radiance FYP (Claude Code running on the GPU hosts)

Final-year project of Suhan (team Radiance, CSE, University of Moratuwa): **language-guided goals for
simulation-trained visuomotor drone policies**. This repo
(github.com/Radiance-ICE-22/Radiance-Environment-Setup) holds the install automation for Stanford MSL's
SOUS-VIDE / FiGS (3D Gaussian Splat simulator + SV-Net visuomotor policy), our pipeline scripts, **Galley**
(FastAPI + React web console in `ui/`) and **radiance_semantics** (per-Gaussian CLIP + DINOv2 features on a
frozen splat, so a phrase resolves offline to a 3D goal that becomes waypoints).

Until 5 Oct 2026 the work was done by Claude in a cloud sandbox with no route to the hosts: it wrote code,
Suhan committed, pulled on the host, ran a gate script and pasted the log back. **You are now running on the
host itself** — run the tests, the probes and the gates yourself, read the logs yourself, and iterate.

## Read these first (they are the source of truth, keep them current)

| File | What it holds |
| --- | --- |
| `docs/SEMANTICS.md` | Semantics status (§1), what each phase added (§2–3e), decisions, every measurement and gate result (§4–10). **Authoritative status.** |
| `docs/SEMANTICS_PLAN.md` | The phase plan P0–P6 with tasks, tests and gates. Live copy with tick-boxes: https://claude.ai/code/artifact/d12ee1d2-3737-424f-bca9-6827f3e195aa |
| `docs/GALLEY_UI.md` | Galley status (§1), running it (§2), architecture (§3), next steps (§6), **traps (§7)**. |
| `docs/FiGS_pipeline_script_guide.md` | `figs/figs_pipeline.py` (capture → splat → course → expert flight). |
| `ui/README.md` | Galley's API. |
| `docs/LAB_MACHINE.md` | intellisense05 (campus-only, older paths). Not the current host. |

Read only the sections you need; they are long.

## Where we left off (6 Oct 2026)

| Semantics phase | State |
| --- | --- |
| P0 env + poses | DONE (refined poses +4.22 dB PSNR over raw → lift and FMGS use refined poses) |
| P1 teachers + lift + CLI query | DONE, 5/5 on backroom |
| P2 Galley backend + CPU query worker | DONE (warm query ~0.74 s, cold 5.8 s) |
| P3 splat editor `#/splat/<scene>` | DONE (query → Send to course → flight) |
| P4 FMGS backend | DONE — gate PASSED 6 Oct 13:31 (run 3, 7344e7a) |
| **P5 evaluation (L-C, L-CD, F-C, F-CD × backroom, flightroom)** | **Results in — `docs/SEMANTICS.md` §13** (lift 0.62/0.63 top-1 best; F-CD CLIP fragmented). Open: sweeps, Galley metrics tile, Compare check |
| P6 instruction → semantic-course → SV-Net cohorts | Not started (experiment waits on SV-Net flying) |

**Phase 4 gate, run 3** (6 Oct, intellisense08, `7344e7a`) — PASSED; details in `docs/SEMANTICS.md` §3e and §11.
Default config, no fallback: 4,200 steps in 31.4 min, peak 5.5 GB PyTorch / 7.95 GB device, Gaussians and
checkpoint unchanged, dev queries FMGS 4/5 (garden cart misses) vs lift 5/5. What it took:
- OOM: PyTorch's cache starved tiny-cuda-nn's own allocator, and the ladder missed tcnn's `RuntimeError`.
  → expandable segments, tcnn OOM recognised, a failed step retried once, shorter tensor lifetimes.
- Then NaN at step 11: CLIP gradients underflowed in tcnn's fp16, and pixel alignment's normalize of ~0
  vectors overflowed. → GradScaler when a part is tcnn, 1e-3 norm floor in pixel alignment.
- pytest 9.1.1 (+ iniconfig, pluggy) added to kitchen on 6 Oct with `--no-deps` and the constraints file
  (frozen copy: `~/Radiance/kitchen_constraints_2026-10-06.txt`); torch re-checked. Gates now run the 68 semantics tests.
- Still to do by hand: View ▸ Compare in the browser on the new FMGS table.
- Planned by Suhan (6 Oct): a separate branch that extends nerfstudio to handle semantics (the
  `ns-train` route the plan first had). `main` keeps the standalone trainer until that branch is ready.

Next: finish Phase 5 (sweeps, Galley metrics tile; table already in `docs/phase5_results/`). GTN_lab_v1 is
folded (SEMANTICS.md §12); the intellisense lab footage could become our own extra scene. Phase 6 parts that don't need SV-Net (parser, `semantic-course`,
feasibility loop). Separate open thread: the SV-Net student (Maverick, cohort `p4_smoke`) does not fly the
course yet; next is cohort `p4_beta` (data_beta) — see `docs/GALLEY_UI.md` §6.

Open questions for Suhan (don't decide these alone): which 15+ objects per scene for the frozen Phase 5 set;
whether to add a larger CLIP (ViT-L/14) to the comparison if 8 GB allows; rule-based vs LLM instruction
parser in Phase 6; whether sparse-point clearance is good enough for the collision metric.

## Machines

| Host | GPU | Repo checkout | FiGS install (`PROJECT_ROOT`) | Role |
| --- | --- | --- | --- | --- |
| **intellisense08** (lab, Ubuntu 22.04, user `yutharsan`, Tailscale) | RTX 2080 8 GB, sm_75, driver 535 | `~/Radiance/Radiance-Environment-Setup` | `~/Radiance/figs` | Main host: semantics gates, SV-Net |
| dummy (home, Ubuntu 24.04, user `hanzo`, Tailscale) | RTX 3050 Ti Laptop 4 GB | `~/FYP-Radiance` | `~/projects/figs_validation` | Galley build/test; too small for commNet |
| RTX 5060 Ti PC (home) | 16 GB, Blackwell | — | — | Future main host; needs CUDA 12.8 / torch ≥ 2.7 (Galley Phase 6 installer, cu128 profile) |
| Suhan's laptop (Windows) | — | `D:\Projects\FYP\FYP-Radiance` (CRLF) | — | Where Suhan reads and commits |

`ui/deploy/host.sh` resolves `FIGS_ROOT`, `RADIANCE_REPO` and `GALLEY_MACHINE` from the hostname;
profiles are `ui/machines/<host>.toml`.

On intellisense08: scene **backroom**, active run `2026-09-26_190524` (532,361 Gaussians, 270 training
cameras); SousVide clone at `~/Radiance/figs/SousVide`; splats in `SousVide/gsplats/workspace/<scene>/`;
semantic artefacts in `gsplats/workspace/<scene>/semantics/`; run records in `SousVide/runs/`; Galley data in
`~/.local/share/galley/` (log `galley.log`, worker log `semantic_worker.log`).

## Rules for working on the host

**Environments**
- Anything FiGS / SousVide / semantics: `source ~/Radiance/figs/figs_env.sh` first (activates conda env
  `kitchen`, sets `ACADOS_SOURCE_DIR`, `LD_LIBRARY_PATH`, `PYTHONNOUSERSITE=1`). Never just `conda activate`.
- Galley has its own uv venv in `ui/backend/.venv`. **Never** install Galley deps into kitchen or kitchen deps
  into Galley's venv.
- The kitchen stack is pinned and compiled against it: torch 2.1.2 (CUDA 11.8), nerfstudio 1.1.4, gsplat
  1.0.0, tinycudann 2.0 (built for `TCNN_CUDA_ARCHITECTURES=75`), numpy 1.x, timm 0.6.7, open_clip 2.24.0,
  huggingface_hub < 1. **Never `pip install` into kitchen without `--no-deps` and the constraints file**
  (`radiance_semantics.env_check`), and re-check torch afterwards. Ask Suhan before adding anything to
  kitchen and before any `sudo`.

**GPU**
- One GPU job at a time. Before any GPU work: `nvidia-smi` and
  `curl -s localhost:8800/api/health` (a `"current_job": <n>` means Galley is running a job — wait).
- The semantic query worker is CPU-only and never takes the GPU.

**Long runs**
- Anything longer than a few minutes runs in **tmux** with output to a log file, then poll the log with
  `tail`/`grep`. Don't hold a foreground command for tens of minutes. The laptop's SSH link goes through
  Tailscale DERP and can drop; tmux keeps the run alive.
- Don't stop or restart Galley while a job runs (`./run_ui.sh stop` refuses; don't pass `--force`): the job's
  log is lost and it's marked interrupted.

**Git**
- `origin` is GitHub. Keep `pull.ff only`. Commit message style:
  `Semantic embedding: Phase 4-commit_4 <what changed>` (or `Galley: …`).
- Commit locally when a coherent step is done **and its tests pass**. **Ask Suhan before pushing.** Never
  force-push, rebase shared history or amend pushed commits.
- Gate scripts `git pull` by default: run them with `SKIP_PULL=1` while testing uncommitted changes.
- Never commit large artefacts: `gsplats/`, checkpoints, teacher caches, tables, `field.pt`, videos
  (see `.gitignore`). `ui/frontend/dist/` **is** committed — rebuild it (`npm ci && npm run build` in
  `ui/frontend`) whenever `ui/frontend/src/` changes. Check `node --version` first; if Node is missing, ask.

**Shared lab account**
- The `yutharsan` account on intellisense08 is a teammate's. Stay inside `~/Radiance` and
  `~/.local/share/galley`, don't touch other files, and don't delete trained runs, archived runs, captures or
  semantic artefacts without asking.

**Ask first before anything that is hard to undo**: retraining or promoting/archiving backroom (changes the
checkpoint key, so every semantic table and the `.splat` cache go stale), deleting runs or caches, changing the
torch stack, re-running teacher extraction (31 min) without need.

## Commands

```bash
# environment + where things are
source ~/Radiance/figs/figs_env.sh
cd ~/Radiance/Radiance-Environment-Setup && source ui/deploy/host.sh   # FIGS_ROOT, RADIANCE_REPO, GALLEY_MACHINE

# tests
(cd ui/backend && .venv/bin/python -m pytest -q)                       # Galley backend
(cd semantics && CUDA_VISIBLE_DEVICES= python -m pytest -q tests)      # 68 tests, CPU

# Phase 4 pieces
(cd semantics && python -m radiance_semantics.fmgs.diag)               # tcnn probes + the field impl auto picks
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python -m radiance_semantics.fmgs.train \
    --project-root ~/Radiance/figs --scene backroom --steps 200 --out /tmp/sem4_smoke   # smoke: exit 0 = loss fell, Gaussians unchanged
python figs/semantic_pipeline.py --scene backroom --backend fmgs --status
python figs/semantic_query.py --scene backroom --backend lift "red tool chest"
python figs/semantic_query.py --scene backroom --eval                  # 5 annotated dev queries

# gates (always in tmux, queue idle)
tmux new -d -s sem4 'SKIP_PULL=1 bash ~/Radiance/Radiance-Environment-Setup/ui/deploy/sem4_gate.sh > ~/sem4_gate.log 2>&1'
tail -f ~/sem4_gate.log                                                # or: tmux attach -t sem4

# Galley
./run_ui.sh status | logs | restart        # tmux session "galley", port 8800, loopback only
```

Suhan views Galley from the laptop with a tunnel (Windows reserves 8800, and VS Code may hold 18800):
`ssh -N -L 28800:localhost:8800 intellisense08` → http://localhost:28800.

## Semantics facts worth knowing before touching code

- **gsplat 1.0.0**: forward takes ≤ 512 channels, **backward only ≤ 32**. `render_features` chunks
  grad-requiring features into 32-channel calls (lift = 29 passes per image; FMGS = 28 chunks per step).
- **Refined poses**: splatfacto applies its SO3xR3 camera correction only while training; `cameras.optimized_c2w`
  applies it explicitly. Lift and FMGS must use it.
- **tcnn on sm_75**: any 192-dim hash grid fails to launch; `FieldConfig.split` builds the 24 levels as
  consecutive 12 × 8 grids (levels 16→84, 98→512, each within 1 % of FMGS's ladder). A reported deviation.
- **FMGS trainer** (`fmgs/train.py`): standalone (not an `ns-train` plugin), splat frozen, most opaque 40 %
  trainable, 480 × 270 features, loss 0.2·CLIP Huber + 0.8·DINO L2 + 0.01·pixel alignment, Adam 1e-2→1e-3,
  4,200 steps, checkpoints every 1,000, Gaussian checksum before/after must match.
- **Table rows = `.splat` records**, both in `course_tools.splat_order()` order; table key = run + checkpoint
  stem + mtime = Galley's `.splat` cache key (stale together).
- **Frames**: splat → course is (x, −y, −z). Annotations and goals are in the course frame.
- **Query**: LERF relevancy with canonical negatives; relative threshold τ = 0.55 + 0.5·(peak − 0.55);
  voxel connected components (0.1 m); approach point at a configurable standoff (default 1.0 m); clearance =
  k-th nearest sparse point minus body radius 0.19 m, min gap 0.15 m.
- **The five backroom queries are a development set** (they shaped the threshold). Phase 5 needs a separate
  frozen set annotated before any results are seen.
- Upstream traps (integer cells in course files, Viper re-timing, upstream's TTE not being a distance,
  `Pilot()` resuming from disk, commNet needing ~6.8 GB): `docs/GALLEY_UI.md` §7.

## Reporting back

Suhan reads progress in the repo docs, not the chat history. After each meaningful step: update
`docs/SEMANTICS.md` / `docs/GALLEY_UI.md` with what changed, the measured numbers (date, commit, host), and
what's next; keep this file's "Where we left off" section short and current.
