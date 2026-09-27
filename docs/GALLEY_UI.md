# Galley — web console for the pipeline: status and handoff

*Last updated 2026-09-27: Phase 3 (course editor) built; backend gate passed on dummy, the
browser half is pending. Written so the next working session (human or
Claude) can pick up without the chat history. The live plan, with diagrams, is the Claude
Doc "SOUS-VIDE Pipeline Console — Implementation Plan":
https://claude.ai/code/artifact/7f7d0bbb-6399-4f1b-bead-21ff96f3f12e*

Galley is a FastAPI + React web UI in `ui/` that drives this repo's pipeline scripts:
`figs/figs_pipeline.py` today, and `figs/svnet_pipeline.py` (SV-Net rollouts, training,
evaluation) once it exists. It edits the SousVide configs, queues pipeline runs, streams
their logs, and shows results. **The scripts stay the source of truth**: the UI only
chooses flags and step ranges and reads the scripts' state files, so anything started
in the browser can be resumed from the shell and vice versa.

---

## 1. Status

| Phase | Scope | Status | Gate result |
|---|---|---|---|
| 0 | Align dummy with the lab setup | done | `figs_pipeline.py` flew backroom + circuit on dummy: 34 s, tracking error max 0.034 m, 614 MiB |
| 1 | Backend core: job queue, config models, log streaming | done | on dummy: 26 tests, real preflight job via the queue, cancel of a running job, bad requests refused |
| 2 | Splat page: `sfm`/`train` split, frontend, 4 GB training | done | retrain from the UI, curve shown, previous model promoted back, flight re-run |
| 3 | Course editor (3D waypoints) | **backend gate passed; browser half pending** | a browser-built course passes the `course` step and flies. On dummy: a course sent the way a browser sends JSON saved as floats and flew (backroom, 173 frames, tracking max 0.074 m, 614 MiB, no dark frames); expert preview of circuit 12.344 s vs recorded flight 12.35 s. Remaining: build and fly one from the page (§6) |
| 4 | SV-Net stages (`svnet_pipeline.py`) | todo | `data_alpha` to `eval_single` end to end from the UI |
| 5 | Hardening: login, systemd, ufw, run diffs, archiving | todo | survives a reboot; reachable on LAN and Tailscale only |
| 6 | Package + installer (cu118 and cu128 profiles) | todo | one command on a fresh clone brings everything up on the RTX 5060 Ti PC |

Machines: **dummy** (MSI laptop, RTX 3050 Ti Laptop 4 GB, 15 GiB RAM, Ubuntu 24.04) is the
build-and-test host now. The **new home PC** (RTX 5060 Ti 16 GB, 16 GB RAM) becomes the main
host once Ubuntu is installed, which is why Phase 6 matters. The **lab machine**
(intellisense05, RTX 2080 8 GB) is campus-only: an install target, not a remote worker.

---

## 2. Running it on dummy

```bash
cd ~/FYP-Radiance && git pull                # see §7 on keeping dummy fast-forward only
cd ui/backend
GALLEY_MACHINE=../machines/dummy.toml .venv/bin/python -m galley
```

Port 8800 is not open in ufw yet (Phase 5), so reach it through an SSH tunnel from the
laptop and browse to http://localhost:8800 (API docs at `/docs`):

```powershell
ssh -L 8800:localhost:8800 hanzo@dummy.stargazer-haddock.ts.net
```

First-time setup of the backend venv (it is **not** the `kitchen` env and must never be
installed into it):

```bash
cd ~/FYP-Radiance/ui/backend
uv venv .venv && uv pip install -p .venv -e '.[test]'
GALLEY_TEST_CONFIGS=~/projects/figs_validation/SousVide/configs .venv/bin/python -m pytest -q
```

To update: edit on the laptop, commit, push; `git pull` on dummy; stop the server with
Ctrl-C and start it again. The frontend build (`ui/frontend/dist/`) is committed, so
dummy needs no Node.js. Rebuild it only after changing `ui/frontend/src/`
(`npm ci && npm run build`, then commit `dist/`).

**Do not run GPU work outside the UI while the UI is in use.** The queue runs one job at
a time and assumes it owns the GPU; a shell run of `figs_pipeline.py` or `ns-viewer` next
to a queued training job will likely run a 4 GB card out of memory.

---

## 3. How it is built

```
Browser (React)  ──HTTP + WebSocket :8800──▶  FastAPI (ui/backend/galley)
                                                 │  SQLite: jobs, log lines
                                                 ▼
                                            one worker ──▶ bash -c 'source figs_env.sh && exec figs_pipeline.py …'
                                                                  │ own process group (cancel kills the whole tree)
                                                                  ▼
                                            SousVide/configs, gsplats/, runs/, .figs_pipeline_state/
```

| File | Role |
|---|---|
| `ui/backend/galley/settings.py` | loads `ui/machines/<host>.toml` (paths, GPU, defaults) |
| `ui/backend/galley/jobs.py` | queue of one; `figs_env.sh` wrapper; streams stdout (tqdm `\r` redraws shown live, not stored); cancel = SIGTERM to the process group, SIGKILL after 10 s |
| `ui/backend/galley/configs.py` | Pydantic models for captures, courses, pilots, frames, methods, nnio; refuses writes that would not round-trip; mirrors captures/courses/pilots into `figs/sousvide_overlay/` |
| `ui/backend/galley/pipeline.py` | builds validated `figs_pipeline.py` command lines; reads step markers, `results.json`, `runs/*.json`; model archive/promote; training metrics |
| `ui/backend/galley/tfevents.py` | dependency-free reader for TensorBoard event files (training curves) |
| `ui/backend/galley/course.py` | course editor: runs `figs/course_tools.py` through `figs_env.sh` (CPU only, outside the GPU queue); geometry cache; one preview at a time |
| `figs/course_tools.py` | kitchen-env helper: point cloud, camera path and boxes in the course frame; FiGS `MinTimeSnap` + `TsFO_to_tXU` preview with the expert's input bounds; KD-tree clearance |
| `ui/frontend/src/pages/Course.tsx`, `src/course/` | the editor page (lazy-loaded with three.js / react-three-fiber), the course model and the 3D view |
| `ui/backend/galley/app.py` | REST + WebSocket routes; optional token (header, or `?token=` for video and WebSocket) |
| `ui/frontend/src/` | pages: Overview, New capture, Scene, Jobs, Job, Configs; `charts.tsx` has the line and bar charts |
| `ui/machines/dummy.toml` | dummy's profile; add one per host |
| `ui/deploy/phase1_gate.sh`, `phase2_train_probe.sh`, `phase3_gate.sh` | the gate and measurement scripts used so far |

Security choices already in place: no generic shell endpoint; every job is an argv list
(never `shell=True`); scene, course and config names are regex-checked; videos must live
in the staging directory; extra `ns-train` options must match an allow-list
(`--pipeline.*`, `--optimizers.*`, one value). Login and firewall rules are Phase 5.

---

## 4. Changes to `figs/figs_pipeline.py` in this cycle

| Change | Why |
|---|---|
| `course` step checks only constrained axes | a `null` (free) waypoint cell became NaN and failed, so the known-good `backroom` + `circuit` pair was refused |
| `simulate` writes `<scene>_flight.partial.mp4`, then renames | imageio's FFMPEG writer refuses a `.partial` extension |
| `transcode` passes `-f mp4` (your fix) | same extension problem for ffmpeg |
| `gsplat` split into `sfm` + `train` (14 steps) | `sfm` runs upstream `generate_gsplat()` with its `ns-train` call intercepted; `train` runs `ns-train` itself so a retrain never repeats SfM |
| new flags `--train-iters`, `--downscale`, `--cache-images`, `--train-vis`, `--train-arg=…`, `--archive-old` | training options; `train` refuses to create a second model unless told to archive the first |
| `gsplat.done` migrates to `sfm.done` + `train.done`; `--redo gsplat` still works | no retrain of scenes finished before the split |
| preflight's "under 6 GB — training will likely OOM" replaced | it was wrong; see §5 |
| `aruco` step stores its 10 s histogram in `results.json` (`aruco.histogram`) | for the UI's detection chart; older scenes need `aruco` re-run to get it |
| Phase 3: `course` and `simulate` fingerprints include a hash of the course file (`course_digest`) | editing a course under the same name used to leave both steps "already done", so the old flight was kept. Existing markers re-run once |
| Phase 3: `course` refuses integer `fo` cells | FiGS's `KF_to_TpFO` takes only float or null: an integer cell silently becomes the previous cell's value (`[0.4, 0]` → `[0.4, 0.4]`, checked against FiGS `11ad36c`). Browsers write 0.0 as 0 |
| Phase 3: `course_inside()`, `course_int_cells()` shared with `course_tools.py` | the editor and the `course` step use the same volume test |

---

## 5. Measured on dummy

**A 4 GB card trains splats.** `ui/deploy/phase2_train_probe.sh` retrained backroom as
`backroom_t4` (workspace symlinked to backroom's, so SfM was reused):

| | Shipped backroom | backroom_t4 on dummy |
|---|---|---|
| Settings | upstream defaults | defaults + `--cache-images cpu`, 960×540, 30k steps |
| Peak VRAM | 5203 MiB (RTX 2080, images on GPU) | **2360 MiB** |
| Training time | 34 min (RTX 2080, other scene) | 60 min 24 s |
| Checkpoint | 371 MB | 375 MB |
| Circuit flight | track error max 0.034 m, pixel std 52.6 | 0.034 m, 52.7, no dark frames |

Most of the reference 5203 MiB was the training images held on the GPU. Denser scenes
(intellisense's checkpoint is twice backroom's) may need
`--train-arg='--pipeline.model.stop-split-at 10000'` on 4 GB; the 16 GB PC will not.
Not yet measured on dummy: SfM time (hloc exhaustive matching on the laptop GPU).

**Course editor on dummy (phase3_gate.sh, 27 Sep):** `course_tools.py geometry` 0.13 s
(backroom: 53,286 sparse points, 300 cameras); fixed-time preview 0.58 s; expert re-time 19 s for
circuit, whose solved duration (12.344 s) matches the recorded flight (12.35 s) against the
file's 12.107 s. Nearest-point clearance put circuit, a known-good flight, 2.7 cm from a sparse
point: isolated SfM outliers. Clearance is now the distance to the 5th-nearest point.
`gate3_loop` (built from the recommended box, sent with integer cells the way a browser does)
saved as floats and flew: re-timed 12.0 s → 8.65 s, 173 frames, tracking max 0.074 m.

**dummy's environment** (audited 2026-09-27):

- Install made by `install_figs.sh` at its default prefix `~/projects/figs_validation`
  (`figs_env.sh` there); SousVide `a2400aa`, FiGS `11ad36c`; torch 2.1.2 + CUDA 11.8,
  nerfstudio 1.1.4, gsplat 1.0.0, pycolmap 4.1.1, OpenCV 5.0.0.
- `sousvide` installed with `pip install --no-deps -e .`, plus `transformers 4.46.3`
  under the install's constraints (torch, numpy, OpenCV unchanged). Three `sousvide`
  modules import `transformers`, which its `pyproject.toml` does not declare.
- A duplicate 17 GB `kitchen` env sits in `~/miniconda3`; the one `.bashrc` activates is
  under `~/projects/figs_validation/miniconda3`.
- Scenes: backroom, flightroom, mid_gate, src_open (shipped) and backroom_t4 (trained on dummy).

---

## 6. What comes next

**Phase 3 — course editor (built; gate pending).** Page *Course editor* (`#/course/<scene>/<course>`):

- 3D view in the course frame (camera up = −z): SfM sparse points (RGB or altitude colours,
  up to 60k sent, gzip), camera path, camera box, recommended waypoint box (flagged when an
  axis is empty because the camera spanned less than 2 × margin), keyframes (red when outside
  the camera box, translucent when an axis is free), yaw arrows, the preview path coloured
  by speed and red where clearance or the volume check fails, the minimum-clearance point, and
  a cursor synced with the charts.
- Tools: Move (gizmo shows only fixed axes), Yaw (rotation about z, kept unwrapped), Add
  (click the tinted plane to insert after the selected keyframe). Undo (Ctrl+Z), Delete, Esc.
- Keyframe table with free (empty) cells, and the full 4 × 5 derivative matrix of the
  selected keyframe. Client-side checks mirror the Pydantic model.
- Preview: FiGS `MinTimeSnap` + `TsFO_to_tXU` with the expert's `hz`, `kT`, `use_l2_time` and
  the frame's mass and thrust coefficient. **Live** mode uses the file's times (kT off, under a
  second). **Re-time like expert** runs the expert's time optimisation (tens of seconds to
  ~2 min): with Viper's kT = 10 the file's `t` values are only SLSQP's starting guess, so the
  flight's timing differs from the file; *Write solved times into the keyframes* makes them
  agree. Charts: speed, acceleration, thrust as a fraction of Viper's limit, largest body rate
  against its limit, clearance (distance to the k-th nearest sparse point, k = 5 by default, so a
  lone SfM outlier does not count; threshold and k adjustable), altitude.
- Save writes floats and one-line `fo` rows (upstream layout), mirrored to the overlay.
  *Save and fly* queues `figs_pipeline.py --from course --stop-after record` with
  `--redo course simulate validate` and shows the tracking error, render check and video.
- Semantic goal marker, saved as `semantic_goal` (label + position) in the course file.
  FiGS/SousVide read only `waypoints` and `forces` (checked in `VehicleRateMPC`,
  `rollout_generator`, `deploy_figs`), so it does not affect flights.

Gate on dummy: `bash ~/FYP-Radiance/ui/deploy/phase3_gate.sh 2>&1 | tee ~/phase3_gate.log`
(tests, tool timings, API checks, lint of every course, a loop built from backroom's box sent
the way a browser sends JSON, saved and flown through the queue). Then the browser half:
Course editor → backroom → *New loop* → drag a keyframe → *Save and fly*; the flight must
pass `course`, `simulate` and `validate`. Not done: rendering the actual splat in the browser
(needs `ns-export gaussian-splat` plus a splat renderer), clearance against the splat rather
than sparse points.

**Phase 4 — SV-Net.** Write `figs/svnet_pipeline.py` as `figs_analysis/SVNet_enablement_plan.md`
§5.4 describes (rollout, observe, train_hist, train_comm, deploy), reusing
`figs_pipeline.py`'s machinery. Add the overlay patch that drops unused depth from
rollouts, `use_compress`, and the invalidation of `train_comm` when `train_hist` reruns.
On dummy, `data_alpha` (~8 GB) and `data_beta` (~88 GB) fit; `data_gamma` only with depth
dropped. UI: method/pilot editors, live loss, TTE/PP table.

**Phase 5 — hardening.** ufw rules for 8800 (LAN + `tailscale0`), a login, systemd units
that run through `figs_env.sh`, run diffs, archive/delete for old runs and cohorts, and
treating `ns-viewer` as a queued GPU job.

**Phase 6 — installer.** Move remaining setup into `install_figs.sh` steps (`sousvide`,
`ui`), pin pycolmap and OpenCV, and add `--cuda-profile cu118|cu128`. The RTX 5060 Ti is
Blackwell (compute capability 12.0): it needs CUDA 12.8+ and PyTorch 2.7+, so it cannot
use the torch 2.1.2 stack at all. That profile needs its own env spec, tiny-cuda-nn built
for architecture 120, rebuilt gsplat, re-checked nerfstudio/hloc/pycolmap/acados, and a
per-profile torch check in `figs_pipeline.py`'s preflight (currently pinned to 2.1.2).
Prototype it on the new PC as soon as Ubuntu is installed.

---

## 7. Traps met in this cycle

- **Don't run git from Claude's Cowork VM against the laptop repo.** Its sandbox cannot
  delete files, so git leaves `.git/index.lock` / `objects/maintenance.lock` behind and
  blocks the next commit. Delete them if you see them.
- **Line endings.** The laptop working copy is CRLF, so `git status` shows every file as
  modified. `git config core.autocrlf true` in that repo quiets it; edit scripts should
  preserve CRLF.
- **dummy diverged once** because `ui/` was copied there by scp and committed locally.
  Set `git config pull.ff only` there and update dummy only by pulling.
- **`--train-arg` needs the `=` form** (`--train-arg=--pipeline.model.x 1`): argparse reads a
  bare `--pipeline…` value as an option.
- **One trained model per scene.** FiGS refuses to guess. Use the UI's Archive/Promote or
  `--archive-old`; archived runs live in `gsplats/workspace/_archive/<scene>/`.
- **Never install anything into `kitchen` without `--no-deps` and a torch check.** Compiled
  extensions are bound to torch 2.1.2.
- **Integer cells in course files.** FiGS's `KF_to_TpFO` reads a JSON integer in `fo` as the
  previous cell's value (or crashes on the first cell), with no error. The old Configs page
  could write these (JSON.stringify turns 0.0 into 0); saves are now written as floats, the
  `course` step refuses integer cells, and `GET /api/courses/{name}/lint` finds them.
- **Course `t` values are a starting guess.** Viper's `kT` makes `MinTimeSnap` re-optimise the
  segment durations (SLSQP, snap cost + kT × total time), so stretching the `t` values has
  limited effect on the flown timing, despite the pipeline's "stretch the t values" hint. To
  fly slower, use a copy of the expert with a smaller `kT` (time costs less) and check it with
  *Re-time like expert*, which shows the times that will actually fly.

---

## 8. Working with Claude on this

Claude's Cowork session cannot SSH to dummy (no route from its sandbox; terminals can only
be clicked, not typed into). What worked: Claude writes scripts and code bundles into
`C:\Users\User\Claude\galley\` and applies code to `D:\Projects\FYP\FYP-Radiance` after
checking checksums; you run things on dummy over your own SSH key and save outputs into the
same folder for Claude to read. Running Claude Code directly on dummy (in `tmux`) would
remove the round trip.

Prompt to start the next session:

> Continue the Galley web console for my FYP pipeline. Phase 3 (course editor) is built; check
> its gate result in ~/phase3_gate.log, then start Phase 4 (SV-Net). Read
> `D:\Projects\FYP\FYP-Radiance\docs\GALLEY_UI.md` first, then the plan doc linked there.
> Code is in `FYP-Radiance/ui/`; the pipeline script is `figs/figs_pipeline.py`. Work on
> my laptop copy and give me commands to run on dummy.
