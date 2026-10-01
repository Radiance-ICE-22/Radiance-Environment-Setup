# Galley — web console for the pipeline: status and handoff

*Last updated 2026-10-01: the team's airframe (CAD) is drawn in the course editor and the
clearance check subtracts its size (§6, "Drone model"). Phase 4 (SV-Net, unchanged upstream)
built and tested off-machine; its gate on dummy is pending. Written so the next working session (human or
Claude) can pick up without the chat history. The live plan, with diagrams, is the Claude
Doc "SOUS-VIDE Pipeline Console — Implementation Plan":
https://claude.ai/code/artifact/7f7d0bbb-6399-4f1b-bead-21ff96f3f12e*

Galley is a FastAPI + React web UI in `ui/` that drives this repo's pipeline scripts:
`figs/figs_pipeline.py` (capture → splat → course → expert flight) and `figs/svnet_pipeline.py`
(SV-Net rollouts, training, evaluation). It edits the SousVide configs, queues pipeline runs, streams
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
| 3 | Course editor (3D waypoints) | done | a browser-built course passes the `course` step and flies. Browser: `backroom_loop` (New loop, start and end keyframes dragged) saved and flew, re-timed 12 s → 8.55 s, 171 frames, tracking max 0.069 m, no dark frames. Script: a course sent the way a browser sends JSON saved as floats and flew (backroom, 173 frames, tracking max 0.074 m, 614 MiB, no dark frames); expert preview of circuit 12.344 s vs recorded flight 12.35 s |
| 4 | SV-Net stages (`svnet_pipeline.py`) | **built, gate pending** | `data_alpha` to `eval_single` end to end from the UI. Off-machine: 55 backend tests; the script run end to end against stand-in `sousvide` modules (markers, chaining, archive, progress, metrics); headless-browser run of the SV-Net pages. Run `ui/deploy/phase4_gate.sh` on dummy (§6) |
| 5 | Hardening: login, systemd, ufw, run diffs, archiving | todo | survives a reboot; reachable on LAN and Tailscale only |
| 6 | Package + installer (cu118 and cu128 profiles) | todo | one command on a fresh clone brings everything up on the RTX 5060 Ti PC |

Machines: **dummy** (MSI laptop, RTX 3050 Ti Laptop 4 GB, 15 GiB RAM, Ubuntu 24.04) is the
build-and-test host now. The **new home PC** (RTX 5060 Ti 16 GB, 16 GB RAM) becomes the main
host once Ubuntu is installed, which is why Phase 6 matters. The **lab machine**
(intellisense05, RTX 2080 8 GB) is campus-only: an install target, not a remote worker.
**intellisense08** (lab workstation, RTX 2080 8 GB, 62 GB RAM, Ubuntu 22.04, user `yutharsan`) is
reachable over Tailscale. Its install lives at `~/Radiance/figs` (install_figs.sh, every step
done, `sousvide` importable) and this repo is cloned at `~/Radiance/Radiance-Environment-Setup`.
This repo now lives at github.com/Radiance-ICE-22/Radiance-Environment-Setup (moved from
Platinum-Saber/FYP-Radiance, which redirects). Profile:
`ui/machines/intellisense08.toml` (Galley on loopback; reach it with `ssh -L`). Bring it up
with `ui/deploy/bringup.sh`; the deploy scripts find each machine's paths through
`ui/deploy/host.sh`. Claude's Cowork sandbox has no route to the Tailscale network either.

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
| `ui/tools/drone_model.py` | CAD OBJ/MTL → `ui/frontend/public/models/drone.glb` (body frame, simplified, meshopt) + `drone.json` (size, radius, rotors) |
| `ui/frontend/src/course/Drone.tsx` | loads the drone model once; places it with FiGS's attitude |
| `figs/svnet_pipeline.py` | SOUS-VIDE's learning half: rollout, observe, train_hist, train_comm, deploy, calling upstream `sousvide` unchanged; resumable per cohort |
| `ui/backend/galley/svnet.py` | builds `svnet_pipeline.py` command lines; reads `.svnet_pipeline_state/<cohort>/` and `SousVide/cohorts/<cohort>/` |
| `ui/frontend/src/pages/SvNet.tsx` | SV-Net pages: cohort list, new cohort, run controls, rollouts, loss curves, evaluation table and videos |
| `ui/backend/galley/app.py` | REST + WebSocket routes; optional token (header, or `?token=` for video and WebSocket) |
| `ui/frontend/src/` | pages: Overview, New capture, Scene, Jobs, Job, Configs; `charts.tsx` has the line and bar charts |
| `ui/machines/dummy.toml` | dummy's profile; add one per host |
| `ui/deploy/phase1_gate.sh` … `phase4_gate.sh`, `phase2_train_probe.sh` | the gate and measurement scripts used so far |

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

**Phase 3 — course editor (done 27 Sep).** Page *Course editor* (`#/course/<scene>/<course>`):

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

**Drone model in the course editor (1 Oct).** The team's CAD export
(`D:\Projects\FYP\Drone Model\Drone-obj_mtl\Drone\Drone_5_2205.obj`, Fusion, cm, z up, camera
along +y; 1.62 M triangles, 229 MB, prop guards fitted) is converted once by
`ui/tools/drone_model.py` into `ui/frontend/public/models/drone.glb` (159k triangles, ~0.6 MB,
meshopt-compressed) and `drone.json`. Both are committed and copied into `dist/models/` by the
build; the OBJ stays out of git. Re-run the tool after a CAD change (needs numpy and Node for
`npx gltfpack`), then `npm run build` and commit `public/models/` and `dist/`.

- Frame: the GLB is in FiGS's body frame, FRD (x forward, y right, z down, metres), the frame of
  the quaternion in FiGS's state (`fo_to_xu`, scipy `[x, y, z, w]`). Origin: rotor centre in x/y,
  middle of the airframe's height in z (the CAD has no mass properties; `--origin` overrides).
- Measured from the CAD: 28.9 cm long × 33.1 cm wide × 10.1 cm tall; rotors at ±7.07 cm forward,
  ±8.6 cm sideways (motor-to-motor diagonal 22.3 cm); prop radius 6.5 cm (5"); bounding sphere
  **0.19 m** from the origin, guards included.
- Editor: the drone sits level on the selected keyframe (else the first); with a preview, it sits
  at the chart cursor with FiGS's own attitude (`course_tools.py preview` now returns `quat`,
  tXU columns 7–10). *▶ Play* (0.25–2×) flies the preview in real time, moving the cursor
  through the charts. *Drone* in the view toolbar hides it.
- Clearance: `course_tools.py preview --body-radius R` (API `body_radius`, 0–2 m, default 0) makes
  every distance a **gap**: centre distance minus R. The editor sends the model's radius by
  default (*Subtract drone size*), so its threshold is now a minimum gap, default 0.15 m
  (≈ 0.34 m centre distance; the old default was 0.3 m centre distance). The red wireframe
  sphere at the closest point is the drone's sphere. The sphere is conservative above and
  below surfaces (the airframe is 10 cm tall, the sphere 38 cm across).
- Checked off-machine with FiGS `11ad36c` and SousVide `a2400aa` configs: circuit over a
  synthetic floor at 0.7 m gives min 0.70 m without and 0.51 m with `--body-radius 0.19`; the
  returned attitude's body z matches the thrust direction (g − a) at every sample and its yaw
  matches the flat output's. Gate on dummy: open the editor, Preview, ▶ Play; the drone should
  bank into the turns of circuit and the clearance chart title should say "Gap".
- Not done: the drone in the SV-Net evaluation view (replaying rollouts), spinning props, and a
  shape-aware clearance (an ellipsoid or the real hull instead of a sphere).

**Phase 4 — SV-Net (built; gate pending).** Get SOUS-VIDE's learning half working *unchanged*
first; semantic feature fields (LangSplat / FMGS style) come after, as a separate step.

`figs/svnet_pipeline.py` replaces `notebooks/sous_vide_examples.ipynb` step for step:

| Step | Upstream call | Notes |
|---|---|---|
| `preflight` | — | kitchen, torch 2.1.2, `sousvide` imports, configs, one model in the scene, integer course cells, data-size estimate against free disk; adds `cohorts/` to `SousVide/.git/info/exclude` |
| `rollout` | `rollout_generator.generate_rollout_data(cohort, courses, scene, method, expert, frame, Nro_ds, use_compress)` | the expert (Viper MPC) flies each course many times, randomised; long, GPU |
| `observe` | `observation_generator.generate_observation_data(cohort, roster, subsample)` | per-pilot network inputs |
| `train_hist` | `train_policy.train_roster(cohort, roster, "histNet", N, lim_sv, lr, batch_size)` | |
| `train_comm` | `train_roster(..., "commNet", N, regen=True, deployment=(course, scene, eval))` | regenerates observations through the new histNet, as upstream requires |
| `deploy` | `deploy_figs.deploy_roster(..., mode="visualize", show_table=True)` | expert + students; metrics and last-rollout videos in `cohorts/<cohort>/deployment_data/` |

Settings are saved per cohort (`.svnet_pipeline_state/<cohort>/config.json`), so a resume needs
only `--cohort`. Each marker's fingerprint includes the previous step's marker: re-running a
step re-runs everything after it (this is also what makes `train_comm` follow `train_hist`).
Only three things are added around upstream, none changing what it computes: plain progress lines
(rich's bars print nothing without a terminal) plus per-epoch losses in `live_*.jsonl`; old
rollout/observation data moved to `cohorts/<cohort>/_archive/` before a re-run (upstream overwrites
by index, so a smaller re-run would mix old and new files); and a second tracking-error figure in
`deploy` (see §7). `--fresh histNet|commNet` archives an existing network, because upstream
otherwise keeps training the one already on disk.

First gate run (27 Sep, cohort `p4_smoke`): preflight 4.6 s; rollout 2 min 12 s, 111 rollouts
kept, 4,440 samples, 5.41 GB, peak 614 MiB; observe 33 s, commNet observations 2.68 GB, peak
3426 MiB; histNet 200 epochs in 76 s (train 1.249, test 1.287). commNet then ran out of memory
in epoch 2: every step ran in one process, and the splat, the observe step's tensors and
PyTorch's cache stayed on the 4 GB card. Each step now runs in its own child process, and
`deploy_roster` calls are followed by `gc.collect()` + `torch.cuda.empty_cache()`. If commNet
still runs out: smaller `--nro-ds` (needs `--redo rollout`), `--comm-eval none`, or
`--batch-size 32`.

UI: *SV-Net* page (`#/svnet`): cohort list; new cohort (scene with one model, courses, rollout
method, students, epochs, in-loop and final evaluation; "Preflight only" prints the size
estimate); per cohort (`#/svnet/<cohort>`): steps, Continue / run steps / force redo / fresh
network, rollouts kept per course, live and saved loss curves (train and test), the evaluation
table and the expert's and students' videos. An SV-Net job also counts as a job for its scene,
so Archive/Promote on the Scene page wait for it. dummy's profile pre-fills in-loop evaluation
with `eval_single` (`svnet_comm_eval`); the notebook's `eval_nominal` flies 10 full-course
rollouts of expert and student at every save.

Gate on dummy: `tmux new -d -s p4 'bash ~/FYP-Radiance/ui/deploy/phase4_gate.sh > ~/phase4_gate.log 2>&1'`
(tests, preflight with the size estimate, then cohort `p4_smoke` = backroom + circuit,
`data_alpha`, Maverick, notebook epochs 200/300, `eval_single`, through Galley's queue; a few
hours). Watch it on the SV-Net page through the usual tunnel. Estimate for circuit: ~108
rollouts, ~4,300 samples, ~6 GB. Deferred until it works as is: the depth-dropping overlay patch,
`use_compress` by default, `data_beta`/`data_gamma`, pilot editors.

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
- **Upstream's flight metric is not a distance.** `sousvide.flight.flight_helper.compute_flight_metrics`
  computes `np.linalg.norm(P[i] - Pd, axis=0)` (the whole reference path per axis) and takes the
  smallest of the three, instead of `axis=1` (distance to each reference point). A path that stays
  in one plane scores ~0 whatever the error. `deploy` reports it as `upstream_tte`/`upstream_pp` and
  adds a per-point tracking error computed the way `rollout_generator`'s `tol_select` check does.
  Upstream also uses its TTE to keep commNet's "best" checkpoint when in-loop evaluation is on.
- **sousvide keeps training what is on disk.** `Pilot()` loads `roster/<pilot>/<net>.pt` if it
  exists, so a second `train_hist` continues from the first. Use `--fresh` to start over.
- **Rich progress bars are silent without a terminal.** Upstream's progress goes through rich,
  which draws nothing when stdout is a pipe (Galley, `tee`, tmux logs); `svnet_pipeline.py`
  substitutes plain lines.

---

## 8. Working with Claude on this

Claude's Cowork session cannot SSH to dummy (no route from its sandbox; terminals can only
be clicked, not typed into). What worked: Claude writes scripts and code bundles into
`C:\Users\User\Claude\galley\` and applies code to `D:\Projects\FYP\FYP-Radiance` after
checking checksums; you run things on dummy over your own SSH key and save outputs into the
same folder for Claude to read. Running Claude Code directly on dummy (in `tmux`) would
remove the round trip.

Prompt to start the next session:

> Continue the Galley web console for my FYP pipeline. Phase 4 (SV-Net) is built: check its gate
> in ~/phase4_gate.log on dummy, then plan semantic feature fields. Read
> `D:\Projects\FYP\FYP-Radiance\docs\GALLEY_UI.md` first, then the plan doc linked there.
> Code is in `FYP-Radiance/ui/`; the pipeline script is `figs/figs_pipeline.py`. Work on
> my laptop copy and give me commands to run on dummy.
