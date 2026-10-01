# Galley — web console for the FiGS / SOUS-VIDE pipeline

Wraps `figs/figs_pipeline.py` and `figs/svnet_pipeline.py` in a web UI: edit
configs, queue pipeline runs, stream their logs, browse results. The scripts stay the
source of truth — anything started here can be resumed from the shell and vice versa.

**Status, measured results and next steps: [`docs/GALLEY_UI.md`](../docs/GALLEY_UI.md).**
Plan with diagrams: https://claude.ai/code/artifact/7f7d0bbb-6399-4f1b-bead-21ff96f3f12e

## Layout

| Path | |
|---|---|
| `backend/galley/` | FastAPI app, SQLite job queue, config models, pipeline bridge, TensorBoard reader |
| `backend/galley/course.py` | course editor backend: runs `figs/course_tools.py` (kitchen env, CPU only) for geometry and previews |
| `backend/galley/svnet.py` | SV-Net bridge: `svnet_pipeline.py` command lines, cohort state readers |
| `backend/tests/` | pytest suite (60 tests; fake pipeline, course tools and SV-Net script; set `GALLEY_TEST_CONFIGS` to round-trip real configs) |
| `machines/<host>.toml` | per-host paths, GPU and defaults (`dummy.toml`, `intellisense08.toml`) |
| `../run_ui.sh` | start / stop / status / logs for Galley on any host (creates the venv when needed) |
| `tools/drone_model.py` | CAD OBJ/MTL → `frontend/public/models/drone.glb` + `drone.json` for the course editor |
| `design/galley-figma-plugin/` | Figma dev plugin that builds the Windows 7 ribbon redesign screens (see `docs/GALLEY_UI.md` §6) |
| `deploy/host.sh`, `deploy/bringup.sh` | per-host paths for the deploy scripts; bring an installed host up to date and check it end to end |
| `frontend/` | React + Vite UI; `frontend/dist/` is the committed build the backend serves |
| `deploy/phase1_gate.sh` | installs the backend on a host, runs the tests, drives the API with curl |
| `deploy/phase2_train_probe.sh` | measures splat training on a small GPU by retraining backroom as `backroom_t4` |
| `deploy/phase3_gate.sh` | course editor gate: tool timings, API checks, course lint, a browser-style course flown through the queue |
| `deploy/phase4_gate.sh` | SV-Net gate: preflight and size estimate, then a `data_alpha` cohort end to end through the queue (hours) |

## Run

From the repo root:

```bash
./run_ui.sh              # start in tmux session "galley"; waits for /api/health; prints how to open it
./run_ui.sh --pull       # git pull --ff-only, then (re)start
./run_ui.sh status       # up? which profile? a job running?
./run_ui.sh logs         # follow <data_dir>/galley.log
./run_ui.sh stop         # refuses while a pipeline job runs (--force to override)
./run_ui.sh restart | fg # fg = run in this terminal
```

`start` finds the machine profile (`$GALLEY_MACHINE`, else `ui/machine.toml`, else
`ui/machines/<hostname prefix>.toml`), creates `backend/.venv` with uv or reinstalls it when
`backend/pyproject.toml` changed, checks `figs_env.sh`, `figs_pipeline.py`, `frontend/dist/`
and whether something else holds the GPU, then starts the server. The backend has its own
venv. **Never install it into `kitchen`**: it only launches pipeline processes, each of which
sources `figs_env.sh` itself. Manual equivalent:
`cd ui/backend && uv venv .venv && uv pip install -p .venv -e '.[test]' && .venv/bin/python -m galley`.

The profiles bind to loopback (no login until Phase 5). From the laptop: in VS Code
Remote-SSH, Ports tab → forward 8800; or `ssh -N -L 18800:localhost:8800 <user>@<host>` and
browse to http://localhost:18800 (Windows often reserves 8800 itself).
Set `GALLEY_TOKEN` (or `[server] token` in the machine file) to require a token.

The built frontend (`frontend/dist/`) is committed, so target machines need no Node.js.
To change the UI, on any machine with Node 20+:

```bash
cd ui/frontend && npm ci && npm run build     # then commit dist/
npm run dev                                    # live reload, proxies /api to :8800
```

## Pages

| Page | What it does |
|---|---|
| Overview | scenes (flags any FiGS cannot load), recent jobs, flight run records, live GPU and disk |
| New capture | video from `video_captures/` → splat; scene-name substring check, marker fields, match-pair count, training options; stops after `bounds` by default |
| Scene | step markers; active and archived models with Archive / Promote; reconstruction, training and flight stats; training curve (TensorBoard); ArUco detections per 10 s; flight video; fly a course; retrain; run arbitrary steps |
| Jobs / Job | queue, live log over WebSocket with the current progress line, cancel |
| Course editor | 3D course over the scene's sparse point cloud and camera path, in the course frame (z down); keyframe table with free cells and derivative matrix; move/yaw gizmos, click-to-add; live minimum-snap preview (FiGS `MinTimeSnap`) with speed, acceleration, thrust and body-rate charts against the expert's bounds, clearance and volume checks; re-time like the expert; save; fly the expert; semantic-goal marker. Loaded on demand (three.js is ~1 MB) |
| SV-Net | cohorts of `svnet_pipeline.py`: new cohort (scene, courses, rollout method, students, epochs, evaluation; preflight-only size estimate), per-cohort steps and run controls (continue, step ranges, force redo, fresh network), rollouts kept per course, live and saved loss curves, evaluation table (per-point tracking error next to upstream's TTE/PP) and deployment videos |
| Configs | JSON editor for every SousVide config family with server-side validation; captures, courses and pilots are mirrored into `figs/sousvide_overlay/` |

Training forms pre-fill from the machine file's `[defaults]` (`cache_images`, `train_vis`,
`downscale`).

## API

| Method | Path | |
|---|---|---|
| GET | `/api/health`, `/api/machine` | status, GPU (live `nvidia-smi`), disk, machine defaults |
| GET | `/api/configs`, `/api/configs/{family}` | list configs |
| GET/PUT | `/api/configs/{family}/{name}` | read / validate and write (`?overwrite=false` to create only); mirrors captures, courses, pilots to the overlay |
| POST | `/api/configs/{family}/validate` | validate without saving |
| GET | `/api/scenes`, `/api/scenes/{scene}` | step markers, `results.json`, trained models |
| GET | `/api/scenes/{scene}/models` | active and archived training runs |
| POST | `/api/scenes/{scene}/models/{run}/archive` | move the active run to `gsplats/workspace/_archive/<scene>/`; clears `verify`/`simulate`/`validate` markers |
| POST | `/api/scenes/{scene}/models/{run}/promote` | make an archived run active (archiving the current one); 409 while a job for the scene is queued or running |
| GET | `/api/scenes/{scene}/metrics?run=` | TensorBoard scalars of the active (or given) run |
| GET | `/api/scenes/{scene}/flight` | the flight MP4 |
| GET | `/api/runs`, `/api/videos` | `runs/*.json`, staged videos |
| GET | `/api/scenes/{scene}/geometry?margin=0.5` | sparse points, camera path, camera and waypoint boxes in the course frame (gzip; cached until the files change) |
| POST | `/api/courses/preview` | `{course, scene?, pilot, frame, mode: fixed\|expert, clearance}` → sampled trajectory, inputs vs bounds, clearance, volume check; 409 while another preview runs |
| GET | `/api/courses/{name}/lint` | integer `fo` cells that FiGS would misread |
| POST | `/api/jobs/svnet` | queue a `svnet_pipeline.py` run: cohort, settings (saved per cohort), `from_step`/`only`/`stop_after`/`redo`, `fresh` |
| GET | `/api/cohorts`, `/api/cohorts/{cohort}` | cohort list; config, step markers, `results.json`, live losses, disk use |
| GET | `/api/cohorts/{cohort}/video/{sim_<course>_<pilot>_rgb.mp4}` | deployment video |
| GET | `/api/jobs?cohort=` | jobs of one cohort |
| POST | `/api/jobs/figs` | queue a `figs_pipeline.py` run: flags, training options, `from_step`/`only`/`stop_after`/`redo` |
| POST | `/api/jobs/selftest` | harmless job for testing the queue |
| GET | `/api/jobs`, `/api/jobs/{id}`, `/api/jobs/{id}/log?after=N` | history and logs |
| POST | `/api/jobs/{id}/cancel` | SIGTERM to the job's process group, SIGKILL after 10 s |
| WS | `/api/jobs/{id}/stream` | backlog, then live lines, progress redraws and status |

Course writes (`PUT /api/configs/courses/…`) are saved from the validated model, so every
`fo` cell and `t` is a float, with each `fo` row on one line.

Jobs run one at a time, in order: the GPU is treated as exclusive. Course previews and
geometry are CPU-only calls that bypass the queue (CUDA hidden), so they work during training. Work started outside the
UI (a shell pipeline run, `ns-viewer`) is invisible to the queue, so avoid it while jobs run.
