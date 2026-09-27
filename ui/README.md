# Galley — web console for the FiGS / SOUS-VIDE pipeline

Wraps `figs/figs_pipeline.py` (and later `figs/svnet_pipeline.py`) in a web UI: edit
configs, queue pipeline runs, stream their logs, browse results. The scripts stay the
source of truth — anything started here can be resumed from the shell and vice versa.

**Status, measured results and next steps: [`docs/GALLEY_UI.md`](../docs/GALLEY_UI.md).**
Plan with diagrams: https://claude.ai/code/artifact/7f7d0bbb-6399-4f1b-bead-21ff96f3f12e

## Layout

| Path | |
|---|---|
| `backend/galley/` | FastAPI app, SQLite job queue, config models, pipeline bridge, TensorBoard reader |
| `backend/galley/course.py` | course editor backend: runs `figs/course_tools.py` (kitchen env, CPU only) for geometry and previews |
| `backend/tests/` | pytest suite (45 tests; fake pipeline and fake course tools; set `GALLEY_TEST_CONFIGS` to round-trip real configs) |
| `machines/<host>.toml` | per-host paths, GPU and defaults (`dummy.toml` so far) |
| `frontend/` | React + Vite UI; `frontend/dist/` is the committed build the backend serves |
| `deploy/phase1_gate.sh` | installs the backend on a host, runs the tests, drives the API with curl |
| `deploy/phase2_train_probe.sh` | measures splat training on a small GPU by retraining backroom as `backroom_t4` |
| `deploy/phase3_gate.sh` | course editor gate: tool timings, API checks, course lint, a browser-style course flown through the queue |

## Run

The backend has its own venv. **Never install it into `kitchen`**: it only launches
pipeline processes, each of which sources `figs_env.sh` itself.

```bash
cd ~/FYP-Radiance/ui/backend
uv venv .venv && uv pip install -p .venv -e '.[test]'
GALLEY_MACHINE=../machines/dummy.toml .venv/bin/python -m galley     # UI: http://<host>:8800  API docs: /docs
```

Port 8800 is not opened in ufw yet; from another machine use
`ssh -L 8800:localhost:8800 <user>@<host>` and browse to http://localhost:8800.
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
