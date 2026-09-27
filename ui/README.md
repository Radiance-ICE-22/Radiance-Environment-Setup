# Galley — web console for the FiGS / SOUS-VIDE pipeline

Wraps `figs/figs_pipeline.py` (and later `figs/svnet_pipeline.py`) in a web UI: edit
configs, queue pipeline runs, stream their logs, browse results. The scripts stay the
source of truth — anything started here can be resumed from the shell and vice versa.

Plan: https://claude.ai/code/artifact/7f7d0bbb-6399-4f1b-bead-21ff96f3f12e

## Layout

| Path | |
|---|---|
| `backend/galley/` | FastAPI app, SQLite job queue, config models, pipeline bridge |
| `backend/tests/` | pytest suite (fake pipeline; set `GALLEY_TEST_CONFIGS` to round-trip real configs) |
| `machines/<host>.toml` | per-host paths, GPU and defaults |
| `frontend/` | React + Vite (Phase 2) |
| `deploy/` | systemd units and ufw rules (Phase 5) |

## Run (development)

The backend has its own venv. **Never install it into `kitchen`**: it only launches
pipeline processes, each of which sources `figs_env.sh` itself.

```bash
cd ~/FYP-Radiance/ui/backend
uv venv .venv && uv pip install -p .venv -e '.[test]'
GALLEY_MACHINE=../machines/dummy.toml .venv/bin/python -m galley     # http://<host>:8800/docs
```

## API (Phase 1)

| Method | Path | |
|---|---|---|
| GET | `/api/health`, `/api/machine` | status, GPU (live `nvidia-smi`), disk |
| GET/PUT | `/api/configs/{family}/{name}` | captures, courses, pilots, frames, methods, nnio; PUT validates and mirrors to the overlay |
| POST | `/api/configs/{family}/validate` | validate without saving |
| GET | `/api/scenes`, `/api/scenes/{scene}` | step markers, `results.json`, trained models (flags scenes FiGS cannot load) |
| GET | `/api/scenes/{scene}/flight` | the flight MP4 |
| GET | `/api/runs`, `/api/videos` | `runs/*.json`, staged videos |
| POST | `/api/jobs/figs` | queue a `figs_pipeline.py` run (flags + `from_step`/`only`/`stop_after`/`redo`) |
| POST | `/api/jobs/selftest` | harmless job for testing the queue |
| GET | `/api/jobs`, `/api/jobs/{id}`, `/api/jobs/{id}/log?after=N` | history and logs |
| POST | `/api/jobs/{id}/cancel` | SIGTERM to the job's process group, SIGKILL after 10 s |
| WS | `/api/jobs/{id}/stream` | backlog, then live lines, progress redraws and status |

Jobs run one at a time, in order: the GPU is treated as exclusive.
