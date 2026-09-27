"""Galley API: configs, jobs, pipeline state. Run with `python -m galley`."""
from __future__ import annotations

import asyncio
import hmac
import shutil
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import pipeline as pl
from .configs import FAMILIES, ConfigError, ConfigStore
from .db import DB
from .jobs import JobRunner
from .settings import Settings, load

SELFTEST = ("import sys, time\n"
            "n = int(sys.argv[1])\n"
            "for i in range(n):\n"
            "    print(f'selftest tick {i + 1}/{n}', flush=True)\n"
            "    print(f'progress {i + 1}/{n}', end='\\r', flush=True)\n"
            "    time.sleep(1)\n"
            "print('selftest done')\n")


class SelfTest(BaseModel):
    seconds: int = Field(10, ge=1, le=600)


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or load()
    db = DB(s.db_path)
    runner = JobRunner(s, db)
    store = ConfigStore(s.configs_dir, s.overlay)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runner.start()
        yield
        await runner.stop()

    app = FastAPI(title="Galley", version="0.1.0", lifespan=lifespan)
    app.state.settings, app.state.db, app.state.runner = s, db, runner

    def auth(request: Request):
        if s.token:
            # header for fetch(); ?token= for <video src> and WebSockets, which cannot set headers
            got = (request.headers.get("authorization", "").removeprefix("Bearer ").strip()
                   or request.query_params.get("token", ""))
            if not hmac.compare_digest(got, s.token):
                raise HTTPException(401, "missing or wrong token")

    api = Depends(auth)

    # ── machine ───────────────────────────────────────────────────────────────
    @app.get("/api/health")
    def health():
        return {"ok": True, "current_job": runner.current,
                "env_script": s.env_script.exists(), "pipeline": s.pipeline.exists()}

    @app.get("/api/machine", dependencies=[api])
    def machine():
        du = shutil.disk_usage(s.project_root) if s.project_root.exists() else None
        return {
            "gpu": {"name": s.gpu_name, "vram_mib": s.vram_mib, "live": _nvidia_smi()},
            "disk": {"free_gb": round(du.free / 1e9, 1), "total_gb": round(du.total / 1e9, 1)} if du else None,
            "paths": {"project_root": str(s.project_root), "pipeline": str(s.pipeline),
                      "overlay": str(s.overlay) if s.overlay else None, "video_dir": str(s.video_dir)},
            "defaults": s.defaults,
        }

    # ── configs ───────────────────────────────────────────────────────────────
    @app.get("/api/configs", dependencies=[api])
    def config_families():
        return {f: store.list(f) for f in FAMILIES}

    @app.get("/api/configs/{family}", dependencies=[api])
    def config_list(family: str):
        return _cfg(lambda: store.list(family))

    @app.get("/api/configs/{family}/{name}", dependencies=[api])
    def config_read(family: str, name: str):
        return _cfg(lambda: store.read(family, name))

    @app.put("/api/configs/{family}/{name}", dependencies=[api])
    def config_write(family: str, name: str, data: dict, overwrite: bool = True):
        return _cfg(lambda: store.write(family, name, data, overwrite))

    @app.post("/api/configs/{family}/validate", dependencies=[api])
    def config_validate(family: str, data: dict):
        if family not in FAMILIES:
            raise HTTPException(404, "unknown family")
        return _cfg(lambda: {"ok": True, "data": store.validate(family, data)})

    # ── pipeline state ────────────────────────────────────────────────────────
    @app.get("/api/scenes", dependencies=[api])
    def scenes():
        return pl.list_scenes(s)

    @app.get("/api/scenes/{scene}", dependencies=[api])
    def scene(scene: str):
        return _val(lambda: pl.scene_status(s, scene))

    @app.get("/api/scenes/{scene}/flight", dependencies=[api])
    def flight(scene: str):
        p = _val(lambda: pl.flight_video(s, scene))
        if not p:
            raise HTTPException(404, "no flight video for this scene")
        return FileResponse(p, media_type="video/mp4")

    @app.get("/api/runs", dependencies=[api])
    def runs(scene: str | None = None):
        return pl.list_runs(s, scene)

    @app.get("/api/videos", dependencies=[api])
    def videos():
        d = s.video_dir
        return [{"name": p.name, "mb": round(p.stat().st_size / 2**20)}
                for p in sorted(d.iterdir()) if p.is_file()] if d.is_dir() else []

    # ── jobs ──────────────────────────────────────────────────────────────────
    @app.post("/api/jobs/figs", dependencies=[api], status_code=201)
    def submit_figs(req: pl.FigsRun):
        argv = _val(lambda: pl.build_argv(s, req))
        steps = req.only or f"{req.from_step or 'start'}..{req.stop_after or 'end'}"
        label = f"figs {req.scene} [{steps}]"
        return {"id": runner.submit("figs", label, argv, req.model_dump(), req.scene)}

    @app.post("/api/jobs/selftest", dependencies=[api], status_code=201)
    def submit_selftest(req: SelfTest):
        """Harmless job for checking the queue, streaming and cancel end to end."""
        argv = [s.python, "-c", SELFTEST, str(req.seconds)]
        return {"id": runner.submit("selftest", f"selftest {req.seconds}s", argv, req.model_dump(), None)}

    @app.get("/api/jobs", dependencies=[api])
    def jobs(limit: int = Query(50, le=500), scene: str | None = None):
        return db.jobs(limit, scene)

    @app.get("/api/jobs/{job_id}", dependencies=[api])
    def job(job_id: int):
        j = db.job(job_id)
        if not j:
            raise HTTPException(404, "no such job")
        return j

    @app.get("/api/jobs/{job_id}/log", dependencies=[api])
    def job_log(job_id: int, after: int = -1, limit: int = Query(5000, le=20000)):
        if not db.job(job_id):
            raise HTTPException(404, "no such job")
        return db.lines(job_id, after, limit)

    @app.post("/api/jobs/{job_id}/cancel", dependencies=[api])
    async def job_cancel(job_id: int):
        if not await runner.cancel(job_id):
            raise HTTPException(409, "job is not queued or running")
        return db.job(job_id)

    @app.websocket("/api/jobs/{job_id}/stream")
    async def job_stream(ws: WebSocket, job_id: int, after: int = -1, token: str | None = None):
        if s.token and not (token and hmac.compare_digest(token, s.token)):
            await ws.close(code=4401)
            return
        await ws.accept()
        j = db.job(job_id)
        if not j:
            await ws.close(code=4404)
            return
        q = runner.subscribe(job_id)       # subscribe first so nothing falls between backlog and live
        try:
            last = after
            for row in db.lines(job_id, after, 20000):
                await ws.send_json({"type": "line", "seq": row["seq"], "line": row["line"]})
                last = row["seq"]
            j = db.job(job_id)
            await ws.send_json({"type": "status", "status": j["status"], "returncode": j["returncode"]})
            if j["status"] not in ("queued", "running"):
                return
            while True:
                msg = await q.get()
                if msg["type"] == "line" and msg["seq"] <= last:
                    continue
                await ws.send_json(msg)
                if msg["type"] == "status" and msg["status"] not in ("queued", "running"):
                    return
        except WebSocketDisconnect:
            pass
        finally:
            runner.unsubscribe(job_id, q)
            try:
                await ws.close()
            except RuntimeError:
                pass

    # ── static frontend (Phase 2+) ────────────────────────────────────────────
    dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if dist.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")

    return app


def _cfg(fn):
    try:
        return fn()
    except FileNotFoundError as e:
        raise HTTPException(404, f"not found: {e}")
    except ConfigError as e:
        raise HTTPException(400, str(e))
    except ValueError as e:  # pydantic ValidationError is a ValueError
        raise HTTPException(422, str(e))


def _val(fn):
    try:
        return fn()
    except ValueError as e:
        raise HTTPException(400, str(e))


def _nvidia_smi() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        used, total, util, temp = (float(x) for x in out.strip().splitlines()[0].split(","))
        return {"used_mib": used, "total_mib": total, "util_pct": util, "temp_c": temp}
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None
