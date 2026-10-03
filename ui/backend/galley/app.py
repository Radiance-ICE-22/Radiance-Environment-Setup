"""Galley API: configs, jobs, pipeline state. Run with `python -m galley`."""
from __future__ import annotations

import asyncio
import gzip
import hmac
import json
import shutil
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from . import pipeline as pl
from . import svnet as sv
from .configs import FAMILIES, ConfigError, ConfigStore
from .course import Busy, CourseTools, PreviewRequest, ToolError, int_cells
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
    tools = CourseTools(s)

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

    def scene_busy(scene: str) -> bool:
        return any(j["scene"] == scene and j["status"] in ("queued", "running") for j in db.jobs(200, scene))

    @app.get("/api/scenes/{scene}/models", dependencies=[api])
    def models(scene: str):
        return _val(lambda: {"active": pl.trained_models(s, pl.check_scene(scene)),
                             "archived": pl.archived_models(s, scene)})

    @app.post("/api/scenes/{scene}/models/{run}/archive", dependencies=[api])
    def model_archive(scene: str, run: str):
        if scene_busy(scene):
            raise HTTPException(409, "a job for this scene is queued or running")
        return _val(lambda: pl.archive_model(s, scene, run))

    @app.post("/api/scenes/{scene}/models/{run}/promote", dependencies=[api])
    def model_promote(scene: str, run: str):
        if scene_busy(scene):
            raise HTTPException(409, "a job for this scene is queued or running")
        return _val(lambda: pl.promote_model(s, scene, run))

    @app.get("/api/scenes/{scene}/metrics", dependencies=[api])
    def metrics(scene: str, run: str | None = None):
        return _val(lambda: pl.training_metrics(s, scene, run))

    # ── course editor (Phase 3) ───────────────────────────────────────────────
    @app.get("/api/scenes/{scene}/geometry", dependencies=[api])
    def geometry(scene: str, request: Request, margin: float = Query(0.5, ge=0, le=5)):
        """Sparse point cloud, camera path and bounds boxes in the course frame."""
        out = _tool(lambda: tools.geometry(pl.check_scene(scene), margin))
        body = json.dumps(out, separators=(",", ":")).encode()
        if "gzip" in request.headers.get("accept-encoding", "") and len(body) > 4096:
            return Response(gzip.compress(body, 5), media_type="application/json",
                            headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"})
        return Response(body, media_type="application/json")

    @app.post("/api/courses/preview", dependencies=[api])
    def course_preview(req: PreviewRequest):
        """MinTimeSnap through the keyframes with the expert's settings; clearance; volume check."""
        course = _cfg(lambda: store.validate("courses", req.course))   # floats, as saved
        if req.scene:
            pl.check_scene(req.scene)
        return _tool(lambda: tools.preview(req, course))

    @app.get("/api/courses/{name}/lint", dependencies=[api])
    def course_lint(name: str):
        data = _cfg(lambda: store.read("courses", name))
        return {"int_cells": int_cells(data)}

    # ── SV-Net cohorts (Phase 4) ───────────────────────────────────────────────
    @app.get("/api/cohorts", dependencies=[api])
    def cohorts():
        return sv.list_cohorts(s)

    @app.get("/api/cohorts/{cohort}", dependencies=[api])
    def cohort(cohort: str):
        return _val(lambda: sv.cohort_status(s, cohort))

    @app.get("/api/cohorts/{cohort}/video/{name}", dependencies=[api])
    def cohort_video(cohort: str, name: str):
        p = _val(lambda: sv.deployment_video(s, cohort, name))
        if not p:
            raise HTTPException(404, "no such deployment video")
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

    @app.post("/api/jobs/svnet", dependencies=[api], status_code=201)
    def submit_svnet(req: sv.SvnetRun):
        if not sv.script(s).exists():
            raise HTTPException(400, f"{sv.script(s)} not found (it ships next to figs_pipeline.py)")
        argv = _val(lambda: sv.build_argv(s, req))
        saved = sv.cohort_status(s, req.cohort)["config"]
        scene = req.scene or saved.get("scene")
        if not scene or not (req.courses or saved.get("courses")):
            raise HTTPException(400, "a new cohort needs a scene and at least one course")
        steps = req.only or f"{req.from_step or 'start'}..{req.stop_after or 'end'}"
        # the scene column makes Archive/Promote refuse while the cohort flies in that splat
        return {"id": runner.submit("svnet", f"svnet {req.cohort} [{steps}]", argv, req.model_dump(), scene)}

    @app.post("/api/jobs/selftest", dependencies=[api], status_code=201)
    def submit_selftest(req: SelfTest):
        """Harmless job for checking the queue, streaming and cancel end to end."""
        argv = [s.python, "-c", SELFTEST, str(req.seconds)]
        return {"id": runner.submit("selftest", f"selftest {req.seconds}s", argv, req.model_dump(), None)}

    @app.get("/api/jobs", dependencies=[api])
    def jobs(limit: int = Query(50, le=500), scene: str | None = None, cohort: str | None = None):
        if cohort:
            return [j for j in db.jobs(500) if j["kind"] == "svnet" and j["params"].get("cohort") == cohort][:limit]
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

        class Frontend(StaticFiles):
            """index.html must be revalidated on every load, or a browser keeps running the old
            build after a `git pull` (no Cache-Control lets it guess a freshness lifetime);
            the hashed files under assets/ never change, so those can be cached for good."""
            async def get_response(self, path, scope):
                r = await super().get_response(path, scope)
                if path.startswith("assets/") and r.status_code == 200:
                    r.headers["Cache-Control"] = "public, max-age=31536000, immutable"
                else:
                    r.headers["Cache-Control"] = "no-cache"
                return r

        app.mount("/", Frontend(directory=dist, html=True), name="frontend")

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


def _tool(fn):
    try:
        return fn()
    except Busy as e:
        raise HTTPException(409, str(e))
    except ToolError as e:
        raise HTTPException(502, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


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
