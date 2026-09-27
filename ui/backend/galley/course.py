"""Course editor backend (Phase 3): scene geometry and trajectory previews.

The numbers come from figs/course_tools.py, run in the kitchen env through figs_env.sh
like every pipeline job, so they use FiGS's own MinTimeSnap and the pipeline's own
course-frame conversion. These calls are CPU-only (numpy/scipy, CUDA hidden) and short,
so they run directly rather than through the one-at-a-time GPU queue: a preview must not
wait an hour behind a training job. Previews are still serialised by a lock of their own.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .configs import NAME_RE
from .settings import Settings

MARK = "GALLEY_JSON "


class ToolError(RuntimeError):
    pass


class Busy(RuntimeError):
    pass


class PreviewRequest(BaseModel):
    course: dict[str, Any]
    scene: Optional[str] = None
    pilot: str = "Viper"
    frame: str = "carl"
    mode: Literal["fixed", "expert"] = "fixed"
    clearance: float = Field(0.3, ge=0, le=5)
    clearance_k: int = Field(5, ge=1, le=50)

    @field_validator("scene", "pilot", "frame")
    @classmethod
    def _name(cls, v):
        if v is not None and not NAME_RE.match(v):
            raise ValueError("names use letters, digits, '_' and '-'")
        return v


class CourseTools:
    TIMEOUT = {"geometry": 120, "fixed": 90, "expert": 600}

    def __init__(self, s: Settings):
        self.s = s
        self._preview_lock = threading.Lock()
        self._geo: dict[tuple, dict] = {}

    @property
    def script(self) -> Path:
        return self.s.pipeline.parent / "course_tools.py"

    def _run(self, args: list[str], stdin: str | None, timeout: int) -> dict:
        if not self.script.exists():
            raise ToolError(f"{self.script} not found (it ships next to figs_pipeline.py)")
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1", TERM="dumb", NO_COLOR="1")
        script = (f'source "{self.s.env_script}" >/dev/null 2>&1 '
                  f'|| {{ echo "cannot source {self.s.env_script}" >&2; exit 97; }}; exec "$@"')
        try:
            p = subprocess.run(["bash", "-c", script, "galley-tool", self.s.python, str(self.script), *args],
                               input=stdin, capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            raise ToolError(f"course_tools.py timed out after {timeout} s")
        for line in reversed(p.stdout.splitlines()):
            if line.startswith(MARK):
                out = json.loads(line[len(MARK):])
                if not out.get("ok"):
                    raise ToolError(out.get("error", "course_tools.py failed"))
                return out
        tail = (p.stderr or p.stdout).strip().splitlines()[-6:]
        raise ToolError(f"course_tools.py exited {p.returncode} without a result: " + " | ".join(tail))

    # ── geometry: cached per scene until transforms.json / sparse_pc.ply change ──
    def geometry(self, scene: str, margin: float) -> dict:
        ws = self.s.repo / "gsplats" / "workspace" / scene
        stamp = tuple((ws / f).stat().st_mtime if (ws / f).exists() else None
                      for f in ("transforms.json", "sparse_pc.ply"))
        if stamp[0] is None:
            raise ValueError(f"{scene} has no transforms.json yet: run the sfm step first")
        key = (scene, round(margin, 3), stamp)
        if key not in self._geo:
            out = self._run(["geometry", "--project-root", str(self.s.project_root), "--scene", scene,
                             "--margin", str(margin)], None, self.TIMEOUT["geometry"])
            self._geo = {k: v for k, v in self._geo.items() if k[0] != scene}   # keep one per scene
            self._geo[key] = out
        return self._geo[key]

    def preview(self, req: PreviewRequest, course: dict) -> dict:
        if not self._preview_lock.acquire(blocking=False):
            raise Busy("another preview is still solving; try again when it finishes")
        try:
            args = ["preview", "--project-root", str(self.s.project_root), "--pilot", req.pilot,
                    "--frame", req.frame, "--mode", req.mode, "--clearance", str(req.clearance),
                    "--clearance-k", str(req.clearance_k)]
            if req.scene:
                args += ["--scene", req.scene]
            return self._run(args, json.dumps(course), self.TIMEOUT[req.mode])
        finally:
            self._preview_lock.release()


def int_cells(course: dict) -> list[str]:
    """Same check as figs_pipeline.course_int_cells: fo cells saved as JSON integers,
    which FiGS's KF_to_TpFO silently replaces with the previous cell's value."""
    kf = (course.get("waypoints") or {}).get("keyframes") or {}
    return [f"{n}.fo[{i}][{j}]" for n, k in kf.items() for i, row in enumerate(k.get("fo", []))
            for j, v in enumerate(row) if isinstance(v, int) and not isinstance(v, bool)]
