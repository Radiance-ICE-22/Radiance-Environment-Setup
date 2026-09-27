"""Bridge to figs_pipeline.py: build validated command lines, read its on-disk state.

The script stays the source of truth. The UI never re-implements a step; it only
chooses flags and step ranges, and reads .figs_pipeline_state/<scene>/ and runs/*.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .settings import Settings

STEPS = ["preflight", "probe", "transcode", "aruco", "config", "patch", "gsplat",
         "verify", "bounds", "course", "simulate", "validate", "record"]
Step = Literal["preflight", "probe", "transcode", "aruco", "config", "patch", "gsplat",
               "verify", "bounds", "course", "simulate", "validate", "record"]
SCENE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")


class FigsRun(BaseModel):
    """One invocation of figs_pipeline.py. Unset flags fall back to the script's defaults."""
    scene: str
    video: Optional[str] = None
    marker_id: Optional[int] = Field(None, ge=0, le=49)
    marker_length: Optional[float] = Field(None, gt=0, lt=5)
    num_images: Optional[int] = Field(None, gt=0, le=5000)
    num_marked: Optional[int] = Field(None, ge=0)
    width: Optional[int] = Field(None, gt=0, le=8192)
    height: Optional[int] = Field(None, gt=0, le=8192)
    fps: Optional[int] = Field(None, gt=0, le=240)
    course: Optional[str] = None
    frame: Optional[str] = None
    pilot: Optional[str] = None
    method: Optional[str] = None
    margin: Optional[float] = Field(None, ge=0, le=5)
    dataloader_workers: Optional[int] = Field(None, ge=0, le=32)
    allow_outside: bool = False
    from_step: Optional[Step] = None
    only: Optional[Step] = None
    stop_after: Optional[Step] = None
    redo: list[Step] = []

    @field_validator("scene", "course", "frame", "pilot", "method")
    @classmethod
    def _name(cls, v):
        if v is not None and not SCENE_RE.match(v):
            raise ValueError("names use letters, digits, '_' and '-'")
        return v

    @model_validator(mode="after")
    def _steps(self):
        if self.only and (self.from_step or self.stop_after):
            raise ValueError("'only' cannot be combined with 'from_step' or 'stop_after'")
        if self.from_step and self.stop_after and STEPS.index(self.stop_after) < STEPS.index(self.from_step):
            raise ValueError("stop_after comes before from_step")
        if self.num_marked is not None and self.num_images is not None and self.num_marked > self.num_images:
            raise ValueError("num_marked cannot exceed num_images")
        return self


def build_argv(s: Settings, r: FigsRun) -> list[str]:
    argv = [s.python, str(s.pipeline), "--project-root", str(s.project_root), "--scene", r.scene]
    if r.video:
        argv += ["--video", str(resolve_video(s, r.video))]
    flags = {
        "--marker-id": r.marker_id, "--marker-length": r.marker_length,
        "--num-images": r.num_images, "--num-marked": r.num_marked,
        "--width": r.width, "--height": r.height, "--fps": r.fps,
        "--course": r.course, "--frame": r.frame, "--pilot": r.pilot, "--method": r.method,
        "--margin": r.margin, "--dataloader-workers": r.dataloader_workers,
        "--from": r.from_step, "--only": r.only, "--stop-after": r.stop_after,
    }
    for k, v in flags.items():
        if v is not None:
            argv += [k, str(v)]
    for step in r.redo:
        argv += ["--redo", step]
    if r.allow_outside:
        argv.append("--allow-outside")
    return argv


def resolve_video(s: Settings, video: str) -> Path:
    """Videos must live in the staging directory; no arbitrary filesystem paths."""
    p = (s.video_dir / video).resolve()
    if s.video_dir.resolve() not in p.parents:
        raise ValueError("video must be a file inside the video staging directory")
    if not p.is_file():
        raise ValueError(f"video not found: {video}")
    return p


def check_scene(scene: str) -> str:
    if not SCENE_RE.match(scene):
        raise ValueError("invalid scene name")
    return scene


# ── readers ──────────────────────────────────────────────────────────────────
def scene_status(s: Settings, scene: str) -> dict:
    check_scene(scene)
    state = s.state_dir / scene
    steps = []
    known = list(STEPS) + sorted(p.stem for p in state.glob("*.done") if p.stem not in STEPS) if state.is_dir() else list(STEPS)
    for name in known:
        m = state / f"{name}.done"
        entry = {"step": name, "done": m.exists(), "when": None, "fingerprint": None}
        if m.exists():
            lines = m.read_text().strip().splitlines()
            entry["fingerprint"] = lines[0] if lines else None
            entry["when"] = lines[-1] if lines else None
        steps.append(entry)
    results = {}
    rp = state / "results.json"
    if rp.exists():
        try:
            results = json.loads(rp.read_text())
        except ValueError:
            results = {"error": "results.json unreadable"}
    return {"scene": scene, "steps": steps, "results": results, "models": trained_models(s, scene)}


def trained_models(s: Settings, scene: str) -> list[dict]:
    out_dir = s.repo / "gsplats" / "workspace" / "outputs" / scene
    models = []
    for cfg in sorted(out_dir.rglob("config.yml")) if out_dir.is_dir() else []:
        ckpts = sorted((cfg.parent / "nerfstudio_models").glob("*.ckpt"))
        models.append({
            "run": cfg.parent.name,
            "config": str(cfg.relative_to(s.repo)),
            "checkpoint": str(ckpts[-1].relative_to(s.repo)) if ckpts else None,
            "checkpoint_mb": round(ckpts[-1].stat().st_size / 2**20) if ckpts else None,
        })
    return models


def list_scenes(s: Settings) -> list[dict]:
    ws = s.repo / "gsplats" / "workspace"
    names = set()
    for base in (ws, ws / "outputs", s.state_dir):
        if base.is_dir():
            names |= {p.name for p in base.iterdir() if p.is_dir() and SCENE_RE.match(p.name) and p.name != "outputs"}
    out = []
    for n in sorted(names):
        models = trained_models(s, n)
        out.append({
            "scene": n,
            "has_workspace": (ws / n / "transforms.json").exists(),
            "models": len(models),
            "loadable": len(models) == 1,   # FiGS refuses 0 or >1 config.yml per scene
            "has_state": (s.state_dir / n).is_dir(),
        })
    return out


def list_runs(s: Settings, scene: str | None = None) -> list[dict]:
    d = s.repo / "runs"
    runs = []
    for p in sorted(d.glob("*.json"), reverse=True) if d.is_dir() else []:
        try:
            rec = json.loads(p.read_text())
        except ValueError:
            continue
        if scene and rec.get("scene") != scene:
            continue
        rec["_file"] = p.name
        runs.append(rec)
    return runs


def flight_video(s: Settings, scene: str) -> Path | None:
    check_scene(scene)
    p = s.repo / "outputs" / "flights" / f"{scene}_flight.mp4"
    return p if p.is_file() else None
