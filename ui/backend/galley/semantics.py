"""Bridge to figs/semantic_pipeline.py and the per-Gaussian feature tables (docs/SEMANTICS.md).

Same contract as pipeline.py and svnet.py: the script is the source of truth. Galley builds its
command line, reads its state (<PROJECT_ROOT>/.semantic_pipeline_state/<scene>/<run>/) and the
tables it exports (SousVide/gsplats/workspace/<scene>/semantics/<run>/<backend>/index.json),
and reads/writes the scene's query annotations (…/semantics/queries.json, course frame).

A table is *stale* when its key (run + checkpoint name + mtime) differs from the scene's active
model — the same key Galley's .splat cache uses, so a stale table never colours a splat whose
records it does not match.
"""
from __future__ import annotations

import json
import os
import re
import secrets
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .pipeline import check_scene, trained_models
from .settings import Settings

STEPS = ["preflight", "cameras", "teachers", "lift", "export"]
SemStep = Literal["preflight", "cameras", "teachers", "lift", "export"]
BACKENDS = ("lift", "fmgs")
RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")


class SemanticRun(BaseModel):
    """One invocation of semantic_pipeline.py. Unset fields fall back to the scene's saved settings,
    then the machine profile's defaults (semantic_feat_width), then the script's defaults."""
    scene: str
    backend: Literal["lift"] = "lift"           # Phase 4 adds "fmgs"
    teachers: list[Literal["clip", "dino"]] = Field(default_factory=list, max_length=2)
    feat_width: Optional[int] = Field(None, ge=64, le=1920)
    dino_width: Optional[int] = Field(None, ge=224, le=1792)
    batch: Optional[int] = Field(None, ge=8, le=1024)
    from_step: Optional[SemStep] = None
    only: Optional[SemStep] = None
    stop_after: Optional[SemStep] = None
    redo: list[SemStep] = []

    @field_validator("scene")
    @classmethod
    def _scene(cls, v):
        return check_scene(v)

    @field_validator("dino_width")
    @classmethod
    def _dino(cls, v):
        if v is not None and v % 14:
            raise ValueError("dino_width must be a multiple of 14")
        return v

    @model_validator(mode="after")
    def _steps(self):
        if self.teachers and "clip" not in self.teachers:
            raise ValueError("teachers must include clip")
        if self.only and (self.from_step or self.stop_after):
            raise ValueError("'only' cannot be combined with 'from_step' or 'stop_after'")
        if self.from_step and self.stop_after and STEPS.index(self.stop_after) < STEPS.index(self.from_step):
            raise ValueError("stop_after comes before from_step")
        return self


def script(s: Settings) -> Path:
    return s.pipeline.parent / "semantic_pipeline.py"


def build_argv(s: Settings, r: SemanticRun) -> list[str]:
    argv = [s.python, str(script(s)), "--project-root", str(s.project_root), "--scene", r.scene,
            "--backend", r.backend]
    fw = r.feat_width or s.defaults.get("semantic_feat_width")
    flags = {"--teachers": ",".join(r.teachers) if r.teachers else None, "--feat-width": fw,
             "--dino-width": r.dino_width, "--batch": r.batch,
             "--from": r.from_step, "--only": r.only, "--stop-after": r.stop_after}
    for k, v in flags.items():
        if v is not None:
            argv += [k, str(v)]
    for step in r.redo:
        argv += ["--redo", step]
    return argv


# ── paths and keys ───────────────────────────────────────────────────────────
def scene_dir(s: Settings, scene: str) -> Path:
    return s.repo / "gsplats" / "workspace" / check_scene(scene)


def semantics_dir(s: Settings, scene: str) -> Path:
    return scene_dir(s, scene) / "semantics"


def active_key(s: Settings, scene: str) -> tuple[str | None, str | None]:
    """(run, key) of the scene's single active model, or (None, None). Key = run-<ckpt stem>-<mtime>,
    exactly CourseTools.splat's cache key."""
    models = trained_models(s, check_scene(scene))
    if len(models) != 1 or not models[0].get("checkpoint"):
        return None, None
    ck = s.repo / models[0]["checkpoint"]
    return models[0]["run"], f"{models[0]['run']}-{ck.stem}-{int(ck.stat().st_mtime)}"


def table_dir(s: Settings, scene: str, run: str, backend: str) -> Path:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    if not RUN_RE.match(run):
        raise ValueError("invalid run name")
    return semantics_dir(s, scene) / run / backend


def _json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def _du(p: Path) -> int:
    return sum(f.stat().st_size for f in p.iterdir() if f.is_file()) if p.is_dir() else 0


# ── status ───────────────────────────────────────────────────────────────────
def status(s: Settings, scene: str) -> dict:
    """Pipeline steps for the active run, and every backend table (active run first; others are
    from archived or older runs and always stale)."""
    run, key = active_key(s, scene)
    st = s.project_root / ".semantic_pipeline_state" / scene / run if run else None
    steps = []
    for name in STEPS:
        m = st / f"{name}.done" if st else None
        lines = m.read_text().strip().splitlines() if m and m.exists() else []
        steps.append({"step": name, "done": bool(lines), "when": lines[-1] if lines else None})
    tables = []
    root = semantics_dir(s, scene)
    for idx in sorted(root.glob("*/*/index.json")) if root.is_dir() else []:
        backend, trun = idx.parent.name, idx.parent.parent.name
        if backend not in BACKENDS:
            continue
        ix = _json(idx, {})
        tables.append({
            "run": trun, "backend": backend, "rows": ix.get("n"), "key": ix.get("key"),
            "stale": ix.get("key") != key or key is None, "active_run": trun == run,
            "teacher_tag": ix.get("teacher_tag"), "order_sha": ix.get("order_sha"),
            "created": ix.get("created"), "mb": round(_du(idx.parent) / 2**20, 1),
            "seen_rows": ix.get("metrics", {}).get("seen_rows"),
            "lift": {k: ix.get("metrics", {}).get("lift", {}).get(k)
                     for k in ("seconds", "passes", "peak_vram_mib", "render_width", "views")},
        })
    tables.sort(key=lambda t: (not t["active_run"], t["run"], t["backend"]))
    ann = load_queries(s, scene)
    return {"scene": scene, "run": run, "key": key, "steps": steps,
            "config": _json(st / "config.json", {}) if st else {},
            "results": _json(st / "results.json", {}) if st else {},
            "tables": tables,
            "ready": [t["backend"] for t in tables if t["active_run"] and not t["stale"]],
            "queries": {"total": len(ann["queries"]),
                        "annotated": sum(q.get("position") is not None for q in ann["queries"])}}


# ── annotations (queries.json, course frame) ─────────────────────────────────
class Annotation(BaseModel):
    model_config = ConfigDict(extra="allow")
    text: str = Field(min_length=1, max_length=200)
    position: Optional[list[float]] = None
    set: str = Field("extra", max_length=64)

    @field_validator("position")
    @classmethod
    def _xyz(cls, v):
        if v is not None and len(v) != 3:
            raise ValueError("position is [x, y, z] (course frame, metres)")
        return v


class Annotations(BaseModel):
    model_config = ConfigDict(extra="allow")
    version: int = 1
    frame: str = "course (x, -y, -z), z down"
    queries: list[Annotation] = Field(default_factory=list, max_length=500)

    @model_validator(mode="after")
    def _unique(self):
        texts = [q.text for q in self.queries]
        if len(texts) != len(set(texts)):
            raise ValueError("each query text may appear once")
        return self


def queries_path(s: Settings, scene: str) -> Path:
    return semantics_dir(s, scene) / "queries.json"


def load_queries(s: Settings, scene: str) -> dict:
    d = _json(queries_path(s, scene), None)
    return d if isinstance(d, dict) and isinstance(d.get("queries"), list) else Annotations().model_dump()


def save_queries(s: Settings, scene: str, data: Annotations) -> dict:
    if not scene_dir(s, scene).is_dir():
        raise ValueError(f"unknown scene {scene}")
    out = data.model_dump(mode="json")
    out["updated"] = datetime.now().isoformat(timespec="seconds")
    p = queries_path(s, scene)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(".queries.json.tmp")
    tmp.write_text(json.dumps(out, indent=2) + "\n")
    os.replace(tmp, p)
    return out


# ── relevancy cache ──────────────────────────────────────────────────────────
class RelevancyCache:
    """Last few per-Gaussian relevancy byte arrays, by opaque id, so the browser fetches the
    (~0.5 MB) bytes separately from the JSON answer and can drop them when it wants."""

    def __init__(self, size: int = 8):
        self.size = size
        self._d: OrderedDict[str, tuple[str, bytes]] = OrderedDict()

    def put(self, scene: str, data: bytes) -> str:
        rid = secrets.token_hex(8)
        self._d[rid] = (scene, data)
        while len(self._d) > self.size:
            self._d.popitem(last=False)
        return rid

    def get(self, scene: str, rid: str) -> bytes | None:
        hit = self._d.get(rid)
        if not hit or hit[0] != scene:
            return None
        self._d.move_to_end(rid)
        return hit[1]


# ── query requests ───────────────────────────────────────────────────────────
class QueryReq(BaseModel):
    text: str = Field(min_length=1, max_length=200)
    backend: Literal["lift", "fmgs"] = "lift"
    negatives: Optional[list[str]] = Field(None, max_length=16)
    threshold: Optional[float] = Field(None, ge=0, le=1)
    rel_alpha: Optional[float] = Field(None, ge=0, le=1)
    standoff: Optional[float] = Field(None, ge=0, le=5)
    margin: Optional[float] = Field(None, ge=0, le=5)
    top: Optional[int] = Field(None, ge=1, le=20)
    relevancy: bool = True

    def settings(self) -> dict:
        return {k: getattr(self, k) for k in ("negatives", "threshold", "rel_alpha", "standoff", "margin", "top")
                if getattr(self, k) is not None}


class LabelsReq(BaseModel):
    backend: Literal["lift", "fmgs"] = "lift"
    index: int = Field(ge=0)
    labels: list[str] = Field(min_length=1, max_length=64)
