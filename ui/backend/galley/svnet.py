"""Bridge to figs/svnet_pipeline.py (Phase 4): build its command lines, read cohort state.

Same contract as pipeline.py: the script is the source of truth. The UI chooses flags and
step ranges and reads .svnet_pipeline_state/<cohort>/ (config.json, *.done, results.json,
live_*.jsonl) and SousVide/cohorts/<cohort>/ (sizes, deployment videos).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .settings import Settings

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")
STEPS = ["preflight", "rollout", "observe", "train_hist", "train_comm", "deploy"]
SvStep = Literal["preflight", "rollout", "observe", "train_hist", "train_comm", "deploy"]


class SvnetRun(BaseModel):
    """One invocation of svnet_pipeline.py. Unset fields fall back to the cohort's saved
    settings, then to the script's defaults (the upstream notebook's)."""
    cohort: str
    scene: Optional[str] = None
    courses: list[str] = Field(default_factory=list, max_length=16)
    method: Optional[str] = None
    roster: list[str] = Field(default_factory=list, max_length=8)
    expert: Optional[str] = None
    frame: Optional[str] = None
    nro_ds: Optional[int] = Field(None, ge=1, le=500)
    use_compress: Optional[bool] = None
    subsample: Optional[float] = Field(None, gt=0, le=1)
    hist_epochs: Optional[int] = Field(None, ge=1, le=5000)
    comm_epochs: Optional[int] = Field(None, ge=1, le=5000)
    lr: Optional[float] = Field(None, gt=0, le=1)
    batch_size: Optional[int] = Field(None, ge=1, le=4096)
    lim_sv: Optional[int] = Field(None, ge=1, le=5000)
    comm_eval: Optional[str] = None           # a method name, or "none"
    deploy_course: Optional[str] = None
    deploy_method: Optional[str] = None
    fresh: list[Literal["histNet", "commNet"]] = []
    from_step: Optional[SvStep] = None
    only: Optional[SvStep] = None
    stop_after: Optional[SvStep] = None
    redo: list[SvStep] = []

    @field_validator("cohort", "scene", "method", "expert", "frame", "comm_eval", "deploy_course", "deploy_method")
    @classmethod
    def _name(cls, v):
        if v is not None and not NAME_RE.match(v):
            raise ValueError("names use letters, digits, '_' and '-'")
        return v

    @field_validator("courses", "roster")
    @classmethod
    def _names(cls, v):
        for x in v:
            if not NAME_RE.match(x):
                raise ValueError(f"invalid name {x!r}")
        return v

    @model_validator(mode="after")
    def _steps(self):
        if self.only and (self.from_step or self.stop_after):
            raise ValueError("'only' cannot be combined with 'from_step' or 'stop_after'")
        if self.from_step and self.stop_after and STEPS.index(self.stop_after) < STEPS.index(self.from_step):
            raise ValueError("stop_after comes before from_step")
        return self


def script(s: Settings) -> Path:
    return s.pipeline.parent / "svnet_pipeline.py"


def state_dir(s: Settings) -> Path:
    return s.project_root / ".svnet_pipeline_state"


def cohorts_dir(s: Settings) -> Path:
    return s.repo / "cohorts"


def check_cohort(c: str) -> str:
    if not NAME_RE.match(c):
        raise ValueError("invalid cohort name")
    return c


def build_argv(s: Settings, r: SvnetRun) -> list[str]:
    argv = [s.python, str(script(s)), "--project-root", str(s.project_root), "--cohort", r.cohort]
    flags = {
        "--scene": r.scene, "--method": r.method, "--expert": r.expert, "--frame": r.frame,
        "--nro-ds": r.nro_ds, "--subsample": r.subsample, "--hist-epochs": r.hist_epochs,
        "--comm-epochs": r.comm_epochs, "--lr": r.lr, "--batch-size": r.batch_size, "--lim-sv": r.lim_sv,
        "--comm-eval": r.comm_eval, "--deploy-course": r.deploy_course, "--deploy-method": r.deploy_method,
        "--from": r.from_step, "--only": r.only, "--stop-after": r.stop_after,
        "--courses": ",".join(r.courses) if r.courses else None,
        "--roster": ",".join(r.roster) if r.roster else None,
        "--use-compress": None if r.use_compress is None else ("yes" if r.use_compress else "no"),
    }
    for k, v in flags.items():
        if v is not None:
            argv += [k, str(v)]
    for net in r.fresh:
        argv += ["--fresh", net]
    for step in r.redo:
        argv += ["--redo", step]
    return argv


# ── readers ──────────────────────────────────────────────────────────────────
def _json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def _du(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else 0


def _steps(st: Path) -> list[dict]:
    out = []
    for name in STEPS:
        m = st / f"{name}.done"
        lines = m.read_text().strip().splitlines() if m.exists() else []
        out.append({"step": name, "done": m.exists(), "when": lines[-1] if lines else None})
    return out


def list_cohorts(s: Settings) -> list[dict]:
    names = set()
    for base in (state_dir(s), cohorts_dir(s)):
        if base.is_dir():
            names |= {p.name for p in base.iterdir() if p.is_dir() and NAME_RE.match(p.name)}
    out = []
    for n in sorted(names):
        st = state_dir(s) / n
        cfg = _json(st / "config.json", {})
        steps = _steps(st)
        res = _json(st / "results.json", {})
        dep = res.get("deploy", {}).get("pilots", {})
        out.append({"cohort": n, "scene": cfg.get("scene"), "courses": cfg.get("courses"),
                    "method": cfg.get("method"), "roster": cfg.get("roster"),
                    "done": [x["step"] for x in steps if x["done"]],
                    "students": {k: v["tte"]["mean_m"] for k, v in dep.items() if v.get("role") == "student"},
                    "managed": st.is_dir()})
    return out


def cohort_status(s: Settings, cohort: str) -> dict:
    check_cohort(cohort)
    st, data = state_dir(s) / cohort, cohorts_dir(s) / cohort
    live = {}
    for f in sorted(st.glob("live_*.jsonl")) if st.is_dir() else []:
        m = re.match(r"live_(.+)_(histNet|commNet)$", f.stem)
        if not m:
            continue
        pts = []
        for line in f.read_text().splitlines()[-5000:]:
            try:
                d = json.loads(line)
                pts.append([d["epoch"], d["loss"]])
            except (ValueError, KeyError):
                pass
        live.setdefault(m.group(1), {})[m.group(2)] = pts
    disk = {}
    if data.is_dir():
        for sub in ("rollout_data", "observation_data", "roster", "deployment_data", "_archive"):
            disk[sub] = round(_du(data / sub) / 1e9, 2)
    return {"cohort": cohort, "config": _json(st / "config.json", {}), "steps": _steps(st),
            "results": _json(st / "results.json", {}), "live": live, "disk_gb": disk,
            "data_dir": str(data), "exists": data.is_dir()}


def deployment_video(s: Settings, cohort: str, name: str) -> Path | None:
    """sim_<course>_<pilot>_rgb.mp4 written by deploy_roster(mode='visualize')."""
    check_cohort(cohort)
    if not re.match(r"^sim_[A-Za-z0-9_\-]+_rgb\.mp4$", name):
        raise ValueError("invalid video name")
    p = cohorts_dir(s) / cohort / "deployment_data" / name
    return p if p.is_file() else None
