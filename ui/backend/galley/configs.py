"""SousVide config families: read, validate, write.

The files stay exactly where upstream expects them (SousVide/configs/<family>/<name>.json)
so notebooks and figs_pipeline.py keep working. Models validate structure only; they
allow unknown keys, and a write refuses anything that would not round-trip unchanged.

Saved captures, courses and pilots are also mirrored into FYP-Radiance's
figs/sousvide_overlay/, which apply_overlay.sh restores after a re-clone.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")
FAMILIES = ("captures", "courses", "pilots", "frames", "methods", "nnio")
MIRRORED = ("captures", "courses", "pilots")

Num = Optional[float]


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


# ── courses ──────────────────────────────────────────────────────────────────
class Keyframe(_Open):
    t: float = Field(ge=0)
    fo: list[list[Num]]

    @field_validator("fo")
    @classmethod
    def _fo(cls, v):
        if len(v) != 4:
            raise ValueError("fo must have 4 rows: x, y, z, yaw")
        for i, row in enumerate(v):
            if not 1 <= len(row) <= 5:
                raise ValueError(f"fo row {i} must have 1-5 derivative columns")
            if any(x is not None and not math.isfinite(x) for x in row):
                raise ValueError(f"fo row {i} has a non-finite value")
        return v


class Waypoints(_Open):
    Nco: int = Field(ge=1, le=12)
    keyframes: dict[str, Keyframe]

    @model_validator(mode="after")
    def _times(self):
        if len(self.keyframes) < 2:
            raise ValueError("a course needs at least 2 keyframes")
        ts = [k.t for k in self.keyframes.values()]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError("keyframe times must strictly increase in file order")
        first, last = list(self.keyframes.values())[0], list(self.keyframes.values())[-1]
        for name, k in (("first", first), ("last", last)):
            if any(row[0] is None for row in k.fo):
                raise ValueError(f"the {name} keyframe must fix x, y, z and yaw")
        return self


class Course(_Open):
    waypoints: Waypoints
    forces: Optional[dict[str, Any]] = None


# ── captures ─────────────────────────────────────────────────────────────────
class Camera(_Open):
    model: str
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    intrinsics_matrix: list[list[float]]
    distortion_coefficients: list[float]

    @field_validator("intrinsics_matrix")
    @classmethod
    def _k(cls, v):
        if len(v) != 3 or any(len(r) != 3 for r in v):
            raise ValueError("intrinsics_matrix must be 3x3")
        return v


class Extractor(_Open):
    num_images: int = Field(gt=0, le=5000)
    num_marked: int = Field(ge=0)
    marker_length: float = Field(gt=0, lt=5, description="metres, black square only")
    marker_id: int = Field(ge=0, le=49, description="DICT_4X4_50")

    @model_validator(mode="after")
    def _counts(self):
        if self.num_marked > self.num_images:
            raise ValueError("num_marked cannot exceed num_images")
        return self


class Capture(_Open):
    camera: Optional[Camera] = None
    extractor: Extractor


# ── methods ──────────────────────────────────────────────────────────────────
class Randomization(_Open):
    parameters: list[float]
    initial: list[float]


class Rollout(_Open):
    frequency: float = Field(gt=0)
    noise: dict[str, Any]


class Method(_Open):
    duration: Optional[float] = None
    reps: Optional[int] = None
    rate: Optional[float] = None
    tol_select: Optional[float] = None
    randomization: Randomization
    rollout: Rollout


# ── pilots, frames, nnio: structure varies, keep light checks ────────────────
class Pilot(_Open):
    @model_validator(mode="after")
    def _kind(self):
        extra = self.model_extra or {}
        if not ({"plan", "track"} <= extra.keys() or {"hz", "networks"} <= extra.keys()):
            raise ValueError("a pilot is either an expert (plan + track) or a student (hz + networks)")
        return self


class Frame(_Open):
    mass: float = Field(gt=0)
    motor_thrust_coeff: float
    number_of_rotors: int
    camera: dict[str, Any]


class Nnio(_Open):
    pass


MODELS: dict[str, type[BaseModel]] = {
    "captures": Capture, "courses": Course, "pilots": Pilot,
    "frames": Frame, "methods": Method, "nnio": Nnio,
}


def pilot_kind(data: dict) -> str:
    return "expert" if "plan" in data else "student"


# ── store ────────────────────────────────────────────────────────────────────
class ConfigError(ValueError):
    pass


class ConfigStore:
    def __init__(self, configs_dir: Path, overlay: Path | None = None):
        self.root = configs_dir
        self.overlay = overlay

    def _path(self, family: str, name: str) -> Path:
        if family not in FAMILIES:
            raise ConfigError(f"unknown config family '{family}'")
        if not NAME_RE.match(name):
            raise ConfigError("names use letters, digits, '_' and '-', max 64 characters")
        return self.root / family / f"{name}.json"

    def list(self, family: str) -> list[dict]:
        self._path(family, "x")
        d = self.root / family
        out = []
        for p in sorted(d.glob("*.json")) if d.is_dir() else []:
            item = {"name": p.stem, "modified": p.stat().st_mtime}
            if family == "pilots":
                try:
                    item["kind"] = pilot_kind(json.loads(p.read_text()))
                except (OSError, ValueError):
                    item["kind"] = "unreadable"
            out.append(item)
        return out

    def read(self, family: str, name: str) -> dict:
        p = self._path(family, name)
        if not p.exists():
            raise FileNotFoundError(f"{family}/{name}.json")
        return json.loads(p.read_text())

    @staticmethod
    def validate(family: str, data: dict) -> dict:
        model = MODELS[family].model_validate(data)
        dumped = model.model_dump(mode="json", exclude_unset=True)
        if dumped != data:
            raise ConfigError("config would not round-trip unchanged (type coercion); send exact JSON types")
        return dumped

    def write(self, family: str, name: str, data: dict, overwrite: bool = True) -> dict:
        p = self._path(family, name)
        if p.exists() and not overwrite:
            raise ConfigError(f"{family}/{name} already exists")
        self.validate(family, data)
        text = json.dumps(data, indent=4) + "\n"
        _atomic_write(p, text)
        result = {"path": str(p), "mirrored": None}
        if self.overlay and family in MIRRORED and self.overlay.is_dir():
            m = self.overlay / "configs" / family / f"{name}.json"
            _atomic_write(m, text)
            result["mirrored"] = str(m)
        return result


def _atomic_write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, p)
