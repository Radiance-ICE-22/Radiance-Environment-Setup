"""Machine profile: where the pipeline lives on this host and what the hardware allows.

Everything host-specific comes from one TOML file so the same code runs on dummy
(RTX 3050 Ti, 4 GB), the lab machine (RTX 2080, 8 GB) and the RTX 5060 Ti PC.
Lookup order: $GALLEY_MACHINE, then ui/machine.toml, then ui/machines/<name>.toml whose
<name> starts this machine's hostname (intellisense08.toml on intellisense08-EWISPro9900G,
dummy.toml on dummy).
"""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore

UI_DIR = Path(__file__).resolve().parents[2]  # .../ui


def _p(v: str | os.PathLike) -> Path:
    return Path(os.path.expandvars(str(v))).expanduser().resolve()


@dataclass
class Settings:
    project_root: Path            # contains figs_env.sh and SousVide/
    pipeline: Path                # figs_pipeline.py
    overlay: Path | None          # FYP-Radiance/figs/sousvide_overlay (mirrors saved configs)
    data_dir: Path                # galley.db, uploads
    video_dir: Path               # raw capture staging (video_captures/)
    host: str = "0.0.0.0"
    port: int = 8800
    token: str | None = None      # optional shared secret; Phase 5 replaces with a login
    gpu_name: str = ""
    vram_mib: int = 0
    defaults: dict = field(default_factory=dict)
    python: str = "python"        # interpreter inside the kitchen env (after figs_env.sh)
    source: Path | None = None    # the profile file this came from

    @property
    def env_script(self) -> Path:
        return self.project_root / "figs_env.sh"

    @property
    def repo(self) -> Path:
        return self.project_root / "SousVide"

    @property
    def configs_dir(self) -> Path:
        return self.repo / "configs"

    @property
    def state_dir(self) -> Path:
        return self.project_root / ".figs_pipeline_state"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "galley.db"


class NoMachineProfile(FileNotFoundError):
    pass


def find_profile(hostname: str | None = None) -> Path:
    env = os.environ.get("GALLEY_MACHINE")
    if env:
        p = _p(env)
        if not p.is_file():
            raise NoMachineProfile(f"GALLEY_MACHINE={env}: no such file")
        return p
    if (UI_DIR / "machine.toml").is_file():
        return UI_DIR / "machine.toml"
    host = (hostname or socket.gethostname()).lower()
    matches = sorted((m for m in (UI_DIR / "machines").glob("*.toml") if host.startswith(m.stem.lower())),
                     key=lambda m: len(m.stem), reverse=True)          # longest (most specific) first
    if matches:
        return matches[0]
    have = ", ".join(sorted(m.name for m in (UI_DIR / "machines").glob("*.toml"))) or "none"
    raise NoMachineProfile(
        f"no machine profile for host '{host}'. Set GALLEY_MACHINE=ui/machines/<file>.toml "
        f"(available: {have}), or add ui/machines/<hostname prefix>.toml")


def load(path: str | os.PathLike | None = None) -> Settings:
    path = Path(path) if path else find_profile()
    raw = tomllib.loads(path.read_text())
    p, srv, gpu = raw.get("paths", {}), raw.get("server", {}), raw.get("gpu", {})
    project_root = _p(p["project_root"])
    s = Settings(
        project_root=project_root,
        pipeline=_p(p["pipeline"]),
        overlay=_p(p["overlay"]) if p.get("overlay") else None,
        data_dir=_p(p.get("data_dir", "~/.local/share/galley")),
        video_dir=_p(p.get("video_dir", project_root / "video_captures")),
        host=srv.get("host", "0.0.0.0"),
        port=int(srv.get("port", 8800)),
        token=os.environ.get("GALLEY_TOKEN") or srv.get("token") or None,
        gpu_name=gpu.get("name", ""),
        vram_mib=int(gpu.get("vram_mib", 0)),
        defaults=raw.get("defaults", {}),
        source=path,
    )
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
