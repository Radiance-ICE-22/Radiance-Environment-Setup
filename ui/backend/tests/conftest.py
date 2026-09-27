"""Fixtures: a throwaway PROJECT_ROOT with a fake figs_env.sh and a fake figs_pipeline.py.

Set GALLEY_TEST_CONFIGS to a SousVide configs/ directory (and optionally
GALLEY_TEST_OVERLAY to FYP-Radiance's overlay configs/) to round-trip the real files.
"""
from __future__ import annotations

import os
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

FAKE_PIPELINE = textwrap.dedent('''
    import argparse, json, os, signal, sys, time
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root"); ap.add_argument("--scene")
    ap.add_argument("--from", dest="from_step"); ap.add_argument("--only"); ap.add_argument("--stop-after")
    ap.add_argument("--redo", action="append", default=[]); ap.add_argument("--course")
    ap.add_argument("--allow-outside", action="store_true")
    a, rest = ap.parse_known_args()
    print("env marker:", os.environ.get("FAKE_ENV_SOURCED"))
    print("argv:", json.dumps(sys.argv[1:]))
    if a.scene == "fails":
        print("step 'course' failed", flush=True); sys.exit(1)
    state = Path(a.project_root) / ".figs_pipeline_state" / a.scene
    state.mkdir(parents=True, exist_ok=True)
    for i in range(int(os.environ.get("FAKE_STEPS", "3"))):
        print(f"\\x1b[32m✔\\x1b[0m step {i}", flush=True)
        print(f"  50%|#####     | {i}/3", end="\\r", flush=True)
        time.sleep(float(os.environ.get("FAKE_SLEEP", "0.05")))
    (state / "bounds.done").write_text("abc123\\n2026-09-27T13:12:24\\n")
    (state / "results.json").write_text(json.dumps({"scene": a.scene, "ok": True}))
    print("Done")
''')


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "figs_validation"
    repo = root / "SousVide"
    (repo / "configs").mkdir(parents=True)
    src = os.environ.get("GALLEY_TEST_CONFIGS")
    if src:
        shutil.copytree(src, repo / "configs", dirs_exist_ok=True)
    extra = os.environ.get("GALLEY_TEST_OVERLAY")
    if extra:
        shutil.copytree(extra, repo / "configs", dirs_exist_ok=True)
    for fam in ("captures", "courses", "pilots", "frames", "methods", "nnio"):
        (repo / "configs" / fam).mkdir(exist_ok=True)
    (root / "figs_env.sh").write_text('export FAKE_ENV_SOURCED=yes\ncd "$(dirname "${BASH_SOURCE[0]}")/SousVide"\n')
    (root / "fake_pipeline.py").write_text(FAKE_PIPELINE)
    (root / "video_captures").mkdir()
    (root / "video_captures" / "lab3.mp4").write_bytes(b"\0" * 10)
    out = repo / "gsplats" / "workspace" / "outputs"
    (out / "backroom" / "splatfacto" / "2025-01-16_122349" / "nerfstudio_models").mkdir(parents=True)
    (out / "backroom" / "splatfacto" / "2025-01-16_122349" / "config.yml").write_text("x")
    (out / "backroom1" / "splatfacto").mkdir(parents=True)
    return root


@pytest.fixture
def settings(project: Path, tmp_path: Path):
    from galley.settings import Settings
    overlay = tmp_path / "overlay"
    (overlay / "configs").mkdir(parents=True)
    return Settings(project_root=project, pipeline=project / "fake_pipeline.py", overlay=overlay,
                    data_dir=tmp_path / "data", video_dir=project / "video_captures",
                    python=sys.executable)


@pytest.fixture
def client(settings):
    from fastapi.testclient import TestClient
    from galley.app import create_app
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(settings)) as c:
        yield c
