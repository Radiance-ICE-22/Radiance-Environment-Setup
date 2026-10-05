"""Galley on :8811 over a throwaway project with the stand-in tools, serving the built frontend.
   E2E_DIR=<this dir> <backend venv>/bin/python server.py <workdir> [--no-table]"""
import os, sys
from pathlib import Path
os.environ.setdefault("E2E_DIR", str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ["E2E_DIR"])
import server_lib
import uvicorn
from galley.app import create_app
from galley.settings import Settings

work = Path(sys.argv[1]); work.mkdir(parents=True, exist_ok=True)
root = work / "figs_validation"
server_lib.make_project(root, with_table="--no-table" not in sys.argv, with_fmgs="--fmgs" in sys.argv)
(work / "overlay/configs").mkdir(parents=True, exist_ok=True)
s = Settings(project_root=root, pipeline=root / "fake_pipeline.py", overlay=work / "overlay", data_dir=work / "data",
             video_dir=root / "video_captures", python=sys.executable, port=8811,
             defaults={"semantic_feat_width": 960, "semantic_fmgs_width": 480, "semantic_worker_idle_s": 600})
s.data_dir.mkdir(parents=True, exist_ok=True)
uvicorn.run(create_app(s), host="127.0.0.1", port=8811, log_level="warning")
