"""Stand-in for figs/semantic_pipeline.py: marks the steps done and (re)writes the lift table's index."""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.environ["E2E_DIR"])
import server_lib
ap = argparse.ArgumentParser()
for f in ("--project-root", "--scene", "--backend", "--teachers", "--feat-width", "--only", "--from", "--stop-after", "--dino-width", "--batch"):
    ap.add_argument(f)
ap.add_argument("--redo", action="append", default=[])
a = ap.parse_args()
print("argv:", json.dumps(sys.argv[1:]), flush=True)
for s in ("preflight", "cameras", "teachers", "lift", "export"):
    print(f"step {s}", flush=True); time.sleep(float(os.environ.get("E2E_STEP_S", "0.3")))
server_lib.write_table(Path(a.project_root), a.scene)
print("Done", flush=True)
