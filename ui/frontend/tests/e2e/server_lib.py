"""Project layout for the end-to-end run (the backend tests' conftest layout + a semantic table)."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["E2E_DIR"])
import synth

RUN = "2025-01-16_122349"
FAKE_PIPELINE = '''
import argparse, json, os, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
for f in ("--project-root", "--scene", "--from", "--only", "--stop-after", "--course", "--pilot", "--frame", "--method"):
    ap.add_argument(f)
ap.add_argument("--redo", action="append", default=[]); ap.add_argument("--allow-outside", action="store_true")
a, rest = ap.parse_known_args()
print("argv:", json.dumps(sys.argv[1:]), flush=True)
for s in ("course", "simulate", "validate", "record"):
    print("step", s, flush=True); time.sleep(0.2)
st = Path(a.project_root) / ".figs_pipeline_state" / a.scene
st.mkdir(parents=True, exist_ok=True)
(st / "results.json").write_text(json.dumps({"course": {"name": a.course, "inside": True},
    "sim": {"frames": 240, "hz": 20, "duration_s": 12, "track_err_mean_m": 0.02, "track_err_max_m": 0.05, "pixel_std": 40, "dark_frames": 0}}))
print("Done", flush=True)
'''


def key_of(root: Path, scene: str) -> str:
    ck = root / "SousVide/gsplats/workspace/outputs" / scene / "splatfacto" / RUN / "nerfstudio_models/step-000029999.ckpt"
    return f"{RUN}-{ck.stem}-{int(ck.stat().st_mtime)}"


def write_table(root: Path, scene: str):
    n = len(synth.scene())
    d = root / "SousVide/gsplats/workspace" / scene / "semantics" / RUN / "lift"
    d.mkdir(parents=True, exist_ok=True)
    rgb = bytearray()
    for (x, y, z), s, c in synth.scene():      # "PCA": the colour, rotated, so objects stand out
        rgb += bytes([c[1], c[2], c[0]])
    (d / "pca_rgb.u8").write_bytes(bytes(rgb))
    (d / "index.json").write_text(json.dumps({"n": n, "key": key_of(root, scene), "backend": "lift", "teacher_tag": "clip+dino e2e",
        "order_sha": "e2e", "created": "2026-10-05T12:00:00", "metrics": {"seen_rows": n - 100, "lift": {"seconds": 181, "passes": 29, "peak_vram_mib": 2400, "render_width": 960, "views": 270}}}))
    st = root / ".semantic_pipeline_state" / scene / RUN
    st.mkdir(parents=True, exist_ok=True)
    for s in ("preflight", "cameras", "teachers", "lift", "export"):
        (st / f"{s}.done").write_text("fp\n2026-10-05T12:00:00\n")


def make_project(root: Path, with_table=True):
    e2e = Path(os.environ["E2E_DIR"])
    repo = root / "SousVide"
    for fam in ("captures", "courses", "pilots", "frames", "methods", "nnio"):
        (repo / "configs" / fam).mkdir(parents=True, exist_ok=True)
    (repo / "configs/pilots/Viper.json").write_text(json.dumps({"plan": {}, "track": {}}))
    (repo / "configs/frames/carl.json").write_text(json.dumps({"mass": 1.0}))
    (root / "figs_env.sh").write_text('cd "$(dirname "${BASH_SOURCE[0]}")/SousVide"\n')
    (root / "fake_pipeline.py").write_text(FAKE_PIPELINE)
    (root / "course_tools.py").write_text((e2e / "fake_course_tools.py").read_text())
    (root / "semantic_worker.py").write_text((e2e / "fake_semantic_worker.py").read_text())
    (root / "semantic_pipeline.py").write_text((e2e / "fake_semantic_pipeline.py").read_text())
    (root / "video_captures").mkdir(exist_ok=True)
    out = repo / "gsplats/workspace/outputs/backroom/splatfacto" / RUN
    (out / "nerfstudio_models").mkdir(parents=True, exist_ok=True)
    (out / "config.yml").write_text("x")
    (out / "nerfstudio_models/step-000029999.ckpt").write_bytes(b"ck")
    ws = repo / "gsplats/workspace/backroom"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "transforms.json").write_text("{}"); (ws / "sparse_pc.ply").write_text("ply")
    if with_table:
        write_table(root, "backroom")
    q = ws / "semantics/queries.json"; q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text(json.dumps({"version": 1, "frame": "course (x, -y, -z), z down", "queries": [
        {"text": "red box", "position": [1.5, -1.0, -0.4], "set": "gate"},
        {"text": "green ball", "position": [-1.5, 1.2, -0.3], "set": "gate"},
        {"text": "blue chair", "position": None, "set": "extra"}]}))
