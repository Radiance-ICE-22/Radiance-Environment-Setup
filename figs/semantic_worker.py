#!/usr/bin/env python3
"""
semantic_worker.py — long-lived semantic query server for Galley, one JSON line in, one out.

Galley (its own venv, no torch) starts this inside kitchen through figs_env.sh with
CUDA_VISIBLE_DEVICES="" — queries run on the CPU and never compete with the GPU job queue — and
keeps it alive between queries so the CLIP text tower (~5 s to load) and the memory-mapped tables
load once. It stops itself after --idle seconds without a request.

Protocol (stdin → stdout, UTF-8, one JSON object per line; every reply echoes "id"):
    → {"id": 1, "op": "ping"}
    ← {"id": 1, "ok": true, "pid": 1234, "loaded": [...]}
    → {"id": 2, "op": "query", "scene": "backroom", "backend": "lift", "text": "red tool chest",
       "settings": {"threshold": 0.55, "rel_alpha": 0.5, "standoff": 1.0, ...}, "relevancy": true}
    ← {"id": 2, "ok": true, "result": {...run_query output...}, "table": {"key": ..., "n": ...},
       "stale": false, "relevancy_b64": "<n bytes, round(rel·255), .splat order>"}
    → {"id": 3, "op": "labels", "scene": "backroom", "backend": "lift", "index": 1234,
       "labels": ["chair", "table", ...]}
    ← {"id": 3, "ok": true, "scores": [[label, cosine], ...] (best first), "seen": true}
    → {"id": 4, "op": "unload"}   drop cached tables (the next request reloads)
Errors: {"id": n, "ok": false, "code": "bad_request" | "no_table" | "no_scene" | "error", "error": "..."}

The first line written is {"ready": true, "pid": ...}. Anything libraries print goes to stderr:
stdout (fd 1) is redirected there at start and replies use a private duplicate of it.
"""

import argparse
import base64
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

FIELDS = ("threshold", "rel_alpha", "peak_k", "min_opacity", "voxel", "min_gaussians", "top", "standoff",
          "margin", "body_radius", "min_gap", "clearance_k", "ambiguous_margin", "negatives")


class BadRequest(ValueError):
    code = "bad_request"


class NoTable(ValueError):
    code = "no_table"


class NoScene(ValueError):
    code = "no_scene"


class Worker:
    """Request handler; holds the text encoder and one cached table per (scene, backend)."""

    def __init__(self, project_root, encoder=None):
        self.root = Path(project_root)
        self._encoder = encoder
        self._tables = {}             # (scene, backend) → (index mtime, Table, run key at load)
        self._scenes = {}             # scene → (transforms mtime, Scene)

    @property
    def encoder(self):
        if self._encoder is None:
            from radiance_semantics.query import TextEncoder
            self._encoder = TextEncoder("cpu")
        return self._encoder

    def _run(self, scene):
        from radiance_semantics.paths import SemanticsError, find_scene_run
        try:
            return find_scene_run(self.root, scene)
        except SemanticsError as e:
            raise NoScene(str(e))

    def table(self, scene, backend):
        from radiance_semantics.store import read_table
        run = self._run(scene)
        d = run.backend_dir(backend) if backend in ("lift", "fmgs") else None
        if d is None:
            raise BadRequest(f"unknown backend {backend!r}")
        idx = d / "index.json"
        if not idx.exists():
            raise NoTable(f"no {backend} table for {scene} yet ({d}): run the semantic pipeline")
        m = idx.stat().st_mtime
        hit = self._tables.get((scene, backend))
        if hit is None or hit[0] != m:
            self._tables[(scene, backend)] = (m, read_table(d))
        t = self._tables[(scene, backend)][1]
        return t, run

    def scene(self, scene):
        from semantic_query import load_scene
        tf = self.root / "SousVide" / "gsplats" / "workspace" / scene / "transforms.json"
        if not tf.exists():
            raise NoScene(f"{scene} has no transforms.json")
        m = tf.stat().st_mtime
        hit = self._scenes.get(scene)
        if hit is None or hit[0] != m:
            self._scenes[scene] = (m, load_scene(self.root, scene))
        return self._scenes[scene][1]

    def handle(self, req):
        op = req.get("op")
        if op == "ping":
            return {"pid": os.getpid(), "loaded": [f"{s}/{b}" for s, b in self._tables]}
        if op == "unload":
            self._tables.clear()
            self._scenes.clear()
            return {}
        if op == "query":
            return self.query(req)
        if op == "labels":
            return self.labels(req)
        raise BadRequest(f"unknown op {op!r}")

    def _table_info(self, t, run):
        return {"key": t.index.get("key"), "n": t.n, "backend": t.index.get("backend"),
                "teacher_tag": t.index.get("teacher_tag"), "order_sha": t.index.get("order_sha"),
                "run": t.index.get("run")}, t.index.get("key") != run.key

    def query(self, req):
        from radiance_semantics.query import Settings, run_query
        text = str(req.get("text", "")).strip()
        if not text or len(text) > 200:
            raise BadRequest("text: 1–200 characters")
        t, run = self.table(req.get("scene", ""), req.get("backend", "lift"))
        s = Settings()
        for k, v in (req.get("settings") or {}).items():
            if k not in FIELDS:
                raise BadRequest(f"unknown setting {k!r}")
            if v is not None:
                setattr(s, k, list(v) if k == "negatives" else type(getattr(s, k))(v))
        res, rel = run_query(t, self.scene(req["scene"]), text, self.encoder, s, return_relevancy=True)
        info, stale = self._table_info(t, run)
        out = {"result": res, "table": info, "stale": stale}
        if req.get("relevancy"):
            out["relevancy_b64"] = base64.b64encode(np.round(np.clip(rel, 0, 1) * 255).astype(np.uint8).tobytes()).decode()
        return out

    def labels(self, req):
        t, run = self.table(req.get("scene", ""), req.get("backend", "lift"))
        labels = [str(x).strip() for x in (req.get("labels") or []) if str(x).strip()]
        if not labels or len(labels) > 64:
            raise BadRequest("labels: 1–64 strings")
        try:
            i = int(req.get("index"))
        except (TypeError, ValueError):
            raise BadRequest("index: an integer row of the table (.splat record)")
        if not 0 <= i < t.n:
            raise BadRequest(f"index out of range (0..{t.n - 1})")
        f = np.asarray(t.clip[i], dtype=np.float32)
        E = self.encoder.encode(labels)
        sc = (E @ f).astype(float)
        order = np.argsort(-sc)
        info, stale = self._table_info(t, run)
        return {"scores": [[labels[k], round(float(sc[k]), 4)] for k in order], "seen": bool(t.weight[i] > 0),
                "position_splat": [round(float(v), 4) for v in t.geom[i, :3]], "table": info, "stale": stale}


def serve(worker, inp, out, idle=600.0, log=sys.stderr):
    """Read requests from `inp`, write replies to `out` (text streams). Returns on EOF or idle timeout."""
    lock = threading.Lock()
    last = [time.time()]
    stop = threading.Event()

    def write(obj):
        with lock:
            out.write(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n")
            out.flush()

    def watchdog():
        while not stop.wait(1.0):
            if idle and time.time() - last[0] > idle:
                print(f"semantic_worker: idle for {idle:.0f} s, exiting", file=log, flush=True)
                os._exit(0)

    if idle:
        threading.Thread(target=watchdog, daemon=True).start()
    write({"ready": True, "pid": os.getpid()})
    for line in inp:
        last[0] = time.time()
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            t0 = time.time()
            body = worker.handle(req)
            write({"id": rid, "ok": True, "ms": round((time.time() - t0) * 1000), **body})
        except (BadRequest, NoTable, NoScene) as e:
            write({"id": rid, "ok": False, "code": e.code, "error": str(e)})
        except json.JSONDecodeError as e:
            write({"id": rid, "ok": False, "code": "bad_request", "error": f"not JSON: {e}"})
        except Exception as e:                    # keep serving; the traceback goes to the log
            traceback.print_exc(file=log)
            write({"id": rid, "ok": False, "code": "error", "error": f"{type(e).__name__}: {e}"})
        last[0] = time.time()
    stop.set()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--idle", type=float, default=600.0, help="exit after this many idle seconds (0 = never)")
    a = ap.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    reply = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)                                  # stray prints from libraries → stderr
    sys.stdout = sys.stderr
    serve(Worker(a.project_root), sys.stdin, reply, a.idle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
