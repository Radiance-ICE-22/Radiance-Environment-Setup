"""Semantic features (docs/SEMANTICS.md, Phase 2): jobs, status/staleness, annotations, and the
query worker — with a fake semantic_pipeline.py and a fake semantic_worker.py (stdlib only) that
speak the real scripts' contracts."""
from __future__ import annotations

import json
import os
import textwrap
import time
from pathlib import Path

import pytest

from galley import semantics as sem

RUN = "2025-01-16_122349"

FAKE_SEM_PIPELINE = textwrap.dedent('''
    import argparse, json, os, sys
    from pathlib import Path
    ap = argparse.ArgumentParser()
    for f in ("--project-root", "--scene", "--backend", "--teachers", "--feat-width", "--only", "--from"):
        ap.add_argument(f)
    a, rest = ap.parse_known_args()
    print("argv:", json.dumps(sys.argv[1:]), flush=True)
    root = Path(a.project_root)
    st = root / ".semantic_pipeline_state" / a.scene / "%s"
    st.mkdir(parents=True, exist_ok=True)
    for s in ("cameras", "teachers", "lift", "export"):
        (st / f"{s}.done").write_text("fp\\n2026-10-05T12:00:00\\n")
    (st / "config.json").write_text(json.dumps({"backend": a.backend, "feat_width": a.feat_width}))
    print("Done", flush=True)
''' % RUN)

FAKE_WORKER = textwrap.dedent('''
    import argparse, base64, json, os, sys, time
    ap = argparse.ArgumentParser(); ap.add_argument("--project-root"); ap.add_argument("--idle", type=float)
    a = ap.parse_args()
    mode = os.environ.get("FAKE_WORKER_MODE", "")
    print("fake worker starting", file=sys.stderr, flush=True)
    if mode == "bad_start":
        print("hello", flush=True); sys.exit(0)
    print(json.dumps({"ready": True, "pid": os.getpid()}), flush=True)
    last = time.time()
    for line in sys.stdin:
        req = json.loads(line)
        op, rid = req.get("op"), req.get("id")
        if op == "query" and req.get("text") == "crash":
            sys.exit(3)
        if op == "query" and req.get("text") == "slow":
            time.sleep(5)
        if op == "ping":
            out = {"id": rid, "ok": True, "pid": os.getpid(), "env_cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}
        elif op == "query":
            if req.get("scene") == "nosemantics":
                out = {"id": rid, "ok": False, "code": "no_table", "error": "no lift table"}
            else:
                res = {"text": req["text"], "candidates": [{"rank": 1, "centroid": [1, 2, 3]}],
                       "settings_seen": req.get("settings")}
                out = {"id": rid, "ok": True, "ms": 5, "result": res, "table": {"key": "k", "n": 4},
                       "stale": False}
                if req.get("relevancy"):
                    out["relevancy_b64"] = base64.b64encode(bytes([0, 64, 128, 255])).decode()
        elif op == "labels":
            out = {"id": rid, "ok": True, "scores": [[l, 0.1] for l in req["labels"]], "seen": True,
                   "position_splat": [0, 0, 0], "table": {}, "stale": False}
        else:
            out = {"id": rid, "ok": False, "code": "bad_request", "error": "unknown op"}
        print(json.dumps(out), flush=True)
''')


@pytest.fixture
def sem_project(project):
    (project / "semantic_pipeline.py").write_text(FAKE_SEM_PIPELINE)
    (project / "semantic_worker.py").write_text(FAKE_WORKER)
    run = project / "SousVide" / "gsplats" / "workspace" / "outputs" / "backroom" / "splatfacto" / RUN
    (run / "nerfstudio_models" / "step-000029999.ckpt").write_bytes(b"ck")
    (project / "SousVide" / "gsplats" / "workspace" / "backroom").mkdir(parents=True, exist_ok=True)
    return project


def write_table(project: Path, key: str, run: str = RUN, backend: str = "lift", n: int = 4):
    d = project / "SousVide" / "gsplats" / "workspace" / "backroom" / "semantics" / run / backend
    d.mkdir(parents=True, exist_ok=True)
    (d / "pca_rgb.u8").write_bytes(bytes(i % 256 for i in range(3 * n)))
    (d / "index.json").write_text(json.dumps({"n": n, "key": key, "teacher_tag": "t1", "order_sha": "o1",
                                              "created": "2026-10-05T11:32:39",
                                              "metrics": {"seen_rows": n - 1, "lift": {"seconds": 181.0, "passes": 7830}}}))
    return d


def current_key(project: Path) -> str:
    ck = (project / "SousVide" / "gsplats" / "workspace" / "outputs" / "backroom" / "splatfacto" / RUN
          / "nerfstudio_models" / "step-000029999.ckpt")
    return f"{RUN}-{ck.stem}-{int(ck.stat().st_mtime)}"


def _wait(client, jid, t=20):
    for _ in range(int(t / 0.1)):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


# ── command lines ──────────────────────────────────────────────────────────────
def test_argv_and_profile_default(settings):
    r = sem.SemanticRun(scene="backroom", only="lift", redo=["lift"])
    argv = sem.build_argv(settings, r)
    assert argv[1].endswith("semantic_pipeline.py") and argv[argv.index("--scene") + 1] == "backroom"
    assert argv[argv.index("--only") + 1] == "lift" and "--feat-width" not in argv
    settings.defaults["semantic_feat_width"] = 480
    argv = sem.build_argv(settings, sem.SemanticRun(scene="backroom", teachers=["clip"]))
    assert argv[argv.index("--feat-width") + 1] == "480" and argv[argv.index("--teachers") + 1] == "clip"
    assert sem.build_argv(settings, sem.SemanticRun(scene="backroom", feat_width=960))[-1] == "960"


@pytest.mark.parametrize("bad", [{"scene": "../x"}, {"scene": "b", "teachers": ["dino"]},
                                 {"scene": "b", "only": "lift", "from_step": "teachers"},
                                 {"scene": "b", "from_step": "export", "stop_after": "teachers"},
                                 {"scene": "b", "dino_width": 900}, {"scene": "b", "backend": "nerf"}])
def test_run_validation(bad):
    with pytest.raises(ValueError):
        sem.SemanticRun(**bad)


# ── status and staleness ───────────────────────────────────────────────────────
def test_status_fresh_stale_and_other_runs(settings, sem_project):
    st = sem.status(settings, "backroom")
    assert st["run"] == RUN and st["tables"] == [] and st["ready"] == [] and not any(x["done"] for x in st["steps"])
    write_table(sem_project, current_key(sem_project))
    write_table(sem_project, "old-key", run="2024-01-01_000000")
    st = sem.status(settings, "backroom")
    assert [t["run"] for t in st["tables"]] == [RUN, "2024-01-01_000000"]      # active run first
    assert st["tables"][0]["stale"] is False and st["tables"][1]["stale"] is True
    assert st["ready"] == ["lift"] and st["tables"][0]["lift"]["passes"] == 7830
    ck = next((sem_project / "SousVide").rglob("step-000029999.ckpt"))
    os.utime(ck, (time.time() + 100, time.time() + 100))                         # retrained/promoted
    st = sem.status(settings, "backroom")
    assert st["tables"][0]["stale"] is True and st["ready"] == []


def test_active_key_matches_splat_cache_key(settings, sem_project):
    """The table key and CourseTools.splat's cache key must be the same string."""
    run, key = sem.active_key(settings, "backroom")
    from galley.course import CourseTools
    from galley.pipeline import trained_models
    models = trained_models(settings, "backroom")
    ck = settings.repo / models[0]["checkpoint"]
    assert key == f"{models[0]['run']}-{ck.stem}-{int(ck.stat().st_mtime)}" and run == RUN
    assert CourseTools(settings).splat_cache("backroom").name == "backroom"
    assert sem.active_key(settings, "backroom1") == (None, None)                 # no checkpoint


# ── API: jobs, status, pca, annotations ────────────────────────────────────────
def test_job_through_queue_and_archive_waits(client, sem_project):
    r = client.post("/api/jobs/semantics", json={"scene": "backroom", "feat_width": 480})
    assert r.status_code == 201, r.text
    j = _wait(client, r.json()["id"])
    assert j["status"] == "succeeded" and j["kind"] == "semantics" and j["scene"] == "backroom"
    log = " ".join(x["line"] for x in client.get(f"/api/jobs/{j['id']}/log").json())
    assert '"--feat-width", "480"' in log and '"--backend", "lift"' in log
    st = client.get("/api/scenes/backroom/semantics").json()
    assert [x["done"] for x in st["steps"]] == [False, True, True, True, True] and st["config"]["feat_width"] == "480"
    assert st["busy"] is False and st["script"] is True
    assert client.post("/api/jobs/semantics", json={"scene": "backroom1"}).status_code == 400   # no model
    assert client.post("/api/jobs/semantics", json={"scene": "../etc"}).status_code == 422


def test_archive_refused_while_semantic_job_runs(client, sem_project, settings):
    (sem_project / "semantic_pipeline.py").write_text("import time; time.sleep(3)\n")
    jid = client.post("/api/jobs/semantics", json={"scene": "backroom"}).json()["id"]
    for _ in range(50):
        if client.get(f"/api/jobs/{jid}").json()["status"] == "running":
            break
        time.sleep(0.05)
    assert client.post(f"/api/scenes/backroom/models/{RUN}/archive").status_code == 409
    client.post(f"/api/jobs/{jid}/cancel")
    _wait(client, jid)


def test_pca_bytes_and_stale_header(client, sem_project):
    assert client.get("/api/scenes/backroom/semantics/lift/pca").status_code == 404
    write_table(sem_project, current_key(sem_project), n=2000)
    r = client.get("/api/scenes/backroom/semantics/lift/pca", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200 and len(r.content) == 6000 and r.headers["x-table-stale"] == "0"
    write_table(sem_project, "old", n=2)
    r = client.get("/api/scenes/backroom/semantics/lift/pca")
    assert r.headers["x-table-stale"] == "1"
    assert client.get("/api/scenes/backroom/semantics/nerf/pca").status_code == 400


def test_annotations_round_trip(client, sem_project):
    assert client.get("/api/scenes/backroom/semantics/queries").json()["queries"] == []
    body = {"version": 1, "frame": "course (x, -y, -z), z down", "queries": [
        {"text": "red tool chest", "position": [-0.366, 2.6, -0.931], "set": "phase1_gate", "note": "easy"},
        {"text": "garden cart", "position": None, "set": "phase1_gate"}]}
    r = client.put("/api/scenes/backroom/semantics/queries", json=body)
    assert r.status_code == 200, r.text
    got = client.get("/api/scenes/backroom/semantics/queries").json()
    assert got["queries"][0]["note"] == "easy" and got["queries"][1]["position"] is None and "updated" in got
    st = client.get("/api/scenes/backroom/semantics").json()
    assert st["queries"] == {"total": 2, "annotated": 1}
    dup = {"queries": [{"text": "a", "position": None}, {"text": "a", "position": None}]}
    assert client.put("/api/scenes/backroom/semantics/queries", json=dup).status_code == 422
    assert client.put("/api/scenes/backroom/semantics/queries",
                      json={"queries": [{"text": "a", "position": [1, 2]}]}).status_code == 422
    assert client.put("/api/scenes/nowhere/semantics/queries", json={"queries": []}).status_code == 400


def test_queries_file_written_by_cli_is_readable(client, sem_project):
    """radiance_semantics.annotations writes queries.json; Galley must read the same file."""
    p = sem_project / "SousVide" / "gsplats" / "workspace" / "backroom" / "semantics" / "queries.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"version": 1, "frame": "course (x, -y, -z), z down", "queries": [
        {"text": "shop vacuum", "position": [-2.09, -5.678, -0.349], "set": "phase1_gate",
         "note": "x", "updated": "2026-10-05T12:00:00"}]}))
    q = client.get("/api/scenes/backroom/semantics/queries").json()["queries"][0]
    assert q["text"] == "shop vacuum" and q["updated"]
    r = client.put("/api/scenes/backroom/semantics/queries", json=client.get("/api/scenes/backroom/semantics/queries").json())
    assert r.status_code == 200                       # a file from the CLI re-saves unchanged


# ── API: the query worker ──────────────────────────────────────────────────────
def test_query_relevancy_labels_and_worker_env(client, sem_project, settings):
    r = client.post("/api/scenes/backroom/semantics/query",
                    json={"text": "red tool chest", "rel_alpha": 0.5, "standoff": 1.2})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["result"]["candidates"][0]["centroid"] == [1, 2, 3] and b["stale"] is False
    assert b["result"]["settings_seen"] == {"rel_alpha": 0.5, "standoff": 1.2}       # only what was set
    rel = client.get(f"/api/scenes/backroom/semantics/relevancy/{b['relevancy_id']}")
    assert rel.status_code == 200 and rel.content == bytes([0, 64, 128, 255])
    assert client.get(f"/api/scenes/backroom1/semantics/relevancy/{b['relevancy_id']}").status_code == 404
    lab = client.post("/api/scenes/backroom/semantics/labels", json={"index": 3, "labels": ["chair", "table"]}).json()
    assert [x[0] for x in lab["scores"]] == ["chair", "table"]
    w = client.get("/api/semantics/worker").json()
    assert w["running"] and w["pid"]
    worker = client.app.state.semworker
    assert worker.request("ping")["env_cuda"] == ""                                # never the GPU
    assert "fake worker starting" in (settings.data_dir / "semantic_worker.log").read_text()
    r = client.post("/api/scenes/backroom/semantics/query", json={"text": "x", "relevancy": False})
    assert r.json()["relevancy_id"] is None


def test_worker_errors_map_to_http(client, sem_project):
    r = client.post("/api/scenes/nosemantics/semantics/query", json={"text": "x"})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "no_table"
    # an invalid scene name reaches the route (no "..": the client normalises it away, and with a
    # built frontend the normalised path lands on the static mount → 405)
    assert client.post("/api/scenes/bad.name/semantics/query", json={"text": "x"}).status_code == 400
    assert client.post("/api/scenes/backroom/semantics/query", json={"text": ""}).status_code == 422
    assert client.post("/api/scenes/backroom/semantics/query", json={"text": "x", "top": 99}).status_code == 422


def test_crash_costs_one_request_then_restarts(client, sem_project):
    worker = client.app.state.semworker
    assert client.post("/api/scenes/backroom/semantics/query", json={"text": "a"}).status_code == 200
    pid = worker.status()["pid"]
    r = client.post("/api/scenes/backroom/semantics/query", json={"text": "crash"})
    assert r.status_code == 503 and "exited" in r.json()["detail"]
    r = client.post("/api/scenes/backroom/semantics/query", json={"text": "b"})
    assert r.status_code == 200 and worker.status()["pid"] != pid and worker.status()["restarts"] >= 1


def test_stop_and_idle_restart(client, sem_project):
    worker = client.app.state.semworker
    client.post("/api/scenes/backroom/semantics/query", json={"text": "a"})
    assert client.post("/api/semantics/worker/stop").json() == {"stopped": True}
    assert not worker.alive() and client.post("/api/semantics/worker/stop").json() == {"stopped": False}
    assert client.post("/api/scenes/backroom/semantics/query", json={"text": "again"}).status_code == 200
    os.killpg(worker._proc.pid, 15)                    # as if it idled out between requests
    time.sleep(0.3)
    assert client.post("/api/scenes/backroom/semantics/query", json={"text": "after idle"}).status_code == 200


def test_timeout_kills_and_recovers(settings, sem_project):
    from galley.semworker import SemWorker, WorkerError
    w = SemWorker(settings, idle_s=60)
    with pytest.raises(WorkerError, match="did not answer"):
        w.request("query", timeout=0.5, scene="backroom", text="slow")
    assert not w.alive()
    assert w.request("ping")["ok"]
    w.stop()


def test_bad_start_and_missing_script(settings, sem_project, monkeypatch):
    from galley.semworker import SemWorker, WorkerError
    monkeypatch.setenv("FAKE_WORKER_MODE", "bad_start")
    with pytest.raises(WorkerError):
        SemWorker(settings).request("ping")
    monkeypatch.delenv("FAKE_WORKER_MODE")
    (sem_project / "semantic_worker.py").unlink()
    with pytest.raises(WorkerError, match="not found"):
        SemWorker(settings).request("ping")


def test_relevancy_cache_evicts():
    c = sem.RelevancyCache(size=2)
    a, b, d = c.put("s", b"a"), c.put("s", b"b"), c.put("s", b"d")
    assert c.get("s", a) is None and c.get("s", b) == b"b" and c.get("s", d) == b"d"
    assert c.get("other", d) is None


def test_course_with_extended_semantic_goal_round_trips(client):
    course = {"waypoints": {"Nco": 6, "keyframes": {
        "start": {"t": 0.0, "fo": [[0.0], [0.0], [-1.0], [0.0]]},
        "end": {"t": 4.0, "fo": [[1.0], [0.0], [-1.0], [0.0]]}}},
        "semantic_goal": {"label": "red tool chest", "position": [-0.366, 2.6, -0.931], "query": "red tool chest",
                          "backend": "lift", "score": 249.04, "extent": {"lo": [-1.0, 2.5, -1.5], "hi": [0.2, 3.1, -0.7]},
                          "approach": [-0.63, 1.111, -1.109]}}
    r = client.put("/api/configs/courses/sem_test", json=course)
    assert r.status_code == 200, r.text
    assert client.get("/api/configs/courses/sem_test").json() == course
