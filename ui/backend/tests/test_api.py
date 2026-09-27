from __future__ import annotations

import json
import os
import time

import pytest


def wait(client, job_id, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still {j['status']}")


def test_health(client):
    r = client.get("/api/health").json()
    assert r["ok"] and r["env_script"] and r["pipeline"]


def test_figs_job_runs_through_env_and_streams(client, project):
    r = client.post("/api/jobs/figs", json={"scene": "backroom", "course": "circuit", "from_step": "course"})
    assert r.status_code == 201, r.text
    j = wait(client, r.json()["id"])
    assert j["status"] == "succeeded" and j["returncode"] == 0
    lines = [l["line"] for l in client.get(f"/api/jobs/{j['id']}/log").json()]
    assert "env marker: yes" in lines                      # figs_env.sh was sourced
    argv = json.loads(next(l for l in lines if l.startswith("argv:"))[5:])
    assert argv[argv.index("--from") + 1] == "course" and "--project-root" in argv
    assert "✔ step 0" in lines                             # ANSI stripped
    assert not any("50%|" in l for l in lines)             # \r progress not persisted
    st = client.get("/api/scenes/backroom").json()
    assert next(s for s in st["steps"] if s["step"] == "bounds")["done"]
    assert st["results"]["ok"] is True and len(st["models"]) == 1


def test_failed_job(client):
    j = wait(client, client.post("/api/jobs/figs", json={"scene": "fails"}).json()["id"])
    assert j["status"] == "failed" and j["returncode"] == 1


def test_queue_is_serial_and_cancel_kills_group(client):
    a = client.post("/api/jobs/selftest", json={"seconds": 30}).json()["id"]
    b = client.post("/api/jobs/selftest", json={"seconds": 1}).json()["id"]
    t0 = time.time()
    while client.get(f"/api/jobs/{a}").json()["status"] != "running":
        assert time.time() - t0 < 10
        time.sleep(0.05)
    time.sleep(1.5)
    assert client.get(f"/api/jobs/{b}").json()["status"] == "queued"   # one GPU job at a time
    pid = client.get(f"/api/jobs/{a}").json()["pid"]
    assert client.post(f"/api/jobs/{a}/cancel").status_code == 200
    ja = wait(client, a)
    assert ja["status"] == "cancelled"
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)                                   # whole process group is gone
    assert wait(client, b)["status"] == "succeeded"        # queue moves on
    assert client.post(f"/api/jobs/{a}/cancel").status_code == 409


def test_cancel_queued(client):
    a = client.post("/api/jobs/selftest", json={"seconds": 3}).json()["id"]
    b = client.post("/api/jobs/selftest", json={"seconds": 3}).json()["id"]
    assert client.post(f"/api/jobs/{b}/cancel").json()["status"] == "cancelled"
    client.post(f"/api/jobs/{a}/cancel")
    wait(client, a)


def test_websocket_stream(client):
    jid = client.post("/api/jobs/selftest", json={"seconds": 2}).json()["id"]
    got = []
    with client.websocket_connect(f"/api/jobs/{jid}/stream") as ws:
        while True:
            m = ws.receive_json()
            got.append(m)
            if m["type"] == "status" and m["status"] not in ("queued", "running"):
                break
    lines = [m["line"] for m in got if m["type"] == "line"]
    assert lines == ["selftest tick 1/2", "selftest tick 2/2", "selftest done"]
    assert got[-1]["status"] == "succeeded"
    seqs = [m["seq"] for m in got if m["type"] == "line"]
    assert seqs == sorted(set(seqs))                        # no duplicates across backlog/live


@pytest.mark.parametrize("bad", [
    {"scene": "../etc"},
    {"scene": "a b"},
    {"scene": "x", "only": "gsplat", "from_step": "course"},
    {"scene": "x", "from_step": "record", "stop_after": "course"},
    {"scene": "x", "video": "../../etc/passwd"},
    {"scene": "x", "video": "missing.mp4"},
    {"scene": "x", "num_images": 10, "num_marked": 40},
    {"scene": "x", "redo": ["rm -rf"]},
    {"scene": "x", "course": "circuit; reboot"},
])
def test_rejects_bad_requests(client, bad):
    assert client.post("/api/jobs/figs", json=bad).status_code in (400, 422)


def test_video_inside_staging_dir(client):
    r = client.post("/api/jobs/figs", json={"scene": "lab3", "video": "lab3.mp4", "stop_after": "aruco"})
    assert r.status_code == 201
    wait(client, r.json()["id"])


def test_scenes_flag_unloadable(client):
    sc = {s["scene"]: s for s in client.get("/api/scenes").json()}
    assert sc["backroom"]["loadable"] is True
    assert sc["backroom1"]["models"] == 0 and sc["backroom1"]["loadable"] is False


def test_token(settings):
    from fastapi.testclient import TestClient
    from galley.app import create_app
    settings.token = "s3cret"
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/scenes").status_code == 401
        assert c.get("/api/scenes", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/api/health").status_code == 200
