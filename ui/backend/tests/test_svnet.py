"""Phase 4: SV-Net jobs and cohort readers, with a fake svnet_pipeline.py."""
from __future__ import annotations

import json
import textwrap
import time

import pytest

from galley import svnet as sv

FAKE_SVNET = textwrap.dedent('''
    import argparse, json, os, sys
    from pathlib import Path
    ap = argparse.ArgumentParser()
    for f in ("--project-root", "--cohort", "--scene", "--courses", "--roster", "--only", "--from", "--stop-after"):
        ap.add_argument(f)
    a, rest = ap.parse_known_args()
    print("argv:", json.dumps(sys.argv[1:]), flush=True)
    root = Path(a.project_root)
    st = root / ".svnet_pipeline_state" / a.cohort
    st.mkdir(parents=True, exist_ok=True)
    cfg = {"scene": a.scene or "backroom", "courses": (a.courses or "circuit").split(","), "roster": ["Maverick"],
           "method": "data_alpha"}
    (st / "config.json").write_text(json.dumps(cfg))
    for s in ("rollout", "observe", "train_hist"):
        (st / f"{s}.done").write_text("abc\\n2026-09-27T20:00:00\\n")
    with open(st / "live_Maverick_histNet.jsonl", "w") as fh:
        for e in range(1, 4):
            fh.write(json.dumps({"epoch": e, "loss": 1.0 / e, "t": 0}) + "\\n")
        fh.write("not json\\n")
    data = root / "SousVide" / "cohorts" / a.cohort
    (data / "rollout_data" / "circuit").mkdir(parents=True, exist_ok=True)
    (data / "rollout_data" / "circuit" / "x.pt").write_bytes(b"\\0" * 1000)
    (data / "deployment_data").mkdir(parents=True, exist_ok=True)
    (data / "deployment_data" / "sim_circuit_Maverick_rgb.mp4").write_bytes(b"\\0" * 10)
    res = {"cohort": a.cohort, "deploy": {"pilots": {
        "Viper": {"role": "expert", "tte": {"mean_m": 0.01}},
        "Maverick": {"role": "student", "tte": {"mean_m": 0.2}}}}}
    (st / "results.json").write_text(json.dumps(res))
    print("Done", flush=True)
''')


@pytest.fixture
def fake(project):
    (project / "svnet_pipeline.py").write_text(FAKE_SVNET)
    return project


def _wait(client, jid, t=20):
    for _ in range(int(t / 0.1)):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_argv(settings):
    r = sv.SvnetRun(cohort="c1", scene="backroom", courses=["circuit", "gate3_loop"], roster=["Maverick", "Iceman"],
                    use_compress=False, comm_eval="none", hist_epochs=10, fresh=["histNet"], redo=["train_hist"],
                    from_step="train_hist")
    a = sv.build_argv(settings, r)
    assert a[a.index("--courses") + 1] == "circuit,gate3_loop" and a[a.index("--roster") + 1] == "Maverick,Iceman"
    assert a[a.index("--use-compress") + 1] == "no" and a[a.index("--comm-eval") + 1] == "none"
    assert a[a.index("--fresh") + 1] == "histNet" and a[a.index("--redo") + 1] == "train_hist"
    assert a[a.index("--from") + 1] == "train_hist" and "--scene" in a and "--lr" not in a
    assert a[1].endswith("svnet_pipeline.py")


@pytest.mark.parametrize("body", [
    {"cohort": "../x", "scene": "backroom", "courses": ["c"]},
    {"cohort": "c1", "scene": "backroom", "courses": ["a;rm"]},
    {"cohort": "c1", "only": "deploy", "from_step": "rollout"},
    {"cohort": "c1", "from_step": "deploy", "stop_after": "rollout"},
    {"cohort": "c1", "fresh": ["featNet"]},
])
def test_bad_requests(client, fake, body):
    assert client.post("/api/jobs/svnet", json=body).status_code == 422


def test_new_cohort_needs_scene_and_course(client, fake):
    r = client.post("/api/jobs/svnet", json={"cohort": "c1"})
    assert r.status_code == 400 and "scene" in r.json()["detail"]


def test_run_and_read_cohort(client, fake):
    r = client.post("/api/jobs/svnet", json={"cohort": "c1", "scene": "backroom", "courses": ["circuit"],
                                             "stop_after": "train_hist"})
    assert r.status_code == 201, r.text
    j = _wait(client, r.json()["id"])
    assert j["status"] == "succeeded" and j["scene"] == "backroom" and j["label"] == "svnet c1 [start..train_hist]"
    assert [x["id"] for x in client.get("/api/jobs?cohort=c1").json()] == [j["id"]]
    assert client.get("/api/jobs?cohort=other").json() == []

    lst = client.get("/api/cohorts").json()
    assert lst == [{"cohort": "c1", "scene": "backroom", "courses": ["circuit"], "method": "data_alpha",
                    "roster": ["Maverick"], "done": ["rollout", "observe", "train_hist"],
                    "students": {"Maverick": 0.2}, "managed": True}]
    d = client.get("/api/cohorts/c1").json()
    assert d["live"] == {"Maverick": {"histNet": [[1, 1.0], [2, 0.5], [3, 1 / 3]]}}
    assert d["disk_gb"]["rollout_data"] == 0.0 and d["exists"]
    assert [s["step"] for s in d["steps"] if s["done"]] == ["rollout", "observe", "train_hist"]

    # a second run needs neither scene nor courses: the cohort remembers them
    r = client.post("/api/jobs/svnet", json={"cohort": "c1", "only": "deploy"})
    assert r.status_code == 201
    assert _wait(client, r.json()["id"])["scene"] == "backroom"


def test_video_endpoint(client, fake):
    _wait(client, client.post("/api/jobs/svnet", json={"cohort": "c2", "scene": "backroom", "courses": ["circuit"]}).json()["id"])
    assert client.get("/api/cohorts/c2/video/sim_circuit_Maverick_rgb.mp4").status_code == 200
    assert client.get("/api/cohorts/c2/video/sim_circuit_Iceman_rgb.mp4").status_code == 404
    assert client.get("/api/cohorts/c2/video/..%2F..%2Fsecret_rgb.mp4").status_code in (400, 404)
    assert client.get("/api/cohorts/c2/video/sim_x_rgb.mp4.pt").status_code == 400


def test_unknown_cohort_is_empty(client, fake):
    d = client.get("/api/cohorts/nothing").json()
    assert d["config"] == {} and not d["exists"] and all(not s["done"] for s in d["steps"])
    assert client.get("/api/cohorts/bad..name").status_code == 400
