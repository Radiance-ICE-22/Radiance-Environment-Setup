"""Phase 3: course editor endpoints, with a fake course_tools.py standing in for the kitchen env."""
from __future__ import annotations

import json
import textwrap

import pytest

FAKE_TOOLS = textwrap.dedent('''
    import json, os, sys, time
    args = sys.argv[1:]
    print("some log noise from figs_env.sh")
    if args[0] == "geometry":
        scene = args[args.index("--scene") + 1]
        n = int(open(os.environ["FAKE_COUNT"]).read() or 0) + 1 if os.path.exists(os.environ["FAKE_COUNT"]) else 1
        open(os.environ["FAKE_COUNT"], "w").write(str(n))
        print("GALLEY_JSON " + json.dumps({"ok": True, "scene": scene, "calls": n,
              "points": [0.0] * 3000, "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}))
    else:
        course = json.loads(sys.stdin.read())
        if "--scene" in args and args[args.index("--scene") + 1] == "broken":
            print("GALLEY_JSON " + json.dumps({"ok": False, "error": "ValueError: singular matrix"}))
            sys.exit(1)
        time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
        print("GALLEY_JSON " + json.dumps({"ok": True, "args": args, "course": course}))
''')

COURSE = {"waypoints": {"Nco": 6, "keyframes": {
    "a": {"t": 0.0, "fo": [[0.4, 0.0], [0.0, 0.0], [-1.0, 0.0], [0.0, 0.0]]},
    "b": {"t": 2.0, "fo": [[1.0, None, None, None], [None, None, None, None], [-1.0, None], [None]]},
    "c": {"t": 4.0, "fo": [[1.5, 0.0], [1.0, 0.0], [-1.2, 0.0], [1.57, 0.0]]}}}, "forces": None}


@pytest.fixture
def tools(project, monkeypatch, tmp_path):
    (project / "course_tools.py").write_text(FAKE_TOOLS)
    ws = project / "SousVide" / "gsplats" / "workspace" / "backroom"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "transforms.json").write_text("{}")
    monkeypatch.setenv("FAKE_COUNT", str(tmp_path / "count"))
    return project


def test_geometry_runs_tool_in_env_and_caches(client, tools):
    r = client.get("/api/scenes/backroom/geometry")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["scene"] == "backroom" and d["calls"] == 1 and d["cuda"] == ""
    assert r.headers.get("content-encoding") == "gzip"          # large payload compressed
    assert client.get("/api/scenes/backroom/geometry").json()["calls"] == 1   # cached
    ws = tools / "SousVide" / "gsplats" / "workspace" / "backroom"
    (ws / "sparse_pc.ply").write_text("ply")                     # a new file invalidates
    assert client.get("/api/scenes/backroom/geometry").json()["calls"] == 2


def test_geometry_refuses_scene_without_sfm(client, tools):
    assert client.get("/api/scenes/nothing/geometry").status_code == 400
    assert client.get("/api/scenes/..%2Fx/geometry").status_code in (400, 404)


def test_preview_passes_float_course_and_flags(client, tools):
    body = json.loads(json.dumps({"course": COURSE, "scene": "backroom", "mode": "expert", "clearance": 0.4}))
    # what a browser sends: JSON.stringify(0.0) == "0"
    raw = json.dumps(body).replace("0.0,", "0,").replace("0.0]", "0]")
    r = client.post("/api/courses/preview", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 200, r.text
    d = r.json()
    a = d["args"]
    assert a[a.index("--mode") + 1] == "expert" and a[a.index("--scene") + 1] == "backroom"
    assert a[a.index("--clearance") + 1] == "0.4" and a[a.index("--clearance-k") + 1] == "5"
    assert a[a.index("--body-radius") + 1] == "0.0"          # default: the drone is a point
    cell = d["course"]["waypoints"]["keyframes"]["a"]["fo"][0][1]
    assert cell == 0.0 and isinstance(cell, float)


def test_preview_rejects_bad_course_and_names(client, tools):
    bad = json.loads(json.dumps(COURSE))
    bad["waypoints"]["keyframes"]["b"]["t"] = 5.0                # times not increasing
    assert client.post("/api/courses/preview", json={"course": bad}).status_code == 422
    assert client.post("/api/courses/preview", json={"course": COURSE, "pilot": "../x"}).status_code == 422


def test_preview_tool_error_is_502(client, tools):
    ws = tools / "SousVide" / "gsplats" / "workspace" / "broken"
    ws.mkdir(parents=True)
    r = client.post("/api/courses/preview", json={"course": COURSE, "scene": "broken"})
    assert r.status_code == 502 and "singular" in r.json()["detail"]


def test_preview_is_single_flight(client, tools, monkeypatch):
    import threading
    monkeypatch.setenv("FAKE_SLEEP", "1.5")
    codes = []
    t = threading.Thread(target=lambda: codes.append(client.post("/api/courses/preview", json={"course": COURSE}).status_code))
    t.start()
    import time
    time.sleep(0.5)
    second = client.post("/api/courses/preview", json={"course": COURSE}).status_code
    t.join()
    assert codes == [200] and second == 409


def test_saved_course_cells_are_floats(client, settings):
    raw = json.dumps(COURSE).replace("0.0,", "0,").replace("0.0]", "0]")
    assert '"t": 0,' in raw
    r = client.put("/api/configs/courses/edited", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 200, r.text
    text = (settings.configs_dir / "courses" / "edited.json").read_text()
    saved = json.loads(text)
    fo = saved["waypoints"]["keyframes"]["a"]["fo"]
    assert fo[0] == [0.4, 0.0] and all(isinstance(v, float) for row in fo for v in row)
    assert client.get("/api/courses/edited/lint").json() == {"int_cells": []}
    mirrored = settings.overlay / "configs" / "courses" / "edited.json"
    assert mirrored.read_text() == text


def test_lint_finds_integer_cells(client, settings):
    p = settings.configs_dir / "courses" / "handmade.json"
    p.write_text('{"waypoints": {"Nco": 6, "keyframes": {"a": {"t": 0, "fo": [[0.4, 0], [0.0], [-1.0], [0.0]]},'
                 ' "b": {"t": 1.0, "fo": [[1.0], [0.0], [-1.0], [0]]}}}, "forces": null}')
    assert client.get("/api/courses/handmade/lint").json() == {"int_cells": ["a.fo[0][1]", "b.fo[3][0]"]}


def test_course_file_layout_and_goal(client, settings):
    body = json.loads(json.dumps(COURSE))
    body["semantic_goal"] = {"label": "the [red] chair, left", "position": [1.0, -2.0, -1.1]}
    assert client.put("/api/configs/courses/withgoal", json=body).status_code == 200
    text = (settings.configs_dir / "courses" / "withgoal.json").read_text()
    assert "[0.4, 0.0]" in text and "[1.0, null, null, null]" in text      # rows on one line
    assert json.loads(text) == body


def test_preview_passes_body_radius(client, tools):
    r = client.post("/api/courses/preview", json={"course": COURSE, "body_radius": 0.19})
    assert r.status_code == 200, r.text
    a = r.json()["args"]
    assert a[a.index("--body-radius") + 1] == "0.19"
    for bad in (-0.1, 3):
        assert client.post("/api/courses/preview", json={"course": COURSE, "body_radius": bad}).status_code == 422
