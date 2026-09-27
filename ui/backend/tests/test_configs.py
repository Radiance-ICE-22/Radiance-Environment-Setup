from __future__ import annotations

import copy
import json

import pytest

from galley.configs import FAMILIES, ConfigStore

CIRCUIT_LIKE = {
    "waypoints": {"Nco": 6, "keyframes": {
        "fo0": {"t": 0.0, "fo": [[1.5, 0.0], [0.0, 0.0], [-0.7, 0.0], [-1.57, 0.0]]},
        "fo0a": {"t": 0.816, "fo": [[1.5, None, None, None], [None, None, None, None],
                                    [None, None, None, None], [-1.8, None, None, None]]},
        "fo7": {"t": 12.107, "fo": [[0.0, 0.0], [0.0, 0.0], [-0.7, 0.0], [-6.28, 0.0]]},
    }},
    "forces": None,
}


def test_every_existing_config_round_trips(settings):
    """Every JSON shipped upstream (and in the overlay) must validate and round-trip unchanged."""
    store = ConfigStore(settings.configs_dir)
    seen = 0
    for fam in FAMILIES:
        for item in store.list(fam):
            data = store.read(fam, item["name"])
            assert store.validate(fam, data) == data, f"{fam}/{item['name']}"
            seen += 1
    if seen == 0:
        pytest.skip("set GALLEY_TEST_CONFIGS to a SousVide configs/ dir to run this")


def test_course_write_and_mirror(client, settings):
    r = client.put("/api/configs/courses/my_loop", json=CIRCUIT_LIKE)
    assert r.status_code == 200, r.text
    assert r.json()["mirrored"].endswith("overlay/configs/courses/my_loop.json")
    assert client.get("/api/configs/courses/my_loop").json() == CIRCUIT_LIKE
    on_disk = (settings.configs_dir / "courses" / "my_loop.json").read_text()
    assert on_disk.startswith("{\n    \"waypoints\"")          # upstream 4-space style


@pytest.mark.parametrize("mutate,msg", [
    (lambda c: c["waypoints"]["keyframes"]["fo0a"].__setitem__("t", 20.0), "strictly increase"),
    (lambda c: c["waypoints"]["keyframes"]["fo0"]["fo"].pop(), "4 rows"),
    (lambda c: c["waypoints"]["keyframes"]["fo0"]["fo"][1].__setitem__(0, None), "first keyframe"),
    (lambda c: c["waypoints"]["keyframes"]["fo0"].__setitem__("t", "0"), "round-trip"),
])
def test_course_rejects(client, mutate, msg):
    bad = copy.deepcopy(CIRCUIT_LIKE)
    mutate(bad)
    r = client.put("/api/configs/courses/bad", json=bad)
    assert r.status_code in (400, 422) and msg in r.text


def test_capture_checks(client):
    cap = {"camera": None, "extractor": {"num_images": 300, "num_marked": 40, "marker_length": 0.18, "marker_id": 0}}
    assert client.put("/api/configs/captures/lab3", json=cap).status_code == 200
    cap["extractor"]["num_marked"] = 400
    assert client.put("/api/configs/captures/lab3", json=cap).status_code == 422
    cap["extractor"].update(num_marked=40, marker_length=18)          # centimetres by mistake
    assert client.put("/api/configs/captures/lab3", json=cap).status_code == 422


def test_names_and_families(client):
    assert client.get("/api/configs/secrets/x").status_code == 400
    assert client.put("/api/configs/courses/..%2Fx", json=CIRCUIT_LIKE).status_code in (400, 404, 405)  # 405 when the static frontend mount answers
    assert client.put("/api/configs/courses/a.b", json=CIRCUIT_LIKE).status_code == 400
    assert client.put("/api/configs/courses/new", params={"overwrite": False}, json=CIRCUIT_LIKE).status_code == 200
    assert client.put("/api/configs/courses/new", params={"overwrite": False}, json=CIRCUIT_LIKE).status_code == 400
