import json

from radiance_semantics import env_check

GOOD = {"torch": "2.1.2", "torchvision": "0.16.2", "numpy": "1.26.4", "timm": "0.6.7", "gsplat": "1.0.0",
        "nerfstudio": "1.1.4", "tinycudann": "1.7", "open_clip_torch": None, "ftfy": None, "regex": None,
        "huggingface_hub": None, "radiance-semantics": None}


def test_good_stack_has_no_problems():
    assert env_check.problems(GOOD) == []


def test_moved_torch_numpy2_and_missing_gsplat():
    bad = dict(GOOD, torch="2.5.1", numpy="2.1.0", gsplat=None)
    msgs = " | ".join(env_check.problems(bad))
    assert "torch is 2.5.1" in msgs and "numpy 2.1.0" in msgs and "gsplat is not installed" in msgs


def test_changed_only_watched_and_previously_installed():
    after = dict(GOOD, timm="1.0.20", open_clip_torch="2.24.0")       # open_clip is new: fine
    assert env_check.changed(GOOD, after) == ["timm: 0.6.7 -> 1.0.20"]
    assert env_check.changed(dict(GOOD, tinycudann=None), dict(GOOD, tinycudann="1.7")) == []


def test_constraints_freeze_everything_installed(tmp_path):
    lines = env_check.constraints()
    names = {line.split("==")[0].lower().replace("_", "-") for line in lines}
    assert "pytest" in names and all("==" in line and " @ " not in line for line in lines)
    assert "radiance-semantics" not in names and "open-clip-torch" not in names   # those may change
    out = tmp_path / "c.txt"
    env_check.main(["--constraints", str(out)])
    assert out.read_text().count("==") == len(lines)


def test_cli_snapshot_and_compare(tmp_path, monkeypatch, capsys):
    snap = tmp_path / "before.json"
    monkeypatch.setattr(env_check, "versions", lambda: dict(GOOD))
    assert env_check.main(["--snapshot", str(snap)]) == 0
    assert json.loads(snap.read_text())["torch"] == "2.1.2"
    assert env_check.main(["--compare", str(snap)]) == 0
    monkeypatch.setattr(env_check, "versions", lambda: dict(GOOD, torchvision="0.20.0"))
    assert env_check.main(["--compare", str(snap)]) == 1
    assert "torchvision: 0.16.2 -> 0.20.0" in capsys.readouterr().err
