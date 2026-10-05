import os

import pytest

from radiance_semantics.paths import SemanticsError, find_scene_run, latest_checkpoint, resolve_project_root


def make_root(tmp_path, scene="backroom", runs=("2026-09-20_101010",), steps=(10000, 29999)):
    root = tmp_path / "figs"
    (root / "SousVide" / "gsplats" / "workspace" / scene).mkdir(parents=True)
    (root / "figs_env.sh").write_text("# env\n")
    for r in runs:
        d = root / "SousVide" / "gsplats" / "workspace" / "outputs" / scene / "splatfacto" / r
        (d / "nerfstudio_models").mkdir(parents=True)
        (d / "config.yml").write_text("x: 1\n")
        for s in steps:
            (d / "nerfstudio_models" / f"step-{s:09d}.ckpt").write_bytes(b"ck")
    return root


def test_single_run_layout(tmp_path):
    root = make_root(tmp_path)
    run = find_scene_run(root, "backroom")
    assert run.run == "2026-09-20_101010"
    assert run.checkpoint.name == "step-000029999.ckpt"            # highest step, not lexical luck
    assert run.workspace_root == root / "SousVide" / "gsplats" / "workspace"
    assert run.backend_dir("lift") == run.scene_dir / "semantics" / run.run / "lift"
    assert run.teachers_dir("abc") == run.scene_dir / "semantics" / "teachers" / "abc"
    assert run.state_dir == root / ".semantic_pipeline_state" / "backroom" / run.run
    assert run.runs_dir == root / "SousVide" / "runs"
    mtime = int(run.checkpoint.stat().st_mtime)
    assert run.key == f"{run.run}-step-000029999-{mtime}"          # Galley's splat-cache key


def test_step_order_is_numeric(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    for s in (999, 30000, 2000):
        (d / f"step-{s:09d}.ckpt").write_bytes(b"")
    assert latest_checkpoint(d).name == "step-000030000.ckpt"


def test_no_model_and_two_models(tmp_path):
    root = make_root(tmp_path, runs=())
    with pytest.raises(SemanticsError, match="no trained model"):
        find_scene_run(root, "backroom")
    root2 = make_root(tmp_path / "b", runs=("r1", "r2"))
    with pytest.raises(SemanticsError, match="2 trained models"):
        find_scene_run(root2, "backroom")


def test_bad_names(tmp_path):
    root = make_root(tmp_path)
    for bad in ("", "../etc", "a/b", "x" * 70):
        with pytest.raises(SemanticsError):
            find_scene_run(root, bad)
    run = find_scene_run(root, "backroom")
    with pytest.raises(SemanticsError):
        run.backend_dir("nerf")


def test_unfinished_training(tmp_path):
    root = make_root(tmp_path, steps=())
    run = find_scene_run(root, "backroom")
    with pytest.raises(SemanticsError, match="training has not finished"):
        run.checkpoint


def test_resolve_project_root(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    monkeypatch.delenv("FIGS_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("ACADOS_SOURCE_DIR", raising=False)
    with pytest.raises(SemanticsError):
        resolve_project_root()
    acados = root / "SousVide" / "FiGS" / "acados"
    acados.mkdir(parents=True)
    monkeypatch.setenv("ACADOS_SOURCE_DIR", str(acados))         # what figs_env.sh exports
    assert resolve_project_root() == root.resolve()
    assert resolve_project_root(str(root)) == root.resolve()
    with pytest.raises(SemanticsError):
        resolve_project_root(str(tmp_path / "nowhere"))
    monkeypatch.setenv("FIGS_PROJECT_ROOT", os.fspath(root))
    assert resolve_project_root() == root.resolve()
