"""Machine profile lookup: env var, ui/machine.toml, then hostname prefix."""
from __future__ import annotations

import pytest

from galley import settings as st

PROFILE = '[paths]\nproject_root = "{root}"\npipeline = "{root}/p.py"\ndata_dir = "{root}/data"\n'


@pytest.fixture
def ui(tmp_path, monkeypatch):
    (tmp_path / "machines").mkdir()
    for name in ("dummy", "intellisense08", "intellisense"):
        (tmp_path / "machines" / f"{name}.toml").write_text(PROFILE.format(root=tmp_path))
    monkeypatch.setattr(st, "UI_DIR", tmp_path)
    monkeypatch.delenv("GALLEY_MACHINE", raising=False)
    return tmp_path


def test_hostname_prefix_most_specific_wins(ui):
    assert st.find_profile("intellisense08-EWISPro9900G").name == "intellisense08.toml"
    assert st.find_profile("intellisense05-EWISPro9900G").name == "intellisense.toml"
    assert st.find_profile("dummy").name == "dummy.toml"


def test_unknown_host_lists_profiles(ui):
    with pytest.raises(st.NoMachineProfile, match="dummy.toml, intellisense.toml, intellisense08.toml"):
        st.find_profile("somewhere-else")


def test_env_and_machine_toml_take_precedence(ui, monkeypatch):
    (ui / "machine.toml").write_text(PROFILE.format(root=ui))
    assert st.find_profile("dummy").name == "machine.toml"
    monkeypatch.setenv("GALLEY_MACHINE", str(ui / "machines" / "dummy.toml"))
    assert st.find_profile("intellisense08-x").name == "dummy.toml"
    monkeypatch.setenv("GALLEY_MACHINE", str(ui / "nope.toml"))
    with pytest.raises(st.NoMachineProfile, match="no such file"):
        st.find_profile()


def test_load_records_source(ui, monkeypatch):
    monkeypatch.setenv("GALLEY_MACHINE", str(ui / "machines" / "dummy.toml"))
    assert st.load().source.name == "dummy.toml"
