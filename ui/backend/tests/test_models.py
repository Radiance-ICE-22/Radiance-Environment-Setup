from __future__ import annotations

import pathlib


def _model(project: pathlib.Path, where: str, run: str, ckpt=True):
    base = project / "SousVide" / "gsplats" / "workspace"
    d = base / ("outputs/backroom/splatfacto" if where == "active" else "_archive/backroom") / run
    (d / "nerfstudio_models").mkdir(parents=True, exist_ok=True)
    (d / "config.yml").write_text("x")
    if ckpt:
        (d / "nerfstudio_models" / "step-000029999.ckpt").write_bytes(b"\0" * 2048)
    return d


def test_archive_and_promote_round_trip(client, project):
    state = project / ".figs_pipeline_state" / "backroom"
    state.mkdir(parents=True, exist_ok=True)
    for st in ("verify", "simulate", "validate", "bounds"):
        (state / f"{st}.done").write_text("fp\nwhen\n")
    _model(project, "archived", "2026-09-27_134959")
    m = client.get("/api/scenes/backroom/models").json()
    assert [x["run"] for x in m["active"]] == ["2025-01-16_122349"]
    assert [x["run"] for x in m["archived"]] == ["2026-09-27_134959"]

    r = client.post("/api/scenes/backroom/models/2026-09-27_134959/promote").json()
    assert r["promoted"] == "2026-09-27_134959" and r["archived"] == ["2025-01-16_122349"]
    assert set(r["cleared_steps"]) == {"verify", "simulate", "validate"}
    assert (state / "bounds.done").exists()                       # frame-only steps untouched
    m = client.get("/api/scenes/backroom/models").json()
    assert [x["run"] for x in m["active"]] == ["2026-09-27_134959"] and len(m["archived"]) == 1
    assert client.get("/api/scenes").json()[0]["loadable"] is True  # still exactly one model

    assert client.post("/api/scenes/backroom/models/2026-09-27_134959/archive").status_code == 200
    assert client.get("/api/scenes/backroom/models").json()["active"] == []


def test_refusals(client, project):
    _model(project, "archived", "half", ckpt=False)
    assert client.post("/api/scenes/backroom/models/half/promote").status_code == 400      # no checkpoint
    assert client.post("/api/scenes/backroom/models/nope/archive").status_code == 400
    assert client.post("/api/scenes/backroom/models/..%2F..%2Fx/archive").status_code in (400, 404, 405)
    jid = client.post("/api/jobs/figs", json={"scene": "backroom", "only": "preflight"}).json()["id"]
    r = client.post("/api/scenes/backroom/models/2025-01-16_122349/archive")
    j = client.get(f"/api/jobs/{jid}").json()
    assert r.status_code == 409 or j["status"] not in ("queued", "running")


def test_metrics_endpoint(client, project):
    from tensorboardX import SummaryWriter  # test-only dependency
    run = project / "SousVide/gsplats/workspace/outputs/backroom/splatfacto/2025-01-16_122349"
    w = SummaryWriter(str(run))
    for s in range(0, 1000, 100):
        w.add_scalar("Train Loss", 1 / (1 + s), s)
    w.close()
    r = client.get("/api/scenes/backroom/metrics").json()
    assert r["run"] == "2025-01-16_122349"
    (tag, pts), = r["series"].items()
    assert "Loss" in tag and pts[0] == [0, 1.0] and len(pts) == 10
