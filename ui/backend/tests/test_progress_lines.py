"""figs_pipeline.sh() must pass a child's progress-bar redraws ("\\r") through unchanged, so
Galley's job log updates one line in place instead of storing every redraw."""
from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# a child that behaves like hloc's matcher under tqdm, plus a UTF-8 character split across writes
CHILD = textwrap.dedent(r'''
    import sys, time
    out = sys.stdout
    out.write("Matching pairs\n"); out.flush()
    for i in range(1, 201):
        out.write(f"\r {i // 2:3d}%|##        | {i}/200 [00:01<00:00, 12.0it/s]"); out.flush()
        if i % 50 == 0:
            time.sleep(0.01)
    out.write("\n"); out.flush()
    b = "done ✓\n".encode()
    sys.stdout.buffer.write(b[:6]); sys.stdout.buffer.flush(); time.sleep(0.05)
    sys.stdout.buffer.write(b[6:]); sys.stdout.buffer.flush()
    out.write("half\r\n"); out.flush()
''')


def _figs():
    spec = importlib.util.spec_from_file_location("figs_pipeline_under_test", REPO / "figs" / "figs_pipeline.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_sh_keeps_redraws_and_collapses_output(capsys):
    m = _figs()
    rc, out = m.sh([sys.executable, "-c", CHILD])
    streamed = capsys.readouterr().out
    assert rc == 0
    assert streamed.count("\r") >= 200                        # redraws passed through as redraws
    assert streamed.count("\n") == 4
    assert out.splitlines() == ["Matching pairs", " 100%|##        | 200/200 [00:01<00:00, 12.0it/s]", "done ✓", "half"]


def test_galley_log_keeps_one_line_per_bar(client, settings):
    """End to end: a job whose script runs a progress-bar child through sh()."""
    wrapper = settings.project_root / "sh_job.py"
    wrapper.write_text(textwrap.dedent(f'''
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("fp", {str(REPO / "figs" / "figs_pipeline.py")!r})
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        m.sh([sys.executable, "-c", {CHILD!r}])
    '''))
    from galley.app import create_app  # noqa: F401  (client fixture already built the app)
    runner = client.app.state.runner
    jid = runner.submit("selftest", "sh progress", [sys.executable, str(wrapper)], {}, None)
    for _ in range(200):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert j["status"] == "succeeded", j
    lines = [r["line"] for r in client.get(f"/api/jobs/{jid}/log").json()]
    # the bar is stored once, in its final state; its 199 earlier redraws were only shown live
    assert lines == ["Matching pairs", " 100%|##        | 200/200 [00:01<00:00, 12.0it/s]", "done ✓", "half"], json.dumps(lines)
