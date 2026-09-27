"""One-at-a-time job runner.

Every job is an argv list executed as

    bash -c 'source <project_root>/figs_env.sh >/dev/null && exec "$@"' galley <argv...>

so it gets exactly the environment the pipeline docs require (ACADOS_SOURCE_DIR,
LD_LIBRARY_PATH, PYTHONNOUSERSITE, the kitchen env). LD_LIBRARY_PATH cannot be fixed
inside a running process, which is why the wrapper is a fresh shell per job.

The GPU is treated as exclusive: jobs run strictly in submission order, one at a time.
Each job gets its own process group, so cancel reaches the whole tree (ns-train,
COLMAP, ffmpeg) rather than just the top-level Python process.
"""
from __future__ import annotations

import asyncio
import os
import re
import signal
import time
from collections import defaultdict

from .db import DB
from .settings import Settings

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
CANCEL_GRACE_S = 10


class JobRunner:
    def __init__(self, settings: Settings, db: DB):
        self.s = settings
        self.db = db
        self._wake = asyncio.Event()
        self._subs: dict[int, set[asyncio.Queue]] = defaultdict(set)
        self._proc: asyncio.subprocess.Process | None = None
        self._current: int | None = None
        self._cancel_requested: set[int] = set()
        self._task: asyncio.Task | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        self.db.recover()
        self._task = asyncio.create_task(self._loop(), name="galley-worker")

    async def stop(self) -> None:
        if self._current is not None:
            await self.cancel(self._current)
        if self._task:
            self._task.cancel()

    # ── public API ────────────────────────────────────────────────────────────
    def submit(self, kind: str, label: str, argv: list[str], params: dict, scene: str | None) -> int:
        job_id = self.db.add_job(kind, label, argv, params, scene)
        self._wake.set()
        return job_id

    @property
    def current(self) -> int | None:
        return self._current

    async def cancel(self, job_id: int) -> bool:
        job = self.db.job(job_id)
        if not job or job["status"] not in ("queued", "running"):
            return False
        if job["status"] == "queued":
            self.db.update_job(job_id, status="cancelled", finished=time.time())
            self._publish(job_id, {"type": "status", "status": "cancelled"})
            return True
        self._cancel_requested.add(job_id)
        proc = self._proc
        if proc and self._current == job_id and proc.returncode is None:
            _signal_group(proc.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), CANCEL_GRACE_S)
            except asyncio.TimeoutError:
                _signal_group(proc.pid, signal.SIGKILL)
        return True

    def subscribe(self, job_id: int) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._subs[job_id].add(q)
        return q

    def unsubscribe(self, job_id: int, q: asyncio.Queue) -> None:
        self._subs[job_id].discard(q)
        if not self._subs[job_id]:
            self._subs.pop(job_id, None)

    # ── internals ─────────────────────────────────────────────────────────────
    def _publish(self, job_id: int, msg: dict) -> None:
        for q in list(self._subs.get(job_id, ())):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:  # a stalled client must not block the worker
                pass

    async def _loop(self) -> None:
        while True:
            job = self.db.next_queued()
            if job is None:
                self._wake.clear()
                await self._wake.wait()
                continue
            await self._run(job)

    async def _run(self, job: dict) -> None:
        jid = job["id"]
        self._current = jid
        env = os.environ.copy()
        env.update(TERM="dumb", NO_COLOR="1", PYTHONUNBUFFERED="1", GALLEY_JOB_ID=str(jid))
        script = f'source "{self.s.env_script}" >/dev/null 2>&1 || {{ echo "galley: cannot source {self.s.env_script}"; exit 97; }}; exec "$@"'
        seq = 0
        try:
            self._proc = await asyncio.create_subprocess_exec(
                "bash", "-c", script, "galley", *job["argv"],
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL, env=env, start_new_session=True,
            )
        except OSError as e:
            self.db.add_line(jid, 0, f"galley: failed to start: {e}")
            self.db.update_job(jid, status="failed", finished=time.time(), returncode=-1)
            self._publish(jid, {"type": "status", "status": "failed"})
            self._current = None
            return

        self.db.update_job(jid, status="running", started=time.time(), pid=self._proc.pid)
        self._publish(jid, {"type": "status", "status": "running"})

        buf = ""
        assert self._proc.stdout is not None
        while True:
            chunk = await self._proc.stdout.read(4096)
            if not chunk:
                break
            buf += ANSI.sub("", chunk.decode("utf-8", "replace"))
            # "\n" ends a line; a bare "\r" is a tqdm/rich progress redraw: shown live, not stored
            while True:
                m = re.search(r"\r\n|\n|\r", buf)
                if not m:
                    break
                seg, sep, buf = buf[: m.start()], m.group(), buf[m.end():]
                if sep == "\r":
                    if seg.strip():
                        self._publish(jid, {"type": "progress", "line": seg})
                    continue
                self.db.add_line(jid, seq, seg)
                self._publish(jid, {"type": "line", "seq": seq, "line": seg})
                seq += 1
        if buf.strip():
            self.db.add_line(jid, seq, buf)
            self._publish(jid, {"type": "line", "seq": seq, "line": buf})

        rc = await self._proc.wait()
        if jid in self._cancel_requested:
            status = "cancelled"
            self._cancel_requested.discard(jid)
        else:
            status = "succeeded" if rc == 0 else "failed"
        self.db.update_job(jid, status=status, finished=time.time(), returncode=rc)
        self._publish(jid, {"type": "status", "status": status, "returncode": rc})
        self._proc = None
        self._current = None


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass
