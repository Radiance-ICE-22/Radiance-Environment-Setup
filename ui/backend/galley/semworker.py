"""Client for figs/semantic_worker.py: a long-lived CPU process in the kitchen env.

Galley's own venv has no torch, so semantic queries go to a worker started like the course
tools (`bash -c 'source figs_env.sh; exec python semantic_worker.py …'`) with
CUDA_VISIBLE_DEVICES="" — it can never take the GPU from a queued job. Unlike the course tools it
stays alive between requests, keeping the CLIP text tower and the memory-mapped feature tables
loaded (first query ~6 s, later ones under a second). It exits by itself after an idle period; the
next request starts it again. A crash costs only the request in flight.

One request at a time (a lock): the worker is single-threaded and queries take well under a second.
The worker's stderr goes to <data_dir>/semantic_worker.log.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from .settings import Settings


class WorkerError(RuntimeError):
    """The worker could not be started or died; HTTP 503."""


class WorkerRefused(ValueError):
    """The worker answered with an error (bad request, no table …); HTTP 400/404."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def script(s: Settings) -> Path:
    return s.pipeline.parent / "semantic_worker.py"


class SemWorker:
    START_TIMEOUT = 60.0         # bash + conda activate + imports
    REQUEST_TIMEOUT = 180.0      # first query loads CLIP (~5 s) and maps a ~1 GB table

    def __init__(self, s: Settings, idle_s: float | None = None):
        self.s = s
        self.idle_s = float(idle_s if idle_s is not None else s.defaults.get("semantic_worker_idle_s", 600))
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._next_id = 0
        self.started_at: float | None = None
        self.last_used: float | None = None
        self.restarts = 0

    @property
    def log_path(self) -> Path:
        return self.s.data_dir / "semantic_worker.log"

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict:
        return {"running": self.alive(), "pid": self._proc.pid if self.alive() else None,
                "started": self.started_at, "last_used": self.last_used, "idle_s": self.idle_s,
                "restarts": self.restarts, "script": str(script(self.s)), "log": str(self.log_path)}

    # ── process management ────────────────────────────────────────────────────
    def _start(self) -> None:
        sc = script(self.s)
        if not sc.exists():
            raise WorkerError(f"{sc} not found (it ships next to figs_pipeline.py)")
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1", TERM="dumb", NO_COLOR="1")
        shell = (f'source "{self.s.env_script}" >/dev/null 2>&1 '
                 f'|| {{ echo "cannot source {self.s.env_script}" >&2; exit 97; }}; exec "$@"')
        self.s.data_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")
        try:
            self._proc = subprocess.Popen(
                ["bash", "-c", shell, "galley-semantic", self.s.python, str(sc), "--project-root",
                 str(self.s.project_root), "--idle", str(self.idle_s)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env, text=True,
                encoding="utf-8", bufsize=1, start_new_session=True)
        finally:
            log.close()
        self._lines = queue.Queue()
        proc, lines = self._proc, self._lines

        def pump():
            for line in proc.stdout:            # ends at EOF (exit, crash, idle stop)
                lines.put(line)
            lines.put(None)
        threading.Thread(target=pump, daemon=True).start()
        first = self._read(self.START_TIMEOUT)
        if not first.get("ready"):
            self._kill()
            raise WorkerError(f"unexpected first line from the worker: {first}")
        self.started_at = time.time()

    def _read(self, timeout: float) -> dict:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            self._kill()
            raise WorkerError(f"semantic worker did not answer within {timeout:.0f} s (killed; see {self.log_path})")
        if line is None:
            rc = self._proc.wait() if self._proc else None
            self._proc = None
            raise WorkerError(f"semantic worker exited (code {rc}); last log lines: {self._tail()}")
        try:
            return json.loads(line)
        except ValueError:
            raise WorkerError(f"semantic worker wrote non-JSON: {line[:200]!r}")

    def _tail(self, n: int = 4) -> str:
        try:
            return " | ".join(self.log_path.read_text(errors="replace").strip().splitlines()[-n:])
        except OSError:
            return ""

    def _kill(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(self._proc.pid, 15)
                self._proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self._proc.pid, 9)
                except ProcessLookupError:
                    pass
        self._proc = None

    def stop(self) -> bool:
        with self._lock:
            was = self.alive()
            if self._proc and self._proc.stdin:
                try:
                    self._proc.stdin.close()     # EOF: the worker returns from serve()
                    self._proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self._kill()
            return was

    # ── requests ──────────────────────────────────────────────────────────────
    def request(self, op: str, timeout: float | None = None, **body) -> dict:
        with self._lock:
            for attempt in (1, 2):              # a worker that idled out or crashed between calls: restart once
                if not self.alive():
                    if self._proc is not None or self.started_at is not None:
                        self.restarts += 1
                    self._start()
                self._next_id += 1
                rid = self._next_id
                try:
                    self._proc.stdin.write(json.dumps({"id": rid, "op": op, **body}) + "\n")
                    self._proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    self._kill()
                    if attempt == 2:
                        raise WorkerError("semantic worker closed its input")
                    continue
                reply = self._read(timeout or self.REQUEST_TIMEOUT)
                while reply.get("id") != rid:  # a stale reply from a timed-out request
                    reply = self._read(timeout or self.REQUEST_TIMEOUT)
                self.last_used = time.time()
                if not reply.get("ok"):
                    raise WorkerRefused(reply.get("code", "error"), reply.get("error", "semantic worker error"))
                return reply
            raise WorkerError("semantic worker unavailable")
