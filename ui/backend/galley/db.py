"""SQLite store for jobs and their log lines. Small, synchronous, guarded by a lock."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    label       TEXT NOT NULL,
    argv        TEXT NOT NULL,          -- JSON list, run after sourcing figs_env.sh
    params      TEXT NOT NULL,          -- JSON of the validated request
    scene       TEXT,
    status      TEXT NOT NULL,          -- queued | running | succeeded | failed | cancelled | interrupted
    created     REAL NOT NULL,
    started     REAL,
    finished    REAL,
    returncode  INTEGER,
    pid         INTEGER
);
CREATE TABLE IF NOT EXISTS logs (
    job_id  INTEGER NOT NULL,
    seq     INTEGER NOT NULL,
    ts      REAL NOT NULL,
    line    TEXT NOT NULL,
    PRIMARY KEY (job_id, seq)
);
"""

ACTIVE = ("queued", "running")


class DB:
    def __init__(self, path: Path):
        self._c = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._c.row_factory = sqlite3.Row
        self._c.execute("PRAGMA journal_mode=WAL")
        self._c.executescript(SCHEMA)
        self._lock = threading.Lock()

    def _x(self, sql: str, args: tuple = ()):
        with self._lock:
            return self._c.execute(sql, args)

    # ── jobs ──────────────────────────────────────────────────────────────────
    def recover(self) -> int:
        """Jobs left running/queued by a previous server process can never finish."""
        cur = self._x("UPDATE jobs SET status='interrupted', finished=? WHERE status IN ('queued','running')",
                      (time.time(),))
        return cur.rowcount

    def add_job(self, kind: str, label: str, argv: list[str], params: dict, scene: str | None) -> int:
        cur = self._x("INSERT INTO jobs(kind,label,argv,params,scene,status,created) VALUES (?,?,?,?,?,?,?)",
                      (kind, label, json.dumps(argv), json.dumps(params), scene, "queued", time.time()))
        return int(cur.lastrowid)

    def update_job(self, job_id: int, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        self._x(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def job(self, job_id: int) -> dict | None:
        r = self._x("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _job(r) if r else None

    def jobs(self, limit: int = 50, scene: str | None = None) -> list[dict]:
        if scene:
            rows = self._x("SELECT * FROM jobs WHERE scene=? ORDER BY id DESC LIMIT ?", (scene, limit))
        else:
            rows = self._x("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))
        return [_job(r) for r in rows.fetchall()]

    def next_queued(self) -> dict | None:
        r = self._x("SELECT * FROM jobs WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
        return _job(r) if r else None

    # ── logs ──────────────────────────────────────────────────────────────────
    def add_line(self, job_id: int, seq: int, line: str) -> None:
        self._x("INSERT INTO logs(job_id,seq,ts,line) VALUES (?,?,?,?)", (job_id, seq, time.time(), line))

    def lines(self, job_id: int, after: int = -1, limit: int = 5000) -> list[dict]:
        rows = self._x("SELECT seq, ts, line FROM logs WHERE job_id=? AND seq>? ORDER BY seq LIMIT ?",
                       (job_id, after, limit)).fetchall()
        return [dict(r) for r in rows]


def _job(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["argv"] = json.loads(d["argv"])
    d["params"] = json.loads(d["params"])
    return d
