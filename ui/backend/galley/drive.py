"""Google Drive imports: the host downloads videos the user picked in Google's file picker.

The browser signs in with Google Identity Services (scope drive.file: only files the user
picks in the Picker become readable) and sends each picked file's id with the short-lived
access token. The host then fetches the bytes straight from Google, so a slow home uplink
or a relayed Tailscale path never carries the video; the lab's download link does.

The token lives only in memory, is never logged, persisted or returned. A download writes to
<video_dir>/.uploads/<id>.part (with <id>.json beside it), resumes with an HTTP Range request,
checks Drive's md5Checksum, and only then is renamed into video_captures/. After a server
restart, or when the token expires (about an hour), the import waits for Resume, which signs
in again and continues from the bytes already on disk.

Credentials (OAuth client id, API key, project number) are public values that end up in the
page anyway; they are kept in <data_dir>/google.toml, outside the repo, set from the UI.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .settings import Settings
from .videos import RESERVE, Conflict, NoSpace, VideoError, Videos, check_video_name

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore

DRIVE_API = os.environ.get("GALLEY_DRIVE_API", "https://www.googleapis.com/drive/v3")
READ = 1 << 20
RETRIES = 6
FILE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{10,200}$")


class DriveConfig(BaseModel):
    client_id: str = Field(pattern=r"^[0-9]+-[a-z0-9]+\.apps\.googleusercontent\.com$")
    api_key: str = Field(pattern=r"^[A-Za-z0-9_\-]{20,64}$")
    app_id: str = Field(pattern=r"^[0-9]{6,20}$")       # the Cloud project number


class ImportReq(BaseModel):
    file_id: str = Field(pattern=FILE_ID_RE.pattern)
    name: str
    token: str = Field(min_length=10, max_length=4096)
    overwrite: bool = False

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        return check_video_name(v)


class TokenReq(BaseModel):
    token: str = Field(min_length=10, max_length=4096)


class AuthError(RuntimeError):
    """Google refused the token (expired, revoked, or no access to this file)."""


class Drive:
    def __init__(self, s: Settings, videos: Videos):
        self.s, self.v = s, videos
        self.items: dict[str, dict] = {}
        self._tokens: dict[str, str] = {}
        self._cancel: set[str] = set()
        self._q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._load_interrupted()

    # ── credentials ──────────────────────────────────────────────────────────
    @property
    def config_path(self) -> Path:
        return self.s.data_dir / "google.toml"

    def config(self) -> dict:
        cfg = {}
        if self.config_path.is_file():
            try:
                cfg = tomllib.loads(self.config_path.read_text()).get("google", {})
            except (OSError, ValueError):
                cfg = {}
        for k in ("client_id", "api_key", "app_id"):
            env = os.environ.get(f"GALLEY_GOOGLE_{k.upper()}")
            if env:
                cfg[k] = env
        ok = all(cfg.get(k) for k in ("client_id", "api_key", "app_id"))
        return {"enabled": ok, "client_id": cfg.get("client_id", ""), "api_key": cfg.get("api_key", ""),
                "app_id": cfg.get("app_id", ""), "scope": "https://www.googleapis.com/auth/drive.file"}

    def set_config(self, c: DriveConfig) -> dict:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.config_path.with_suffix(".tmp")
        tmp.write_text(f'[google]\nclient_id = "{c.client_id}"\napi_key = "{c.api_key}"\napp_id = "{c.app_id}"\n')
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.config_path)
        return self.config()

    # ── imports ──────────────────────────────────────────────────────────────
    def _paths(self, iid: str) -> tuple[Path, Path]:
        return self.v.up / f"{iid}.part", self.v.up / f"{iid}.json"

    def _load_interrupted(self) -> None:
        if not self.v.up.is_dir():
            return
        for m in self.v.up.glob("g*.json"):
            try:
                meta = json.loads(m.read_text())
            except (OSError, ValueError):
                continue
            if meta.get("kind") != "drive":
                continue
            part = self.v.up / f"{m.stem}.part"
            self.items[m.stem] = {**meta, "id": m.stem, "received": part.stat().st_size if part.exists() else 0,
                                  "state": "interrupted", "error": "Galley restarted; Resume signs in again and continues",
                                  "rate": 0.0, "started": None, "finished": None}

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(it) for it in sorted(self.items.values(), key=lambda x: x.get("created", 0))]

    def _get(self, url: str, token: str, headers: dict | None = None, timeout: float = 60):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", **(headers or {})})
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise AuthError(_google_msg(e) or "Google refused access")
            if e.code == 404:
                raise FileNotFoundError("Google Drive has no such file, or Galley may not read it (pick it in the Drive window)")
            raise

    def metadata(self, file_id: str, token: str) -> dict:
        q = urllib.parse.urlencode({"fields": "id,name,size,mimeType,md5Checksum,modifiedTime", "supportsAllDrives": "true"})
        with self._get(f"{DRIVE_API}/files/{file_id}?{q}", token, timeout=30) as r:
            return json.loads(r.read())

    def start(self, req: ImportReq) -> dict:
        try:
            meta = self.metadata(req.file_id, req.token)
        except AuthError as e:
            raise VideoError(f"Google refused the request: {e}. Sign in again.")
        except urllib.error.URLError as e:
            raise VideoError(f"cannot reach Google Drive from this host: {getattr(e, 'reason', e)}")
        mime, size = meta.get("mimeType", ""), int(meta.get("size") or 0)
        generic = mime in ("application/octet-stream", "") and req.name.lower().endswith((".mov", ".mp4", ".m4v", ".mkv", ".avi", ".webm", ".mts"))
        if not mime.startswith("video/") and not generic:
            raise VideoError(f"{meta.get('name')} is not a video ({mime})")
        if size <= 0:
            raise VideoError(f"{meta.get('name')} has no downloadable content (Google Docs files cannot be imported)")
        final = self.v.dir / req.name
        if final.exists() and not req.overwrite:
            if final.stat().st_size == size:
                return {"id": None, "name": req.name, "size": size, "received": size, "state": "exists"}
            raise Conflict(f"{req.name} exists already in video_captures/ with another size; import again choosing Replace", exists=True)
        iid = "g" + hashlib.sha256(req.file_id.encode()).hexdigest()[:19]   # never a browser-upload id (hex)
        with self._lock:
            cur = self.items.get(iid)
            if cur and cur["state"] in ("queued", "downloading", "verifying"):
                return dict(cur)
        for it in self.list():
            if it["name"] == req.name and it["id"] != iid and it["state"] in ("queued", "downloading", "verifying"):
                raise Conflict(f"another import is already writing {req.name}")
        self.v.dir.mkdir(parents=True, exist_ok=True)
        self.v.up.mkdir(exist_ok=True)
        part, mp = self._paths(iid)
        have = part.stat().st_size if part.exists() and mp.exists() else 0
        free = self.v.free_bytes()
        if free is not None and free - (size - have) < RESERVE:
            raise NoSpace(f"not enough disk space: {free / 2**30:.1f} GB free, the import needs {(size - have) / 2**30:.1f} GB more "
                          f"and {RESERVE / 2**30:.0f} GB stays free for the pipeline")
        record = {"kind": "drive", "file_id": req.file_id, "name": req.name, "drive_name": meta.get("name"),
                  "size": size, "md5": meta.get("md5Checksum"), "overwrite": req.overwrite, "created": time.time()}
        if not mp.exists():
            part.write_bytes(b"")
        mp.write_text(json.dumps(record))
        with self._lock:
            self.items[iid] = {**record, "id": iid, "received": have, "state": "queued", "error": None,
                               "rate": 0.0, "started": None, "finished": None}
            self._tokens[iid] = req.token
            self._cancel.discard(iid)
        self._enqueue(iid)
        return dict(self.items[iid])

    def resume(self, iid: str, token: str) -> dict:
        with self._lock:
            it = self.items.get(iid)
            if not it:
                raise FileNotFoundError(f"import {iid}")
            if it["state"] in ("queued", "downloading", "verifying"):
                return dict(it)
            if it["state"] == "done":
                raise Conflict("this import has finished")
            self._tokens[iid] = token
            self._cancel.discard(iid)
            it.update(state="queued", error=None)
        self._enqueue(iid)
        return dict(self.items[iid])

    def cancel(self, iid: str) -> dict:
        with self._lock:
            it = self.items.get(iid)
            if not it:
                raise FileNotFoundError(f"import {iid}")
            running = it["state"] in ("downloading", "verifying")
            self._cancel.add(iid)
            if not running:
                self._discard_files(iid)
                it.update(state="cancelled", rate=0.0)
            self._tokens.pop(iid, None)
            return dict(it)

    def dismiss(self, iid: str) -> None:
        with self._lock:
            it = self.items.get(iid)
            if not it:
                raise FileNotFoundError(f"import {iid}")
            if it["state"] in ("queued", "downloading", "verifying"):
                raise Conflict("cancel the import first")
            if it["state"] in ("error", "interrupted"):
                self._discard_files(iid)
            del self.items[iid]

    def _discard_files(self, iid: str) -> None:
        part, mp = self._paths(iid)
        part.unlink(missing_ok=True)
        mp.unlink(missing_ok=True)

    # ── worker ───────────────────────────────────────────────────────────────
    def _enqueue(self, iid: str) -> None:
        self._q.put(iid)
        if not self._worker or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._loop, name="galley-drive", daemon=True)
            self._worker.start()

    def _loop(self) -> None:
        while True:
            iid = self._q.get()
            with self._lock:
                it = self.items.get(iid)
                if not it or it["state"] != "queued" or iid in self._cancel:
                    continue
                it.update(state="downloading", started=time.time(), rate=0.0)
            try:
                self._download(iid)
            except AuthError as e:
                self._set(iid, state="error", rate=0.0, error=f"Google sign-in expired or access was refused ({e}). Resume signs in again.")
            except Cancelled:
                with self._lock:
                    self._discard_files(iid)
                    self.items[iid].update(state="cancelled", rate=0.0)
            except Exception as e:   # network, disk, checksum
                self._set(iid, state="error", rate=0.0, error=str(e) or e.__class__.__name__)
            finally:
                with self._lock:                            # keep a token no longer than its import runs
                    if self.items.get(iid, {}).get("state") != "queued":
                        self._tokens.pop(iid, None)

    def _set(self, iid: str, **kv) -> None:
        with self._lock:
            if iid in self.items:
                self.items[iid].update(kv)

    def _download(self, iid: str) -> None:
        it = self.items[iid]
        part, mp = self._paths(iid)
        size, url = it["size"], f"{DRIVE_API}/files/{it['file_id']}?alt=media&supportsAllDrives=true"
        fails = 0
        while True:
            have = part.stat().st_size if part.exists() else 0
            if have > size:
                part.write_bytes(b""); have = 0
            if have == size:
                break
            token = self._tokens.get(iid)
            if not token:
                raise AuthError("no sign-in token")
            try:
                r = self._get(url, token, {"Range": f"bytes={have}-"} if have else None)
                with r, part.open("r+b" if have else "wb") as f:
                    if have and r.status != 206:          # the server ignored Range: start over
                        f.truncate(0); have = 0
                    f.seek(have)
                    t0, b0 = time.monotonic(), have
                    while True:
                        if iid in self._cancel:
                            raise Cancelled()
                        buf = r.read1(READ)                  # whatever has arrived, up to 1 MiB
                        if not buf:
                            break
                        f.write(buf)
                        have += len(buf)
                        dt = time.monotonic() - t0
                        if dt >= 1.0:
                            rate = (have - b0) / dt
                            old = self.items[iid]["rate"]
                            self._set(iid, received=have, rate=rate if not old else old * 0.6 + rate * 0.4)
                            t0, b0 = time.monotonic(), have
                    self._set(iid, received=have)
                fails = 0
                if have < size:
                    raise ConnectionError(f"connection closed at {have} of {size} bytes")
            except (AuthError, Cancelled, FileNotFoundError):
                raise
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                fails += 1
                if fails > RETRIES:
                    raise RuntimeError(f"download kept failing ({getattr(e, 'reason', e)}); Resume continues from "
                                       f"{part.stat().st_size >> 20} MB")
                self._set(iid, rate=0.0, error=f"retrying after: {getattr(e, 'reason', e)} ({fails}/{RETRIES})")
                time.sleep(min(30, 2 ** fails))
                continue
        self._set(iid, state="verifying", received=size, rate=0.0, error=None)
        if it.get("md5"):
            h = hashlib.md5()
            with part.open("rb") as f:
                for b in iter(lambda: f.read(8 << 20), b""):
                    if iid in self._cancel:
                        raise Cancelled()
                    h.update(b)
            if h.hexdigest() != it["md5"]:
                part.unlink(missing_ok=True)
                raise RuntimeError("MD5 does not match Google Drive's checksum; the partial file was deleted, import again")
        final = self.v.dir / it["name"]
        if final.exists() and not it.get("overwrite"):
            raise RuntimeError(f"{it['name']} appeared in video_captures/ meanwhile; import again choosing Replace")
        os.replace(part, final)
        mp.unlink(missing_ok=True)
        self._set(iid, state="done", finished=time.time(), error=None)


class Cancelled(Exception):
    pass


def _google_msg(e: urllib.error.HTTPError) -> str | None:
    try:
        return json.loads(e.read().decode())["error"]["message"]
    except Exception:
        return None
