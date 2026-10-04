"""Video staging: resumable uploads into video_captures/, plus listing, probing and deleting.

A browser on another machine (the laptop, through an SSH port forward or Tailscale) sends a
phone video in chunks. Each upload is a pair of files in <video_dir>/.uploads/:

    <id>.part   the bytes received so far (always a prefix of the file)
    <id>.json   name, size, the file's lastModified, overwrite flag, timestamps

The id is derived from (name, size, lastModified), so choosing the same file again after a
dropped connection, a closed tab or a server restart resumes from the bytes already on disk.
A chunk either lands whole or not at all (a failed or mismatched chunk is truncated away),
and only a complete file is renamed into video_captures/, on the same filesystem, so
figs_pipeline.py --video never sees a partial file. Hidden names never resolve as videos.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field

from .settings import Settings

VIDEO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,95}\.(mov|mp4|m4v|mkv|avi|webm|mts)$", re.I)
UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{20}$")
UP_DIR = ".uploads"
MAX_CHUNK = 64 * 2**20          # bytes per PUT; the browser sends 8 MiB
MAX_SIZE = 64 * 2**30           # a 4K phone video is ~1.3 GB for 7 minutes
RESERVE = 2 * 2**30             # leave this much free on the disk after the upload
STALE_S = 7 * 24 * 3600         # partial uploads untouched this long are removed at startup
LIVE_S = 60                     # a partial written to this recently belongs to a live upload


class VideoError(ValueError):
    """Bad request (400)."""


class Conflict(RuntimeError):
    """409: name taken, offset mismatch, upload busy, video in use."""
    def __init__(self, msg: str, **extra):
        super().__init__(msg)
        self.extra = extra


class NoSpace(RuntimeError):
    """507: not enough free disk."""


class TooLarge(RuntimeError):
    """413: chunk bigger than allowed or past the declared size."""


class UploadStart(BaseModel):
    name: str
    size: int = Field(gt=0, le=MAX_SIZE)
    modified: int = Field(0, ge=0)          # File.lastModified (ms); part of the resume key
    overwrite: bool = False


def check_video_name(name: str) -> str:
    if not VIDEO_RE.match(name or ""):
        raise VideoError("video names use letters, digits, '_', '-' and '.', start with a letter or "
                         "digit, and end in .mov/.mp4/.m4v/.mkv/.avi/.webm/.mts")
    return name


def _mb(n: int) -> int:
    return round(n / 2**20)


class Videos:
    def __init__(self, s: Settings):
        self.s = s
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._probe: dict[tuple, dict] = {}
        self.cleanup()

    @property
    def dir(self) -> Path:
        return self.s.video_dir

    @property
    def up(self) -> Path:
        return self.s.video_dir / UP_DIR

    # ── staged videos ─────────────────────────────────────────────────────────
    def list(self) -> list[dict]:
        if not self.dir.is_dir():
            return []
        out = []
        for p in sorted(self.dir.iterdir(), key=lambda p: p.name.lower()):
            if p.name.startswith(".") or not p.is_file():
                continue
            st = p.stat()
            out.append({"name": p.name, "mb": _mb(st.st_size), "bytes": st.st_size, "modified": st.st_mtime})
        return out

    def path(self, name: str) -> Path:
        check_video_name(name)
        p = self.dir / name
        if not p.is_file():
            raise FileNotFoundError(name)
        return p

    def delete(self, name: str, in_use: set[str]) -> None:
        p = self.path(name)
        if name in in_use:
            raise Conflict(f"{name} is the video of a queued or running job; cancel it first")
        p.unlink()

    def probe(self, name: str) -> dict:
        p = self.path(name)
        st = p.stat()
        key = (name, st.st_size, st.st_mtime_ns)
        if key not in self._probe:
            self._probe[key] = summarize(self._ffprobe(p))
        return self._probe[key]

    def _ffprobe(self, p: Path) -> dict:
        args = ["-v", "error", "-show_format", "-show_streams", "-of", "json", str(p)]
        exe = shutil.which("ffprobe")
        if exe:
            argv = [exe, *args]
        elif self.s.env_script.is_file():    # the kitchen env ships ffmpeg; the backend venv may not
            argv = ["bash", "-c", f'source "{self.s.env_script}" >/dev/null 2>&1; exec ffprobe "$@"', "galley-probe", *args]
        else:
            raise VideoError("ffprobe not found on this host")
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            raise VideoError("ffprobe timed out")
        except OSError as e:
            raise VideoError(f"ffprobe failed: {e}")
        if r.returncode != 0:
            raise VideoError(f"ffprobe could not read {p.name}: {r.stderr.strip()[-300:] or r.returncode}")
        return json.loads(r.stdout or "{}")

    # ── uploads ──────────────────────────────────────────────────────────────
    def _meta(self, uid: str) -> Path:
        if not UPLOAD_ID_RE.match(uid or ""):
            raise VideoError("bad upload id")
        return self.up / f"{uid}.json"

    def _part(self, uid: str) -> Path:
        return self.up / f"{uid}.part"

    def _read(self, uid: str) -> dict:
        m = self._meta(uid)
        if not m.is_file():
            raise FileNotFoundError(f"upload {uid}")
        meta = json.loads(m.read_text())
        part = self._part(uid)
        meta["offset"] = part.stat().st_size if part.exists() else 0
        meta["id"] = uid
        return meta

    def _lock(self, uid: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(uid, threading.Lock())

    def uploads(self) -> list[dict]:
        if not self.up.is_dir():
            return []
        out = []
        for m in sorted(self.up.glob("*.json")):
            try:
                out.append(self._read(m.stem))
            except (OSError, ValueError):
                continue
        return out

    def free_bytes(self) -> int | None:
        for d in (self.dir, self.dir.parent):
            if d.is_dir():
                return shutil.disk_usage(d).free
        return None

    def cleanup(self) -> None:
        if not self.up.is_dir():
            return
        now = time.time()
        for m in self.up.glob("*.json"):
            part = self._part(m.stem)
            last = max(m.stat().st_mtime, part.stat().st_mtime if part.exists() else 0)
            if now - last > STALE_S:
                part.unlink(missing_ok=True)
                m.unlink(missing_ok=True)
        for part in self.up.glob("*.part"):           # orphan without metadata
            if not (self.up / f"{part.stem}.json").exists():
                part.unlink(missing_ok=True)

    def start(self, req: UploadStart) -> dict:
        name = check_video_name(req.name)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.up.mkdir(exist_ok=True)
        final = self.dir / name
        if final.exists() and not req.overwrite:
            size = final.stat().st_size
            if size == req.size:
                # same name, same size: treat as already staged (re-dropping a file is a no-op)
                return {"id": None, "name": name, "size": size, "offset": size, "status": "exists"}
            raise Conflict(f"{name} exists already ({_mb(size)} MB, this file is {_mb(req.size)} MB). "
                           "Rename the file, or upload again choosing Replace.", exists=True)
        uid = hashlib.sha256(f"{name}\0{req.size}\0{req.modified}".encode()).hexdigest()[:20]
        meta_p, part = self._meta(uid), self._part(uid)
        # another, different file under the same name: drop its partial unless it is still being written
        for other in self.uploads():
            if other["name"] == name and other["id"] != uid:
                if time.time() - other.get("updated", 0) < LIVE_S:
                    raise Conflict(f"another upload of {name} is in progress (from another tab or browser)")
                self.abort(other["id"])
        offset = part.stat().st_size if part.exists() and meta_p.exists() else 0
        free = self.free_bytes()
        if free is not None and free - (req.size - offset) < RESERVE:
            raise NoSpace(f"not enough disk space: {free / 2**30:.1f} GB free under {self.dir}, the upload needs "
                          f"{(req.size - offset) / 2**30:.1f} GB more and {RESERVE / 2**30:.0f} GB stays free for the pipeline")
        now = time.time()
        if not meta_p.exists():
            part.write_bytes(b"")
            created = now
        else:
            created = json.loads(meta_p.read_text()).get("created", now)
        meta_p.write_text(json.dumps({"name": name, "size": req.size, "modified": req.modified,
                                      "overwrite": req.overwrite, "created": created, "updated": now}))
        return {"id": uid, "name": name, "size": req.size, "offset": offset,
                "status": "resumed" if offset else "started"}

    def write(self, uid: str, offset: int, chunks, sha256: str | None) -> dict:
        """Append one chunk at `offset`. `chunks` yields bytes (the request body).
        The whole chunk lands or none of it does."""
        meta = self._read(uid)
        lock = self._lock(uid)
        if not lock.acquire(blocking=False):
            raise Conflict("this upload is being written by another request", offset=meta["offset"])
        try:
            part = self._part(uid)
            cur = part.stat().st_size
            if offset != cur:
                raise Conflict(f"offset {offset} does not match the {cur} bytes received", offset=cur)
            h = hashlib.sha256() if sha256 else None
            n = 0
            try:
                with part.open("r+b") as f:
                    f.seek(cur)
                    for piece in chunks:
                        n += len(piece)
                        if n > MAX_CHUNK:
                            raise TooLarge(f"chunk larger than {MAX_CHUNK // 2**20} MiB")
                        if cur + n > meta["size"]:
                            raise TooLarge("chunk goes past the declared file size")
                        f.write(piece)
                        if h:
                            h.update(piece)
                    if h and h.hexdigest() != sha256.lower():
                        raise VideoError("chunk checksum mismatch (corrupted in transit); it was discarded, send it again")
                    f.flush()
            except BaseException:
                with part.open("r+b") as f:
                    f.truncate(cur)                      # never keep a partial or bad chunk
                raise
            meta_p = self._meta(uid)
            m = json.loads(meta_p.read_text())
            m["updated"] = time.time()
            meta_p.write_text(json.dumps(m))
            return {"id": uid, "offset": cur + n, "size": meta["size"]}
        finally:
            lock.release()

    def complete(self, uid: str) -> dict:
        meta = self._read(uid)
        if meta["offset"] != meta["size"]:
            raise Conflict(f"upload incomplete: {meta['offset']} of {meta['size']} bytes", offset=meta["offset"])
        final = self.dir / meta["name"]
        if final.exists() and not meta.get("overwrite"):
            raise Conflict(f"{meta['name']} appeared in video_captures/ meanwhile; upload again choosing Replace", exists=True)
        part = self._part(uid)
        os.replace(part, final)                          # same filesystem: atomic
        if meta.get("modified"):
            t = meta["modified"] / 1000
            os.utime(final, (t, t))                      # keep the phone's timestamp
        self._meta(uid).unlink(missing_ok=True)
        st = final.stat()
        return {"name": final.name, "mb": _mb(st.st_size), "bytes": st.st_size, "modified": st.st_mtime}

    def abort(self, uid: str) -> None:
        m = self._meta(uid)
        if not m.exists() and not self._part(uid).exists():
            raise FileNotFoundError(f"upload {uid}")
        self._part(uid).unlink(missing_ok=True)
        m.unlink(missing_ok=True)


def _fps(rate: str | None) -> float | None:
    try:
        a, b = (rate or "").split("/")
        return round(float(a) / float(b), 3) if float(b) else None
    except ValueError:
        return None


def summarize(raw: dict) -> dict:
    """The parts of ffprobe's output the capture page shows (see figs_pipeline.py step_probe)."""
    streams = raw.get("streams", [])
    fmt = raw.get("format", {})
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise VideoError("no video stream")
    rotation = None
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            rotation = int(float(sd["rotation"]))
    if rotation is None and (v.get("tags") or {}).get("rotate"):
        rotation = -int(float(v["tags"]["rotate"]))
    duration = v.get("duration") or fmt.get("duration")
    tags = fmt.get("tags") or {}
    pix = v.get("pix_fmt") or ""
    fps, avg = _fps(v.get("r_frame_rate")), _fps(v.get("avg_frame_rate"))
    w, h = v.get("width"), v.get("height")
    if rotation in (90, -90, 270, -270) and w and h:
        w, h = h, w                                      # as it is displayed
    return {
        "codec": v.get("codec_name"), "profile": v.get("profile"), "pix_fmt": pix,
        "width": w, "height": h, "rotation": rotation or 0,
        "fps": fps, "avg_fps": avg, "vfr": bool(fps and avg and abs(fps - avg) > 1.0),
        "duration": round(float(duration), 2) if duration else None,
        "frames": int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None,
        "bit_depth": 10 if "10" in pix else 12 if "12" in pix else 8,
        "hdr": v.get("color_transfer") in ("arib-std-b67", "smpte2084"),
        "color_transfer": v.get("color_transfer"),
        "audio": any(s.get("codec_type") == "audio" for s in streams),
        "device": tags.get("com.apple.quicktime.model") or tags.get("com.android.model"),
        "created": tags.get("creation_time") or tags.get("com.apple.quicktime.creationdate"),
        "bytes": int(fmt["size"]) if str(fmt.get("size", "")).isdigit() else None,
    }
