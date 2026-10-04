"""Google Drive imports against a fake Drive API (metadata, alt=media with Range, tokens)."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

DATA = bytes(range(256)) * 4096 * 3          # 3 MiB
FILES = {
    "VIDEOFILE0001": {"name": "GTN lab.MOV", "mimeType": "video/quicktime", "data": DATA},
    "DOCUMENT00001": {"name": "notes", "mimeType": "application/vnd.google-apps.document", "data": b""},
    "TEXTFILE00001": {"name": "a.txt", "mimeType": "text/plain", "data": b"hello"},
}


class Fake:
    tokens = {"good-token-123"}
    drop_once = False          # close the first media response half-way
    expire_after = None        # media requests allowed before the token "expires"
    slow = 0.0
    media_requests: list = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _err(self, code, msg):
        body = json.dumps({"error": {"code": code, "message": msg}}).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        fid = u.path.rsplit("/", 1)[-1]
        tok = self.headers.get("Authorization", "").removeprefix("Bearer ")
        if tok not in Fake.tokens:
            return self._err(401, "Invalid Credentials")
        f = FILES.get(fid)
        if not f:
            return self._err(404, "File not found")
        if q.get("alt") != ["media"]:
            meta = {"id": fid, "name": f["name"], "mimeType": f["mimeType"]}
            if f["data"]:
                meta.update(size=str(len(f["data"])), md5Checksum=hashlib.md5(f["data"]).hexdigest())
            body = json.dumps(meta).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
            return self.wfile.write(body)
        Fake.media_requests.append(self.headers.get("Range"))
        if Fake.expire_after is not None and len(Fake.media_requests) > Fake.expire_after:
            return self._err(401, "Invalid Credentials")
        data, start = f["data"], 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        if Fake.drop_once:
            Fake.drop_once = False
            self.wfile.write(body[: len(body) // 2]); self.wfile.flush()
            self.connection.shutdown(2)
            return
        for i in range(0, len(body), 256 * 1024):
            self.wfile.write(body[i:i + 256 * 1024])
            if Fake.slow:
                time.sleep(Fake.slow)


@pytest.fixture
def fake(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    import galley.drive as d
    monkeypatch.setattr(d, "DRIVE_API", f"http://127.0.0.1:{srv.server_address[1]}/drive/v3")
    Fake.tokens = {"good-token-123"}; Fake.drop_once = False; Fake.expire_after = None; Fake.slow = 0.0
    Fake.media_requests = []
    yield Fake
    srv.shutdown()


def wait(client, iid, states=("done", "error", "cancelled"), timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        it = next(i for i in client.get("/api/drive/imports").json() if i["id"] == iid)
        if it["state"] in states:
            return it
        time.sleep(0.05)
    raise AssertionError(f"import stuck: {it}")


def test_config_roundtrip(client, settings):
    assert client.get("/api/drive/config").json()["enabled"] is False
    bad = client.put("/api/drive/config", json={"client_id": "x", "api_key": "y", "app_id": "z"})
    assert bad.status_code == 422
    good = {"client_id": "123456789012-abcdefghijklmnop.apps.googleusercontent.com",
            "api_key": "AIzaSyA1234567890abcdefghijklmnopqrstu", "app_id": "123456789012"}
    r = client.put("/api/drive/config", json=good).json()
    assert r["enabled"] and r["client_id"] == good["client_id"] and r["scope"].endswith("/drive.file")
    assert (settings.data_dir / "google.toml").stat().st_mode & 0o777 == 0o600


def test_import_and_verify(client, settings, fake):
    r = client.post("/api/drive/imports", json={"file_id": "VIDEOFILE0001", "name": "GTN_lab.MOV", "token": "good-token-123"})
    assert r.status_code == 200, r.text
    it = wait(client, r.json()["id"])
    assert it["state"] == "done", it
    assert (settings.video_dir / "GTN_lab.MOV").read_bytes() == DATA
    assert "token" not in json.dumps(client.get("/api/drive/imports").json())
    assert "GTN_lab.MOV" in [v["name"] for v in client.get("/api/videos").json()]
    # same file again: already there
    r = client.post("/api/drive/imports", json={"file_id": "VIDEOFILE0001", "name": "GTN_lab.MOV", "token": "good-token-123"})
    assert r.json()["state"] == "exists"


def test_dropped_connection_resumes_with_range(client, settings, fake, monkeypatch):
    import galley.drive as d
    monkeypatch.setattr(d.time, "sleep", lambda s: None)
    fake.drop_once = True
    r = client.post("/api/drive/imports", json={"file_id": "VIDEOFILE0001", "name": "a.mov", "token": "good-token-123"})
    it = wait(client, r.json()["id"])
    assert it["state"] == "done", it
    assert fake.media_requests[0] is None and fake.media_requests[1].startswith("bytes=")
    assert (settings.video_dir / "a.mov").read_bytes() == DATA


def test_token_expiry_then_resume(client, settings, fake):
    fake.drop_once = True
    fake.expire_after = 1                    # the second media request gets 401
    r = client.post("/api/drive/imports", json={"file_id": "VIDEOFILE0001", "name": "b.mov", "token": "good-token-123"})
    iid = r.json()["id"]
    it = wait(client, iid)
    assert it["state"] == "error" and "sign" in it["error"] and 0 < it["received"] < len(DATA)
    fake.expire_after = None
    fake.tokens = {"fresh-token-456"}
    assert client.post(f"/api/drive/imports/{iid}/resume", json={"token": "fresh-token-456"}).status_code == 200
    it = wait(client, iid)
    assert it["state"] == "done"
    assert fake.media_requests[-1].startswith("bytes=")
    assert (settings.video_dir / "b.mov").read_bytes() == DATA


def test_refusals(client, fake):
    base = {"name": "x.mov", "token": "good-token-123"}
    assert client.post("/api/drive/imports", json={**base, "file_id": "VIDEOFILE0001", "token": "wrong-token-1"}).status_code == 400
    assert client.post("/api/drive/imports", json={**base, "file_id": "DOCUMENT00001"}).status_code == 400
    assert client.post("/api/drive/imports", json={**base, "file_id": "TEXTFILE00001"}).status_code == 400
    assert client.post("/api/drive/imports", json={**base, "file_id": "MISSINGFILE01"}).status_code == 404
    assert client.post("/api/drive/imports", json={**base, "file_id": "../../etc"}).status_code == 422
    assert client.post("/api/drive/imports", json={**base, "file_id": "VIDEOFILE0001", "name": "../x.mov"}).status_code == 422


def test_cancel_running(client, settings, fake):
    fake.slow = 0.2
    r = client.post("/api/drive/imports", json={"file_id": "VIDEOFILE0001", "name": "c.mov", "token": "good-token-123"})
    iid = r.json()["id"]
    wait(client, iid, ("downloading",))
    client.post(f"/api/drive/imports/{iid}/cancel")
    it = wait(client, iid)
    assert it["state"] == "cancelled"
    assert not (settings.video_dir / "c.mov").exists()
    assert not list((settings.video_dir / ".uploads").glob(f"{iid}.*"))
    assert client.delete(f"/api/drive/imports/{iid}").status_code == 200


def test_restart_shows_interrupted(settings, fake):
    """A Galley killed mid-download leaves <id>.part + <id>.json; the next start lists it as interrupted."""
    from fastapi.testclient import TestClient
    from galley.app import create_app
    up = settings.video_dir / ".uploads"; up.mkdir()
    iid = "g" + hashlib.sha256(b"VIDEOFILE0001").hexdigest()[:19]
    (up / f"{iid}.part").write_bytes(DATA[:1000000])
    (up / f"{iid}.json").write_text(json.dumps({"kind": "drive", "file_id": "VIDEOFILE0001", "name": "d.mov", "size": len(DATA),
                                                "md5": hashlib.md5(DATA).hexdigest(), "overwrite": False, "created": 1}))
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/uploads").json()["uploads"] == []          # not mistaken for a browser upload
        it = next(i for i in c.get("/api/drive/imports").json() if i["id"] == iid)
        assert it["state"] == "interrupted" and it["received"] == 1000000
        assert c.post(f"/api/drive/imports/{iid}/resume", json={"token": "good-token-123"}).status_code == 200
        assert wait(c, iid)["state"] == "done"
        assert fake.media_requests == ["bytes=1000000-"]
        assert (settings.video_dir / "d.mov").read_bytes() == DATA
