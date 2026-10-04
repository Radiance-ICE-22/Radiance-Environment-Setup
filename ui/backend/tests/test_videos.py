"""Video staging: resumable chunked uploads, probe, delete."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess

import pytest

CHUNK = 1000


def _upload(client, name, data, modified=1, chunk=CHUNK, overwrite=False):
    r = client.post("/api/uploads", json={"name": name, "size": len(data), "modified": modified, "overwrite": overwrite})
    assert r.status_code == 200, r.text
    up = r.json()
    if up["status"] == "exists":
        return up
    off = up["offset"]
    while off < len(data):
        piece = data[off:off + chunk]
        r = client.put(f"/api/uploads/{up['id']}?offset={off}", content=piece,
                       headers={"x-chunk-sha256": hashlib.sha256(piece).hexdigest()})
        assert r.status_code == 200, r.text
        off = r.json()["offset"]
    r = client.post(f"/api/uploads/{up['id']}/complete")
    assert r.status_code == 200, r.text
    return r.json()


def test_upload_roundtrip(client, settings):
    data = bytes(range(256)) * 17                   # 4352 bytes, 5 chunks
    done = _upload(client, "GTN_lab_v1.MOV", data, modified=1_700_000_000_000)
    assert done["name"] == "GTN_lab_v1.MOV" and done["bytes"] == len(data)
    f = settings.video_dir / "GTN_lab_v1.MOV"
    assert f.read_bytes() == data
    assert int(f.stat().st_mtime) == 1_700_000_000          # phone timestamp kept
    names = [v["name"] for v in client.get("/api/videos").json()]
    assert names == ["GTN_lab_v1.MOV", "lab3.mp4"]            # .uploads/ never listed
    assert client.get("/api/uploads").json()["uploads"] == []
    # the staged video is accepted by a capture job
    r = client.post("/api/jobs/figs", json={"scene": "GTN_lab_v1", "video": "GTN_lab_v1.MOV", "stop_after": "aruco"})
    assert r.status_code == 201


def test_resume_after_interruption(client, settings):
    data = b"abcdefghij" * 300
    body = {"name": "room.mp4", "size": len(data), "modified": 42}
    up = client.post("/api/uploads", json=body).json()
    assert up["status"] == "started" and up["offset"] == 0
    assert client.put(f"/api/uploads/{up['id']}?offset=0", content=data[:CHUNK]).json()["offset"] == CHUNK
    # the tab closes; choosing the same file again resumes at the same offset with the same id
    again = client.post("/api/uploads", json=body).json()
    assert again == {**up, "offset": CHUNK, "status": "resumed"}
    listed = client.get("/api/uploads").json()["uploads"]
    assert [(u["name"], u["offset"]) for u in listed] == [("room.mp4", CHUNK)]
    # a wrong offset is refused with the server's offset, so the client can re-sync
    r = client.put(f"/api/uploads/{up['id']}?offset=0", content=data[:CHUNK])
    assert r.status_code == 409 and r.json()["detail"]["offset"] == CHUNK
    # complete before all bytes arrived is refused
    assert client.post(f"/api/uploads/{up['id']}/complete").status_code == 409
    _upload(client, "room.mp4", data, modified=42)
    assert (settings.video_dir / "room.mp4").read_bytes() == data


def test_bad_chunk_is_discarded(client, settings):
    data = b"x" * 2500
    up = client.post("/api/uploads", json={"name": "a.mp4", "size": len(data)}).json()
    r = client.put(f"/api/uploads/{up['id']}?offset=0", content=data[:CHUNK], headers={"x-chunk-sha256": "0" * 64})
    assert r.status_code == 400 and "checksum" in r.json()["detail"]
    part = settings.video_dir / ".uploads" / f"{up['id']}.part"
    assert part.stat().st_size == 0
    # past the declared size
    r = client.put(f"/api/uploads/{up['id']}?offset=0", content=b"y" * 3000)
    assert r.status_code == 413 and part.stat().st_size == 0
    # abort removes both files
    assert client.delete(f"/api/uploads/{up['id']}").status_code == 200
    assert not part.exists()
    assert client.delete(f"/api/uploads/{up['id']}").status_code == 404


@pytest.mark.parametrize("name", ["../x.mp4", ".hidden.mp4", "a/b.mp4", "notes.txt", "x.mp4.part", "", "a b.mp4"])
def test_bad_names(client, name):
    r = client.post("/api/uploads", json={"name": name, "size": 10})
    assert r.status_code in (400, 422)


def test_existing_name(client, settings):
    # same name and size: nothing to send
    r = client.post("/api/uploads", json={"name": "lab3.mp4", "size": 10})
    assert r.json()["status"] == "exists"
    # same name, other size: refused unless overwrite
    r = client.post("/api/uploads", json={"name": "lab3.mp4", "size": 20})
    assert r.status_code == 409 and r.json()["detail"]["exists"] is True
    _upload(client, "lab3.mp4", b"z" * 20, overwrite=True)
    assert (settings.video_dir / "lab3.mp4").read_bytes() == b"z" * 20


def test_disk_space_checked(client, monkeypatch):
    import galley.videos as v
    monkeypatch.setattr(v.Videos, "free_bytes", lambda self: 3 * 2**30)
    r = client.post("/api/uploads", json={"name": "big.mov", "size": 2 * 2**30})
    assert r.status_code == 507 and "disk space" in r.json()["detail"]


def test_partial_uploads_never_resolve_as_videos(client, settings):
    up = client.post("/api/uploads", json={"name": "p.mp4", "size": 100}).json()
    for video in (f".uploads/{up['id']}.part", ".uploads", "../video_captures/lab3.mp4"):
        r = client.post("/api/jobs/figs", json={"scene": "p", "video": video})
        assert r.status_code == 400, video


def test_delete_video(client, settings):
    assert client.delete("/api/videos/nope.mp4").status_code == 404
    assert client.delete("/api/videos/..%2Fx.mp4").status_code in (400, 404, 405)
    r = client.post("/api/jobs/figs", json={"scene": "lab3", "video": "lab3.mp4", "stop_after": "aruco"})
    jid = r.json()["id"]
    r = client.delete("/api/videos/lab3.mp4")
    if r.status_code == 409:                                  # job still queued/running
        client.post(f"/api/jobs/{jid}/cancel")
        for _ in range(100):
            if client.get(f"/api/jobs/{jid}").json()["status"] not in ("queued", "running"):
                break
            import time; time.sleep(0.05)
        r = client.delete("/api/videos/lab3.mp4")
    assert r.status_code == 200
    assert not (settings.video_dir / "lab3.mp4").exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_probe_real_video(client, settings):
    out = settings.video_dir / "tiny.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", str(out)], check=True)
    p = client.get("/api/videos/tiny.mp4/probe").json()
    assert (p["codec"], p["width"], p["height"], p["fps"], p["bit_depth"], p["hdr"]) == ("h264", 320, 240, 30.0, 8, False)
    assert p["frames"] == 30 and abs(p["duration"] - 1) < 0.1
    assert client.get("/api/videos/lab3.mp4/probe").status_code == 400     # 10 zero bytes
    assert client.get("/api/videos/missing.mp4/probe").status_code == 404


def test_summarize_rotation_and_hdr():
    from galley.videos import summarize
    raw = {"streams": [{"codec_type": "video", "codec_name": "hevc", "pix_fmt": "yuv420p10le", "width": 3840,
                        "height": 2160, "r_frame_rate": "30/1", "avg_frame_rate": "1523760/50789",
                        "nb_frames": "12698", "duration": "423.3", "color_transfer": "arib-std-b67",
                        "side_data_list": [{"rotation": -90}]},
                       {"codec_type": "audio"}],
           "format": {"size": "1328887222", "tags": {"com.apple.quicktime.model": "iPhone 15"}}}
    s = summarize(raw)
    assert (s["width"], s["height"], s["rotation"]) == (2160, 3840, -90)
    assert s["hdr"] and s["bit_depth"] == 10 and not s["vfr"] and s["audio"] and s["device"] == "iPhone 15"
    assert s["frames"] == 12698 and s["bytes"] == 1328887222
