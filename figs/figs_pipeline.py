#!/usr/bin/env python3
"""
figs_pipeline.py — capture to simulated flight, in one resumable script.

Replaces the notebook. Same phases, but every step writes a completion marker, so an
interrupt or crash costs you only the step that was running. Re-run the identical command
and it picks up where it stopped.

    source ~/FYP-Radiance/figs/figs_env.sh          # REQUIRED, not just `conda activate`
    ./figs_pipeline.py --scene lab3 --video ~/FYP-Radiance/video_captures/foo.mp4 \
                       --marker-id 0 --marker-length 0.18 --course intellisense_loop

    ./figs_pipeline.py --scene lab3 --list-steps
    ./figs_pipeline.py --scene lab3 --redo aruco
    ./figs_pipeline.py --scene lab3 --from simulate
    ./figs_pipeline.py --scene lab3 --status

State lives in <PROJECT_ROOT>/.figs_pipeline_state/<scene>/. Delete a marker to redo a step.

Each step also records a fingerprint of the parameters it depended on. Change
--marker-length and the steps downstream of it are invalidated automatically rather than
silently reusing stale output.

Validated 2026-08-06/07 on intellisense05, RTX 2080 8 GB.
"""

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════════
#  Output helpers
# ════════════════════════════════════════════════════════════════════════════════

_C = {"g": "\033[32m", "y": "\033[33m", "r": "\033[31m", "b": "\033[1m", "d": "\033[2m",
      "x": "\033[0m"}
if not sys.stdout.isatty():
    _C = {k: "" for k in _C}


def section(t): print(f"\n{_C['b']}{'═' * 78}\n  {t}\n{'═' * 78}{_C['x']}")
def ok(m):      print(f"  {_C['g']}✔{_C['x']} {m}")
def warn(m):    print(f"  {_C['y']}!{_C['x']} {m}")
def fail(m):    print(f"  {_C['r']}✗{_C['x']} {m}")
def info(m):    print(f"    {_C['d']}{m}{_C['x']}")


class StepFailed(Exception):
    """Carries an actionable message rather than a bare traceback."""


def elapsed(t0):
    s = time.time() - t0
    return f"{int(s // 60)}m {s % 60:.0f}s" if s >= 60 else f"{s:.1f}s"


# ════════════════════════════════════════════════════════════════════════════════
#  Known failure modes — matched against captured subprocess output
# ════════════════════════════════════════════════════════════════════════════════

_DIAGNOSES = [
    (r"DataLoader worker.*(exited unexpectedly|killed by signal|segmentation fault)",
     "A worker process segfaulted during hloc feature extraction. Seen on this machine with "
     "OpenCV 5.0.0 under fork. Fix: set num_workers=0 in "
     "FiGS/Hierarchical-Localization/hloc/extract_features.py (line ~263) — the loader then "
     "runs in the main process and cannot fork-crash. Costs a few seconds on a 20s stage. "
     "Also seen when launched from a Jupyter kernel; a plain shell is more reliable."),

    (r"cannot open shared object file|ACADOS_SOURCE_DIR",
     "acados shared libraries are not on the loader path. The process was started without "
     "`source ~/FYP-Radiance/figs/figs_env.sh` — `conda activate kitchen` alone is not "
     "enough. LD_LIBRARY_PATH is read once at process start, so it cannot be fixed after "
     "launch: re-source and re-run."),

    (r"returned multiple configurations",
     "More than one trained model under gsplats/workspace/outputs/<scene>/. FiGS requires "
     "exactly one config.yml per scene and will not guess. Retraining leaves the old "
     "timestamped directory behind — move it to gsplats/workspace/_archive/."),

    (r"import_images\(\)[\s\S]{0,400}incompatible function arguments"
     r"|incompatible function arguments[\s\S]{0,400}import_images",
     "pycolmap/hloc API drift: newer pycolmap renamed import_images(image_list=) to "
     "image_names=. The `patch` step fixes this — run it, or re-run the whole pipeline."),

    (r"Mismatched number of aruco and sfm transforms",
     "ArUco alignment failed: fewer valid marker detections than --num-marked. Either "
     "--marker-id is wrong or the marker is too sparsely visible. Check the `aruco` step "
     "output and consider lowering --num-marked (at the cost of alignment quality)."),

    (r"CUDA out of memory|CUDA error: out of memory",
     "GPU OOM. Close ns-viewer first — it holds ~4.6 GB. If training still OOMs, retrain "
     "without repeating SfM: --from train --archive-old with --cache-images cpu and/or "
     "--downscale 4, or fewer Gaussians via "
     "--train-arg '--pipeline.model.stop-split-at 10000'."),

    (r"Could not reconstruct any model",
     "COLMAP found no consistent reconstruction. Almost always insufficient overlap or "
     "motion blur while filming. Re-shoot: walk slower, keep generous overlap."),

    (r"FATAL: torch changed",
     "STOP. Something moved torch off 2.1.2, silently invalidating tiny-cuda-nn and gsplat. "
     "Rebuild: install_figs.sh --prefix ~/FYP-Radiance/figs "
     "--redo conda_env --redo tcnn --redo pips"),

    (r"ValueError.*match|multiple.*match",
     "Ambiguous filename glob in gsplats/capture/ — more than one file contains the scene "
     "name as a substring. Keep exactly one unambiguous file per scene. Note 'foo' matches "
     "'foo_02.mp4', so pick disjoint names."),
]


def diagnose(text):
    for pattern, explanation in _DIAGNOSES:
        if re.search(pattern, text, re.I):
            return explanation
    return None


def sh(cmd, cwd=None, stream=True, check=True, env=None):
    """Run a command, streaming output. Returns (rc, combined_output)."""
    p = subprocess.Popen(cmd, cwd=cwd, shell=isinstance(cmd, str), env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    buf = []
    try:
        for line in p.stdout:
            buf.append(line)
            if stream:
                print(line, end="")
        rc = p.wait()
    except KeyboardInterrupt:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
        raise
    out = "".join(buf)
    if check and rc != 0:
        d = diagnose(out)
        raise StepFailed(f"{d}\n\n  last output:\n{out[-1200:]}" if d
                         else f"command failed (rc={rc}):\n{out[-1500:]}")
    return rc, out


# ════════════════════════════════════════════════════════════════════════════════
#  VRAM
# ════════════════════════════════════════════════════════════════════════════════

def vram_now():
    try:
        return int(subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0])
    except Exception:
        return -1


class VramMonitor:
    def __init__(self, interval=5.0):
        self.interval = interval
        self.baseline = vram_now()
        self.peak = self.baseline
        self._stop = threading.Event()

    def _loop(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, vram_now())

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=self.interval + 1)
        info(f"VRAM baseline {self.baseline} MiB → peak {self.peak} MiB")


# ════════════════════════════════════════════════════════════════════════════════
#  Context: paths, parameters, state
# ════════════════════════════════════════════════════════════════════════════════

def resolve_project_root(explicit=None):
    """
    Find PROJECT_ROOT without depending on where this script happens to live.

    Order of preference:
      1. --project-root
      2. $FIGS_PROJECT_ROOT
      3. derived from $ACADOS_SOURCE_DIR, which figs_env.sh always exports as
         <root>/SousVide/FiGS/acados  — the most reliable signal, because if it is not
         set the run cannot succeed anyway
      4. probe near this script: alongside it, one level down, or one level up
      5. the conventional path, as a last resort

    A valid root contains both figs_env.sh and SousVide/.
    """
    def valid(d):
        d = Path(d).expanduser()
        return d if (d / "figs_env.sh").exists() and (d / "SousVide").is_dir() else None

    if explicit:
        d = Path(explicit).expanduser()
        if not valid(d):
            warn(f"--project-root {d} does not look like a FiGS root "
                 "(expected figs_env.sh and SousVide/ inside)")
        return d

    env = os.environ.get("FIGS_PROJECT_ROOT")
    if env and valid(env):
        return valid(env)

    acados = os.environ.get("ACADOS_SOURCE_DIR")
    if acados:
        # <root>/SousVide/FiGS/acados  ->  <root>
        cand = Path(acados).resolve().parents[2]
        if valid(cand):
            return cand

    here = Path(__file__).resolve().parent
    for cand in (here, here / "figs", here.parent, here.parent / "figs"):
        if valid(cand):
            return cand

    return Path("~/FYP-Radiance/figs").expanduser()


class Ctx:
    def __init__(self, a):
        self.a = a
        self.project_root = resolve_project_root(a.project_root)
        self.repo = self.project_root / "SousVide"
        self.scene = a.scene
        self.raw_video = Path(a.video).expanduser() if a.video else None
        self.capture_mp4 = self.repo / "gsplats" / "capture" / f"{a.scene}.mp4"
        self.workspace = self.repo / "gsplats" / "workspace" / a.scene
        self.out_dir = self.repo / "gsplats" / "workspace" / "outputs" / a.scene
        self.state = self.project_root / ".figs_pipeline_state" / a.scene
        self.logs = self.project_root / ".figs_pipeline_logs" / a.scene
        self.state.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        # Rendered flights land in a dedicated directory rather than the repo root,
        # which is upstream SousVide and should stay free of our artefacts.
        self.flights = self.repo / "outputs" / "flights"
        self.flights.mkdir(parents=True, exist_ok=True)
        self.out_mp4 = self.flights / f"{a.scene}_flight.mp4"

        # One-time migration: earlier versions wrote to the repo root. If a completed
        # `simulate` marker exists but the file only lives at the old path, move it so a
        # resumed run does not re-render an 80-minute-old scene for want of a filename.
        legacy = self.repo / f"{a.scene}_flight.mp4"
        if legacy.exists() and not self.out_mp4.exists():
            legacy.rename(self.out_mp4)
            print(f"  moved {legacy.name} → {self.out_mp4.relative_to(self.repo)}")

        self.results = {"scene": a.scene, "started": datetime.now().isoformat(timespec="seconds")}
        self._load_results()

    # ---- results carry across invocations so a resumed run still writes a full record
    @property
    def _results_path(self):
        return self.state / "results.json"

    def _load_results(self):
        if self._results_path.exists():
            try:
                self.results.update(json.loads(self._results_path.read_text()))
            except Exception:
                pass

    def save_results(self):
        self._results_path.write_text(json.dumps(self.results, indent=2, default=str) + "\n")

    # ---- step state ------------------------------------------------------------
    def fingerprint(self, keys):
        """Hash of the parameters a step depends on. Changing one invalidates the step."""
        vals = {k: getattr(self.a, k, None) for k in keys}
        vals = {k: (str(v) if v is not None else None) for k, v in vals.items()}
        return hashlib.sha256(json.dumps(vals, sort_keys=True).encode()).hexdigest()[:16]

    def is_done(self, name, fp):
        m = self.state / f"{name}.done"
        if not m.exists():
            return False
        prev = m.read_text().strip().split("\n")[0]
        if prev != fp:
            warn(f"'{name}' was completed with different parameters — redoing")
            m.unlink()
            return False
        return True

    def mark_done(self, name, fp):
        (self.state / f"{name}.done").write_text(
            f"{fp}\n{datetime.now().isoformat(timespec='seconds')}\n")

    def clear(self, name):
        (self.state / f"{name}.done").unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
#  Steps
# ════════════════════════════════════════════════════════════════════════════════

def probe_video(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames,duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    s["fps"] = float(num) / float(den)
    return s


# ── preflight ────────────────────────────────────────────────────────────────────

def step_preflight(c):
    problems = []

    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    (ok if env == "kitchen" else problems.append)(
        f"conda env: {env}" if env == "kitchen"
        else f"conda env is '{env or 'none'}', expected 'kitchen'")

    # acados: the failure that otherwise surfaces two phases later
    acados = os.environ.get("ACADOS_SOURCE_DIR", "")
    if not acados:
        problems.append("ACADOS_SOURCE_DIR unset — run `source "
                        f"{c.project_root}/figs_env.sh` first. `conda activate` is not enough.")
    else:
        ok(f"ACADOS_SOURCE_DIR: {acados}")
        libdir = str(Path(acados) / "lib")
        if libdir not in os.environ.get("LD_LIBRARY_PATH", ""):
            problems.append(f"LD_LIBRARY_PATH missing {libdir} — re-source figs_env.sh")
        else:
            ok("LD_LIBRARY_PATH includes acados/lib")

    # load by SONAME, so we exercise the same search AcadosSimSolver will
    for so in ("libqpOASES_e.so", "libacados.so"):
        try:
            ctypes.CDLL(so)
            ok(f"{so} resolves")
        except OSError as e:
            problems.append(f"cannot load {so}: {e}")

    if not c.repo.exists():
        problems.append(f"repo not found: {c.repo}")
    else:
        ok(f"repo: {c.repo}")

    import importlib
    from importlib.metadata import version as dist_version, PackageNotFoundError

    def ver(mod, m):
        v = getattr(m, "__version__", None)
        if v:
            return str(v)
        for d in (mod, mod.replace("_", "-")):
            try:
                return dist_version(d)
            except PackageNotFoundError:
                pass
        return "?"

    for mod, pin in [("torch", "2.1.2"), ("nerfstudio", "1.1.4"), ("gsplat", None),
                     ("hloc", None), ("figs", None), ("acados_template", None),
                     ("pycolmap", None), ("cv2", None), ("open3d", None)]:
        try:
            m = importlib.import_module(mod)
            v = ver(mod, m)
            if pin and v == "?":
                warn(f"{mod} imported, version undeterminable (expected {pin})")
            elif pin and not v.startswith(pin):
                problems.append(f"{mod} is {v}, expected {pin} — a moved torch invalidates "
                                "tiny-cuda-nn and gsplat")
            else:
                ok(f"{mod} {v}")
        except Exception as e:
            problems.append(f"import {mod} failed: {e}")

    for exe in ("colmap", "ns-train", "ffmpeg", "ffprobe", "nvidia-smi"):
        (ok if shutil.which(exe) else problems.append)(
            f"{exe}" if shutil.which(exe) else f"{exe} not on PATH")

    import torch
    if not torch.cuda.is_available():
        problems.append("torch cannot see the GPU")
    else:
        total = torch.cuda.get_device_properties(0).total_memory // 1024 ** 2
        used = vram_now()
        ok(f"GPU: {torch.cuda.get_device_name(0)}  {total} MiB, {used} MiB in use")
        c.results["gpu"] = torch.cuda.get_device_name(0)
        if total < 6144 and c.a.cache_images != "cpu":
            warn("under 6 GB: train with --cache-images cpu. With it, backroom trained at the "
                 "default 960x540 in 2360 MiB on a 4 GB RTX 3050 Ti; with the images cached on the "
                 "GPU the RTX 2080 reference peaked at 5203 MiB.")
        if used > 1000:
            warn(f"{used} MiB already allocated. ns-viewer holds ~4.6 GB — close it before "
                 "training, or the peak measurement is meaningless.")

    if problems:
        print()
        for p in problems:
            fail(p)
        raise StepFailed(f"{len(problems)} preflight problem(s)")
    ok("preflight clean")


# ── probe ────────────────────────────────────────────────────────────────────────

def step_probe(c):
    if not c.raw_video or not c.raw_video.exists():
        raise StepFailed(f"--video not found: {c.raw_video}")
    s = probe_video(c.raw_video)
    info(f"codec      {s['codec_name']} ({s['pix_fmt']})")
    info(f"resolution {s['width']}x{s['height']} @ {s['fps']:.2f} fps")
    info(f"duration   {float(s['duration']):.1f}s, {s.get('nb_frames', '?')} frames")
    info(f"size       {c.raw_video.stat().st_size / 1024 ** 3:.2f} GB")
    c.results["source"] = {k: s.get(k) for k in
                           ("codec_name", "pix_fmt", "width", "height", "fps",
                            "nb_frames", "duration")}

    if s["pix_fmt"] not in ("yuv420p", "yuvj420p"):
        warn(f"{s['pix_fmt']} is not 8-bit 4:2:0 — likely HDR/10-bit. OpenCV may decode it "
             "slowly or with shifted colour. The transcode fixes this.")
    # VFR detection: nominal fps vs actual frames/duration
    try:
        actual = int(s["nb_frames"]) / float(s["duration"])
        if abs(actual - s["fps"]) > 1.0:
            warn(f"nominal {s['fps']:.2f} fps but {actual:.2f} implied — variable frame rate. "
                 "The transcode forces constant frame rate, which the ArUco time-sampling "
                 "relies on.")
    except Exception:
        pass

    need = (s["width"] != c.a.width or s["height"] != c.a.height
            or s["fps"] > c.a.fps + 1 or s["codec_name"] != "h264")
    (warn if need else ok)("transcode needed" if need else "already in target format")


# ── transcode ────────────────────────────────────────────────────────────────────

def step_transcode(c):
    c.capture_mp4.parent.mkdir(parents=True, exist_ok=True)

    rivals = [p for p in c.capture_mp4.parent.glob("*")
              if c.scene in p.name and p != c.capture_mp4]
    if rivals:
        raise StepFailed(
            f"filename collision — these also contain '{c.scene}': {[p.name for p in rivals]}. "
            "FiGS globs by substring and raises on multiple matches. Rename to disjoint names "
            "(note 'foo' matches 'foo_02.mp4').")

    if c.capture_mp4.exists():
        d = probe_video(c.capture_mp4)
        ok(f"staged: {c.capture_mp4.name} ({d['width']}x{d['height']} @ {d['fps']:.1f}fps, "
           f"{c.capture_mp4.stat().st_size / 1024 ** 2:.0f} MB)")
    else:
        t0 = time.time()
        tmp = c.capture_mp4.with_suffix(".mp4.partial")
        warn("transcoding — HEVC/4K decode is the bottleneck, expect several minutes")
        try:
            sh(["ffmpeg", "-y", "-i", str(c.raw_video),
                "-vf", f"scale={c.a.width}:{c.a.height},format=yuv420p",
                "-r", str(c.a.fps), "-c:v", "libx264", "-crf", "18",
                "-preset", "medium", "-an", "-f", "mp4", str(tmp)])
        except BaseException:
            tmp.unlink(missing_ok=True)      # never leave a truncated file behind
            raise
        tmp.rename(c.capture_mp4)            # atomic: only a complete file gets the real name
        d = probe_video(c.capture_mp4)
        ok(f"wrote {c.capture_mp4.name} in {elapsed(t0)} "
           f"({c.capture_mp4.stat().st_size / 1024 ** 2:.0f} MB)")

    c.results["capture"] = {"width": d["width"], "height": d["height"], "fps": d["fps"]}


# ── aruco ────────────────────────────────────────────────────────────────────────

def step_aruco(c):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(c.capture_mp4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters())
    stride, n, hits, all_ids = 2, 0, [], {}
    t0 = time.time()
    while True:
        okf, frame = cap.read()
        if not okf:
            break
        n += 1
        if n % stride:
            continue
        corners, ids, _ = det.detectMarkers(frame)
        if ids is None:
            continue
        for k, i in enumerate(ids.flatten()):
            all_ids[int(i)] = all_ids.get(int(i), 0) + 1
            if int(i) == c.a.marker_id:
                px = float(cv2.contourArea(
                    corners[k].reshape(-1, 2).astype(np.float32))) ** 0.5
                hits.append((n / fps, px))
    cap.release()

    dur = n / fps
    info(f"scanned {n} frames (every {stride}nd) in {elapsed(t0)}")
    info(f"all IDs seen: {all_ids}")
    spurious = {k: v for k, v in all_ids.items() if k != c.a.marker_id}
    if spurious:
        info(f"ignoring {spurious} — low counts are ArUco false positives")

    print(f"\n  {'window':>12}  {'hits':>5}  {'median px':>9}")
    windows, hist = 0, []
    for lo in range(0, int(dur) + 1, 10):
        w = [(t, s) for t, s in hits if lo <= t < lo + 10]
        windows += bool(w)
        hist.append({"t0": lo, "hits": len(w) * stride,
                     "median_px": round(float(np.median([s for _, s in w])), 1) if w else None})
        print(f"  {lo:4d}-{lo + 10:3d}s  {len(w):5d}  "
              f"{np.median([s for _, s in w]) if w else 0:9.0f}  {'#' * min(50, len(w))}")
    print()

    est = len(hits) * stride
    med = float(np.median([s for _, s in hits])) if hits else 0.0
    c.results["aruco"] = {"marker_id": c.a.marker_id, "marker_length_m": c.a.marker_length,
                          "est_marked_frames": est, "median_px": round(med, 1),
                          "windows_10s": windows, "spurious_ids": spurious,
                          "duration_s": round(dur, 1), "histogram": hist}

    if c.a.marker_id not in all_ids:
        raise StepFailed(
            f"marker id {c.a.marker_id} never detected. Either the printed marker is from a "
            f"different dictionary, or it was never clearly in shot. IDs seen: {all_ids}. "
            "No config change fixes this — it is a re-shoot.")
    ok(f"marker {c.a.marker_id} in ~{est} frames (need {c.a.num_marked})")

    if est < c.a.num_marked * 3:
        warn(f"thin pool ({est} vs {c.a.num_marked} needed) — FiGS subsamples across time and "
             "may not find enough")
    if windows < 3:
        warn(f"detections span only {windows} 10s window(s) — short alignment baseline, "
             "noise-sensitive scale")
    else:
        ok(f"spread across {windows} separate 10s windows")
    (warn if med < 40 else ok)(
        f"median apparent size {med:.0f} px" + (" — small, imprecise corners" if med < 40 else ""))


# ── config ───────────────────────────────────────────────────────────────────────

def step_config(c):
    p = c.repo / "configs" / "captures" / f"{c.scene}.json"
    cfg = {"camera": None,
           "extractor": {"num_images": c.a.num_images, "num_marked": c.a.num_marked,
                         "marker_length": c.a.marker_length, "marker_id": c.a.marker_id}}
    p.write_text(json.dumps(cfg, indent=4) + "\n")
    for line in p.read_text().splitlines():
        info(line)
    ok(f"wrote {p.relative_to(c.repo)}")
    warn(f"marker_length = {c.a.marker_length} m must be the BLACK SQUARE's side, not the "
         "paper or the white quiet zone. Nothing downstream can detect an error here.")
    c.results["capture_cfg"] = cfg["extractor"]


# ── patch ────────────────────────────────────────────────────────────────────────

def step_patch(c):
    """Re-apply the two submodule fixes. Idempotent; safe to run every time."""
    import pycolmap
    info(f"pycolmap {pycolmap.__version__}")
    hloc = c.repo / "FiGS" / "Hierarchical-Localization" / "hloc"

    # 1. import_images kwarg rename
    recon = hloc / "reconstruction.py"
    src = recon.read_text()
    wants_names = "image_names" in (pycolmap.import_images.__doc__ or "")
    if wants_names and "image_list=image_list" in src:
        bak = recon.with_suffix(".py.bak")
        if not bak.exists():
            shutil.copy(recon, bak)
        recon.write_text(src.replace("image_list=image_list", "image_names=image_list"))
        ok("patched reconstruction.py: import_images image_list → image_names")
    elif wants_names:
        ok("reconstruction.py already patched")
    else:
        ok("pycolmap uses the image_list API — no patch needed")

    # 2. DataLoader fork segfault (OpenCV 5.x)
    ef = hloc / "extract_features.py"
    src = ef.read_text()
    if c.a.dataloader_workers is not None:
        want = f"num_workers={c.a.dataloader_workers}"
        m = re.search(r"num_workers=(\d+)", src)
        if m and m.group(0) != want:
            bak = ef.with_suffix(".py.bak")
            if not bak.exists():
                shutil.copy(ef, bak)
            ef.write_text(src.replace(m.group(0), want, 1))
            ok(f"patched extract_features.py: {m.group(0)} → {want}")
        elif m:
            ok(f"extract_features.py already {want}")

    warn("both files are git submodules — `install_figs.sh --redo clone` reverts them. "
         "This step re-applies on every run; .bak files are kept.")
    c.results["pycolmap_version"] = pycolmap.__version__


# ── sfm ────────────────────────────────────────────────────────────────────────
# FiGS's generate_gsplat() does frame extraction → hloc → COLMAP → ArUco alignment and
# then shells out to `ns-train` with hard-coded flags. `sfm` runs it with that one
# subprocess call intercepted, so SfM output is produced exactly as upstream does it;
# `train` then runs ns-train itself, with the training options exposed. Splitting them
# means a retrain (new downscale, fewer iterations, a 4 GB card) never repeats SfM.

_SFM_SCRIPT = r"""
import subprocess, sys
import figs.render.capture_generation as pg

class _Skip:
    returncode, stdout, stderr = 0, "", ""

def _run(cmd, *a, **kw):
    if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "ns-train":
        print("[figs_pipeline] ns-train deferred to the `train` step", flush=True)
        return _Skip()
    return subprocess.run(cmd, *a, **kw)

class _SubprocessShim:
    def __getattr__(self, name):
        return _run if name == "run" else getattr(subprocess, name)

pg.subprocess = _SubprocessShim()
pg.generate_gsplat(sys.argv[1], capture_cfg_name=sys.argv[1])
"""


def step_sfm(c):
    log = c.logs / f"sfm_{datetime.now():%Y-%m-%d_%H%M}.log"
    info(f"log: {log}")
    info("frame extraction → hloc features → SuperGlue matching → COLMAP → ArUco alignment. "
         "Reference (300 images): ~45 min, dominated by SuperGlue matching.")
    info("hloc caches features/matches to h5, so an interrupt here resumes cheaply.")
    n = c.a.num_images or 0
    if n:
        info(f"{n} images → {n * (n - 1) // 2:,} exhaustive match pairs")
    t0 = time.time()
    with VramMonitor() as vm:
        rc, out = sh([sys.executable, "-c", _SFM_SCRIPT, c.scene], cwd=c.repo, check=False)
    log.write_text(out)
    if rc != 0:
        d = diagnose(out)
        raise StepFailed(f"{d}\n\n  full log: {log}" if d
                         else f"generate_gsplat (SfM part) failed (rc={rc}); see {log}")
    if "ns-train deferred" not in out:
        raise StepFailed("generate_gsplat finished without reaching its ns-train call — the "
                         f"upstream function may have changed; see {log}")
    for f in ("transforms.json", "sparse_pc.ply"):
        if not (c.workspace / f).exists():
            raise StepFailed(f"SfM finished but {c.workspace / f} is missing; see {log}")
    c.results["sfm_run"] = {"wallclock": elapsed(t0), "peak_vram_mib": vm.peak}
    ok(f"SfM + alignment completed in {elapsed(t0)}, peak VRAM {vm.peak} MiB")


# ── train ────────────────────────────────────────────────────────────────────────

def _archive_models(c, configs):
    arch = c.repo / "gsplats" / "workspace" / "_archive" / c.scene
    arch.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        run = cfg.parent
        dest = arch / run.name
        shutil.move(str(run), str(dest))
        ok(f"archived {run.relative_to(c.out_dir)} → {dest.relative_to(c.repo)}")


def train_command(c):
    """The upstream ns-train invocation from generate_gsplat(), plus the exposed options."""
    cmd = ["ns-train", "splatfacto",
           "--data", c.scene,
           "--viewer.quit-on-train-completion", "True",
           "--output-dir", "outputs",
           "--pipeline.model.camera-optimizer.mode", "SO3xR3"]
    if c.a.train_iters:
        cmd += ["--max-num-iterations", str(c.a.train_iters)]
    if c.a.cache_images:
        cmd += ["--pipeline.datamanager.cache-images", c.a.cache_images]
    if c.a.train_vis:
        cmd += ["--vis", c.a.train_vis]
    for extra in c.a.train_arg:
        cmd += extra.split(maxsplit=1) if extra.startswith("--") and " " in extra else [extra]
    cmd += ["nerfstudio-data",
            "--orientation-method", "none",
            "--center-method", "none",
            "--auto-scale-poses", "False"]
    if c.a.downscale:
        cmd += ["--downscale-factor", str(c.a.downscale)]
    return cmd


def step_train(c):
    for f in ("transforms.json", "sparse_pc.ply"):
        if not (c.workspace / f).exists():
            raise StepFailed(f"{c.workspace / f} missing — run the `sfm` step first")
    existing = sorted(c.out_dir.rglob("config.yml")) if c.out_dir.exists() else []
    if existing:
        if not c.a.archive_old:
            listing = "\n      ".join(str(p.parent.relative_to(c.out_dir)) for p in existing)
            raise StepFailed(
                f"outputs/{c.scene}/ already holds {len(existing)} trained model(s):\n      {listing}\n\n"
                "FiGS loads exactly one per scene, so training another would make the scene "
                "unloadable. Re-run with --archive-old to move the existing model(s) to "
                "gsplats/workspace/_archive/<scene>/ first.")
        _archive_models(c, existing)
    if vram_now() > 1000:
        warn(f"{vram_now()} MiB allocated — close ns-viewer (holds ~4.6 GB) before training")

    cmd = train_command(c)
    log = c.logs / f"train_{datetime.now():%Y-%m-%d_%H%M}.log"
    info(f"log: {log}")
    info("cmd: " + " ".join(cmd))
    info("Reference: 30k steps in 34 min, peak 5203 MiB at 960x540 on an RTX 2080 (8 GB).")
    t0 = time.time()
    with VramMonitor() as vm:
        rc, out = sh(cmd, cwd=c.repo / "gsplats" / "workspace", check=False)
    log.write_text(out)
    c.results["train"] = {"cmd": cmd[2:], "wallclock": elapsed(t0), "peak_vram_mib": vm.peak,
                          "rc": rc, "iters": c.a.train_iters, "downscale": c.a.downscale,
                          "cache_images": c.a.cache_images}
    c.results["train_peak_vram_mib"] = vm.peak
    if rc != 0:
        d = diagnose(out)
        raise StepFailed(f"{d}\n\n  full log: {log}" if d else f"ns-train failed (rc={rc}); see {log}")
    ok(f"trained in {elapsed(t0)}, peak VRAM {vm.peak} MiB")


# ── verify ───────────────────────────────────────────────────────────────────────

def step_verify(c):
    import open3d as o3d

    configs = sorted(c.out_dir.rglob("config.yml")) if c.out_dir.exists() else []
    if len(configs) > 1:
        listing = "\n      ".join(str(p.parent.relative_to(c.out_dir)) for p in configs)
        raise StepFailed(
            f"{len(configs)} trained models under outputs/{c.scene}/ — FiGS requires exactly "
            f"one and will not guess:\n      {listing}\n\n"
            f"  mkdir -p {c.repo}/gsplats/workspace/_archive\n"
            f"  mv {c.out_dir}/splatfacto/<timestamp> {c.repo}/gsplats/workspace/_archive/")
    if not configs:
        raise StepFailed(f"no config.yml under outputs/{c.scene}/ — training did not complete")

    n_images = len(list((c.workspace / "images").glob("*")))
    tf = json.loads((c.workspace / "transforms.json").read_text())
    n_reg = len(tf["frames"])
    n_pts = len(o3d.io.read_point_cloud(str(c.workspace / "sparse_pc.ply")).points)
    ckpt = sorted(c.out_dir.rglob("*.ckpt"))[-1]
    mb = ckpt.stat().st_size / 1024 ** 2
    pct = 100 * n_reg / max(n_images, 1)

    info(f"extracted images  {n_images}")
    info(f"registered poses  {n_reg} ({pct:.1f}%)")
    info(f"sparse points     {n_pts}")
    info(f"checkpoint        {mb:.0f} MB")
    c.results["sfm"] = {"images": n_images, "registered": n_reg, "pct": round(pct, 1),
                        "sparse_points": n_pts, "ckpt_mb": round(mb),
                        "ckpt": str(ckpt.relative_to(c.repo))}

    if pct < 90:
        raise StepFailed(
            f"only {pct:.1f}% registered. Below ~90% means insufficient overlap or motion "
            "blur while filming. Everything downstream inherits the holes — re-shoot.")
    ok(f"{pct:.1f}% registered")
    (warn if n_pts < 5000 else ok)(f"{n_pts} sparse points")
    ok(f"exactly one trained model: {configs[0].parent.name}")


# ── bounds ───────────────────────────────────────────────────────────────────────

def _bounds(c):
    import numpy as np
    tf = json.loads((c.workspace / "transforms.json").read_text())
    P = np.array([f["transform_matrix"] for f in tf["frames"]])[:, :3, 3]
    return P, P.min(axis=0), P.max(axis=0)


def splat_to_course(p):
    import numpy as np
    return np.array([p[0], -p[1], -p[2]])


course_to_splat = splat_to_course      # 180° rotation about x — self-inverse


def step_bounds(c):
    import numpy as np
    P, lo, hi = _bounds(c)
    path = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    print("  splat frame (z-up, metres):")
    for i, ax in enumerate("xyz"):
        print(f"    {ax}: {lo[i]:+6.2f} .. {hi[i]:+6.2f}   "
              f"(span {hi[i] - lo[i]:5.2f}, median {np.median(P[:, i]):+6.2f})")

    dur = c.results.get("source", {}).get("duration")
    if dur:
        info(f"camera path {path:.1f} m over {float(dur):.0f}s → {path / float(dur):.2f} m/s")
        if path / float(dur) > 1.5:
            warn("faster than ~1.5 m/s — slower walking is the first thing to change if a "
                 "future capture reconstructs poorly")

    c_lo = np.minimum(splat_to_course(lo), splat_to_course(hi))
    c_hi = np.maximum(splat_to_course(lo), splat_to_course(hi))
    print("\n  course frame (z-down) — course = (x, -y, -z):")
    for i, ax in enumerate("xyz"):
        print(f"    {ax}: {c_lo[i]:+6.2f} .. {c_hi[i]:+6.2f}")
    print(f"\n  recommended waypoint box (inset {c.a.margin} m):")
    for i, ax in enumerate("xyz"):
        print(f"    {ax}: {c_lo[i] + c.a.margin:+6.2f} .. {c_hi[i] - c.a.margin:+6.2f}")
    print()

    c.results["bounds_splat"] = {ax: [round(float(lo[i]), 2), round(float(hi[i]), 2)]
                                 for i, ax in enumerate("xyz")}
    c.results["path_length_m"] = round(path, 1)
    warn("these are bounds of where the CAMERA went, not free space — check the course "
         "against the splat in ns-viewer; furniture does not appear in a bounding box")


# ── course ───────────────────────────────────────────────────────────────────────

def step_course(c):
    import numpy as np
    p = c.repo / "configs" / "courses" / f"{c.a.course}.json"
    if not p.exists():
        raise StepFailed(
            f"{p.relative_to(c.repo)} not found. Author it using the recommended box printed "
            "by the `bounds` step, remembering z is NEGATIVE (1.1 m altitude → z = -1.1), "
            "then re-run with --from course.")

    _, lo, hi = _bounds(c)
    kf = json.loads(p.read_text())["waypoints"]["keyframes"]
    outside = 0
    print(f"  {'keyframe':<8} {'t':>6}  {'course':<24} {'splat':<24} in?")
    for name, k in kf.items():
        pc = np.array([(f[0] if isinstance(f, list) else f) for f in k["fo"][:3]], dtype=float)
        ps = course_to_splat(pc)
        # A null position cell means "free" (unconstrained) in FiGS course files, e.g. the
        # pass-through keyframes fo0a/fo4a in circuit.json. Only check constrained axes.
        fixed = ~np.isnan(ps)
        inside = bool(np.all(ps[fixed] >= lo[fixed]) and np.all(ps[fixed] <= hi[fixed]))
        outside += not inside
        print(f"  {name:<8} {k['t']:6.2f}  "
              f"({pc[0]:+5.2f},{pc[1]:+5.2f},{pc[2]:+5.2f})        "
              f"({ps[0]:+5.2f},{ps[1]:+5.2f},{ps[2]:+5.2f})        "
              f"{'✔' if inside else '✗ OUTSIDE'}"
              f"{'  (free: ' + ''.join(ax for ax, f in zip('xyz', fixed) if not f) + ')' if not fixed.all() else ''}")
    print()
    c.results["course"] = {"name": c.a.course, "keyframes": len(kf), "outside": outside}
    if outside:
        warn(f"{outside} waypoint(s) outside the captured volume. The sim will run without "
             "error and render floaters at exactly those points.")
        if not c.a.allow_outside:
            raise StepFailed("refusing to fly outside the capture. Shrink the course, or "
                             "pass --allow-outside to proceed anyway.")
    else:
        ok(f"all {len(kf)} keyframes inside the captured volume")


# ── simulate ─────────────────────────────────────────────────────────────────────

def step_simulate(c):
    import numpy as np
    if vram_now() > 1000:
        warn(f"{vram_now()} MiB in use — close ns-viewer for a clean measurement")

    os.chdir(c.repo)
    from figs.simulator import Simulator
    from figs.control.vehicle_rate_mpc import VehicleRateMPC
    import figs.visualize.generate_videos as gv

    t0 = time.time()
    with VramMonitor(interval=2) as vm:
        sim = Simulator(c.scene, c.a.method, c.a.frame)
        ctl = VehicleRateMPC(c.a.pilot, c.a.course, c.a.frame)
        tA, tB, x0 = ctl.tXUd[0, 0], ctl.tXUd[-1, 0], ctl.tXUd[0, 1:11]
        info(f"trajectory {tA:.2f}s → {tB:.2f}s at {ctl.hz} Hz")
        info("first run for a new course triggers acados C codegen — slow once, cached after")
        Tro, Xro, Uro, Fro, Rgb, Dpt, Aux = sim.simulate(ctl, tA, tB, x0)
        # imageio's FFMPEG writer picks the container from the extension and refuses
        # ".partial", so keep ".mp4" last: backroom_flight.partial.mp4 -> backroom_flight.mp4
        tmp = c.out_mp4.with_name(f"{c.out_mp4.stem}.partial.mp4")
        gv.images_to_mp4(Rgb, str(tmp), ctl.hz)
        Path(tmp).rename(c.out_mp4)

    wall = elapsed(t0)
    rel = c.out_mp4.relative_to(c.repo)
    ok(f"{rel} ({Rgb.shape[0]} frames) in {wall}")
    c.results["sim"] = {"video": str(rel), "frames": int(Rgb.shape[0]), "hz": float(ctl.hz),
                        "wallclock": wall, "peak_vram_mib": vm.peak,
                        "duration_s": round(float(tB - tA), 2)}

    ref = np.array([np.interp(Tro, ctl.tXUd[:, 0], ctl.tXUd[:, i]) for i in (1, 2, 3)]).T
    err = np.linalg.norm(Xro[:, :3] - ref, axis=1)
    info(f"tracking error: mean {err.mean():.3f} m, max {err.max():.3f} m")
    c.results["sim"]["track_err_mean_m"] = round(float(err.mean()), 3)
    c.results["sim"]["track_err_max_m"] = round(float(err.max()), 3)
    if err.max() > 0.5:
        warn("large tracking error — trajectory may be too aggressive for the MPC; "
             "stretch the course `t` values")

    # content statistics, computed while Rgb is still in memory
    c.results["sim"]["pixel_mean"] = round(float(Rgb.mean()), 1)
    c.results["sim"]["pixel_std"] = round(float(Rgb.std()), 1)
    c.results["sim"]["dark_frames"] = int(
        (Rgb.reshape(Rgb.shape[0], -1).mean(axis=1) < 10).sum())


# ── validate ─────────────────────────────────────────────────────────────────────

def step_validate(c):
    p = c.out_mp4
    rel = p.relative_to(c.repo)
    s = c.results.get("sim", {})
    if not p.exists() or p.stat().st_size == 0:
        raise StepFailed(f"{rel} missing or empty")
    ok(f"{rel}  {p.stat().st_size / 1024 ** 2:.1f} MB")

    meta = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,nb_read_frames,duration",
         "-of", "json", str(p)], capture_output=True, text=True).stdout)["streams"][0]
    n_file = int(meta["nb_read_frames"])
    info(f"{meta['width']}x{meta['height']}  {n_file} frames  {float(meta['duration']):.2f}s")

    problems = []
    if s.get("frames") and n_file != s["frames"]:
        problems.append(f"file has {n_file} frames, simulator produced {s['frames']} — "
                        "truncated write")
    if s.get("duration_s") and abs(float(meta["duration"]) - s["duration_s"]) > 0.5:
        warn(f"duration {float(meta['duration']):.2f}s vs course {s['duration_s']:.2f}s")

    # the checks that catch "ran fine, rendered garbage"
    info(f"pixel mean {s.get('pixel_mean')}  std {s.get('pixel_std')}  "
         f"near-black {s.get('dark_frames')}/{s.get('frames')}")
    if s.get("pixel_std", 99) < 5:
        problems.append(f"pixel std {s['pixel_std']} — frames are near-uniform, not a render. "
                        "Usually means the camera is outside the splat entirely; re-check the "
                        "course sign convention (z is NEGATIVE in course frame).")
    if s.get("dark_frames", 0) > 0.05 * max(s.get("frames", 1), 1):
        warn(f"{s['dark_frames']} near-black frames — some waypoints likely fall outside the "
             "captured volume")

    if problems:
        for x in problems:
            fail(x)
        raise StepFailed("output validation failed")
    ok("output validated")


# ── record ───────────────────────────────────────────────────────────────────────

def step_record(c):
    c.results["finished"] = datetime.now().isoformat(timespec="seconds")
    runs = c.repo / "runs"
    runs.mkdir(exist_ok=True)
    rec = runs / f"{c.scene}_{datetime.now():%Y-%m-%d_%H%M}.json"
    rec.write_text(json.dumps(c.results, indent=2, default=str) + "\n")
    print(json.dumps(c.results, indent=2, default=str))
    ok(f"run record: {rec.relative_to(c.repo)}")
    ck = c.results.get("sfm", {}).get("ckpt", "")
    if ck:
        cfg = Path(ck).parent.parent / "config.yml"
        print(f"""
  Inspect the splat:
    cd {c.repo}/gsplats/workspace
    ns-viewer --load-config {Path(*cfg.parts[2:])}
    → http://localhost:7007   (holds ~4.6 GB; Ctrl-C to release)

  Watch the flight:  {c.out_mp4}
""")


# ════════════════════════════════════════════════════════════════════════════════
#  Step registry.  (name, function, fingerprint keys, description)
# ════════════════════════════════════════════════════════════════════════════════

STEPS = [
    ("preflight", step_preflight, [], "environment, imports, acados libs, GPU"),
    ("probe", step_probe, ["video"], "inspect the source video"),
    ("transcode", step_transcode, ["video", "width", "height", "fps"], "stage into gsplats/capture/"),
    ("aruco", step_aruco, ["marker_id"], "marker presence and time distribution"),
    ("config", step_config, ["marker_id", "marker_length", "num_images", "num_marked"],
     "write configs/captures/<scene>.json"),
    ("patch", step_patch, ["dataloader_workers"], "re-apply hloc submodule fixes"),
    ("sfm", step_sfm, ["marker_id", "marker_length", "num_images", "num_marked"],
     "extract → hloc → COLMAP → ArUco alignment  (LONG)"),
    ("train", step_train, ["marker_id", "marker_length", "num_images", "num_marked",
                           "train_iters", "downscale", "cache_images", "train_vis", "train_arg"],
     "ns-train splatfacto  (LONG, GPU)"),
    ("verify", step_verify, [], "registration rate, point cloud, single model"),
    ("bounds", step_bounds, [], "capture extent and course frame conversion"),
    ("course", step_course, ["course"], "validate waypoints against the capture"),
    ("simulate", step_simulate, ["course", "frame", "pilot", "method"], "fly it, render MP4"),
    ("validate", step_validate, [], "check the MP4 is real, not garbage"),
    ("record", step_record, [], "write runs/<scene>_<ts>.json"),
]
_ALWAYS = {"preflight", "patch", "record"}      # cheap and/or must run every invocation


def main():
    ap = argparse.ArgumentParser(
        description="FiGS capture-to-flight pipeline (resumable)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--scene", required=True, help="scene name; must NOT be a substring of "
                                                   "another scene's capture filename")
    ap.add_argument("--video", help="source video (required unless already transcoded)")
    ap.add_argument("--project-root", default=None,
                    help="FiGS PROJECT_ROOT. Auto-detected from $ACADOS_SOURCE_DIR or "
                         "this script's location; override only if that fails.")
    ap.add_argument("--marker-id", type=int, default=0)
    ap.add_argument("--marker-length", type=float, default=0.18, help="metres, tape-measured")
    ap.add_argument("--num-images", type=int, default=600)
    ap.add_argument("--num-marked", type=int, default=40)
    ap.add_argument("--train-iters", type=int, default=None,
                    help="splatfacto iterations (nerfstudio default 30000)")
    ap.add_argument("--downscale", type=int, default=None,
                    help="training image downscale factor (nerfstudio default: auto, ≤1600 px)")
    ap.add_argument("--cache-images", choices=["cpu", "gpu"], default=None,
                    help="where training images are cached; cpu saves VRAM")
    ap.add_argument("--train-vis", default=None,
                    help="nerfstudio --vis (viewer, tensorboard, viewer+tensorboard, ...)")
    ap.add_argument("--train-arg", action="append", default=[], metavar="ARG",
                    help="extra ns-train argument before the dataparser, repeatable, "
                         "e.g. --train-arg='--pipeline.model.stop-split-at 10000' (use '=')")
    ap.add_argument("--archive-old", action="store_true",
                    help="move existing trained models of this scene to _archive/ before training")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--course", default="intellisense_loop")
    ap.add_argument("--frame", default="carl")
    ap.add_argument("--pilot", default="Viper")
    ap.add_argument("--method", default="eval_single")
    ap.add_argument("--margin", type=float, default=0.5, help="waypoint-box inset, metres")
    ap.add_argument("--dataloader-workers", type=int, default=0,
                    help="hloc DataLoader workers; 0 avoids the OpenCV fork segfault")
    ap.add_argument("--allow-outside", action="store_true",
                    help="fly even if waypoints leave the captured volume")
    ap.add_argument("--redo", action="append", default=[], metavar="STEP")
    ap.add_argument("--from", dest="from_step", metavar="STEP")
    ap.add_argument("--only", metavar="STEP")
    ap.add_argument("--stop-after", metavar="STEP")
    ap.add_argument("--list-steps", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    names = [s[0] for s in STEPS]
    if a.list_steps:
        for n, _, keys, d in STEPS:
            print(f"  {n:<10} {d}" + (f"   {_C['d']}[{','.join(keys)}]{_C['x']}" if keys else ""))
        return 0

    # `gsplat` was split into `sfm` + `train`; accept the old name everywhere.
    _alias = {"gsplat": ["sfm", "train"]}
    a.redo = [x for r in a.redo for x in _alias.get(r, [r])]
    if a.from_step == "gsplat":
        a.from_step = "sfm"
    if a.stop_after == "gsplat":
        a.stop_after = "train"
    if a.only == "gsplat":
        ap.error("--only gsplat is now two steps: use --from sfm --stop-after train")

    for opt in (a.from_step, a.only, a.stop_after, *a.redo):
        if opt and opt not in names:
            ap.error(f"unknown step '{opt}'. Options: {', '.join(names)}")

    c = Ctx(a)

    # One-time migration: a scene trained before the split has gsplat.done. Its SfM
    # parameters are the same fingerprint keys, and it was trained with the upstream
    # defaults, i.e. with every train_* option unset.
    old = c.state / "gsplat.done"
    if old.exists() and not (c.state / "sfm.done").exists():
        fp_old = old.read_text().strip().split("\n")[0]
        fp_sfm = c.fingerprint(["marker_id", "marker_length", "num_images", "num_marked"])
        if fp_old == fp_sfm:
            c.mark_done("sfm", fp_sfm)
            saved = {k: getattr(a, k) for k in ("train_iters", "downscale", "cache_images",
                                                 "train_vis", "train_arg")}
            a.train_iters = a.downscale = a.cache_images = a.train_vis = None
            a.train_arg = []
            c.mark_done("train", c.fingerprint(["marker_id", "marker_length", "num_images",
                                                "num_marked", "train_iters", "downscale",
                                                "cache_images", "train_vis", "train_arg"]))
            for k, v in saved.items():
                setattr(a, k, v)
            info("migrated gsplat.done → sfm.done + train.done")
            old.rename(old.with_suffix(".migrated"))

    if a.status:
        section(f"Status — {a.scene}")
        for n, _, keys, d in STEPS:
            m = c.state / f"{n}.done"
            when = m.read_text().strip().split("\n")[-1] if m.exists() else ""
            print(f"  {'✔' if m.exists() else '·'} {n:<10} {when:<20} {d}")
        info(f"state: {c.state}")
        return 0

    for n in a.redo:
        c.clear(n)
        info(f"cleared marker for '{n}'")

    selected = names
    if a.only:
        selected = [a.only]
    else:
        if a.from_step:
            selected = names[names.index(a.from_step):]
        if a.stop_after:
            selected = [n for n in selected if names.index(n) <= names.index(a.stop_after)]

    print(f"{_C['b']}FiGS pipeline{_C['x']}  scene={a.scene}  steps={','.join(selected)}")
    info(f"project root: {c.project_root}")
    info(f"state dir:    {c.state}   (delete a .done marker to redo a step)")

    t_all = time.time()
    for n, fn, keys, desc in STEPS:
        if n not in selected:
            continue
        fp = c.fingerprint(keys)
        if n not in _ALWAYS and c.is_done(n, fp) and n not in a.redo:
            print(f"  {_C['d']}· {n:<10} already done — skipping{_C['x']}")
            continue

        section(f"{n}  —  {desc}")
        t0 = time.time()
        try:
            fn(c)
        except KeyboardInterrupt:
            print()
            warn(f"interrupted during '{n}' — marker NOT written, this step will re-run")
            info(f"resume with:  {' '.join(sys.argv)}")
            c.save_results()
            return 130
        except StepFailed as e:
            print()
            fail(f"step '{n}' failed:\n\n{e}\n")
            info(f"after fixing, resume with:  {' '.join(sys.argv)}")
            c.save_results()
            return 1
        except Exception:
            import traceback
            print()
            traceback.print_exc()
            d = diagnose(traceback.format_exc())
            if d:
                print()
                fail(d)
            info(f"after fixing, resume with:  {' '.join(sys.argv)}")
            c.save_results()
            return 1

        if n not in _ALWAYS:
            c.mark_done(n, fp)
        c.save_results()
        info(f"'{n}' completed in {elapsed(t0)}")

    section(f"Done — total {elapsed(t_all)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
