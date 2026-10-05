"""Console output in the same style as figs_pipeline.py, plus VRAM measurement.

Kept self-contained (no import from figs/): this package is pip-installed into kitchen and
must work from any working directory, while figs_pipeline.py lives next to the scripts.
"""

import json
import subprocess
import sys
import threading
import time

_C = {"g": "\033[32m", "y": "\033[33m", "r": "\033[31m", "b": "\033[1m", "d": "\033[2m", "x": "\033[0m"}
if not sys.stdout.isatty():
    _C = {k: "" for k in _C}

# One machine-readable line per command, the convention course_tools.py uses for Galley.
MARK = "GALLEY_JSON "


def section(t): print(f"\n{_C['b']}{'═' * 78}\n  {t}\n{'═' * 78}{_C['x']}", flush=True)
def ok(m):      print(f"  {_C['g']}✔{_C['x']} {m}", flush=True)
def warn(m):    print(f"  {_C['y']}!{_C['x']} {m}", flush=True)
def fail(m):    print(f"  {_C['r']}✗{_C['x']} {m}", flush=True)
def info(m):    print(f"    {_C['d']}{m}{_C['x']}", flush=True)


def emit(obj):
    print(MARK + json.dumps(obj, default=str), flush=True)


def vram_now():
    """Device-wide MiB in use (nvidia-smi), -1 if unavailable. Includes other processes."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


class VramMonitor:
    """Samples nvidia-smi in a thread; `.peak` is the device-wide peak in MiB."""

    def __init__(self, interval=1.0):
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
        self.peak = max(self.peak, vram_now())


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0
