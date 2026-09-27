"""Read scalar series from TensorBoard event files without TensorFlow.

nerfstudio (`--vis tensorboard`) writes events.out.tfevents.* into the training run
directory through torch's SummaryWriter. The format is TFRecord framing around
`Event` protobufs; scalars arrive either as Summary.Value.simple_value or as a
one-element float tensor. Only what the UI plots is decoded: step, tag, value.
"""
from __future__ import annotations

import struct
from pathlib import Path


def _varint(b: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        c = b[i]
        i += 1
        result |= (c & 0x7F) << shift
        if not c & 0x80:
            return result, i
        shift += 7


def _fields(b: bytes):
    """Yield (field_number, wire_type, value) for one protobuf message."""
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(b, i)
            v, i = b[i:i + ln], i + ln
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:  # groups (3/4) are not used by TensorBoard
            return
        yield f, wt, v


def _tensor_value(b: bytes) -> float | None:
    for f, wt, v in _fields(b):
        if f == 5 and wt == 2 and len(v) >= 4:         # packed float_val
            return struct.unpack("<f", v[:4])[0]
        if f == 5 and wt == 5:
            return struct.unpack("<f", v)[0]
        if f == 4 and wt == 2 and len(v) == 4:          # tensor_content, one float32
            return struct.unpack("<f", v)[0]
    return None


def _records(path: Path):
    with open(path, "rb") as fh:
        while True:
            head = fh.read(12)
            if len(head) < 12:
                return
            (length,) = struct.unpack("<Q", head[:8])
            data = fh.read(length)
            fh.read(4)                                   # data crc (not verified)
            if len(data) < length:
                return                                   # file still being written
            yield data


def read_scalars(run_dir: Path, max_points: int = 400) -> dict[str, list[list[float]]]:
    """{tag: [[step, value], ...]} across every event file in run_dir, downsampled per tag."""
    series: dict[str, dict[int, float]] = {}
    for ev in sorted(run_dir.glob("events.out.tfevents.*")):
        for rec in _records(ev):
            step, summary = 0, None
            for f, wt, v in _fields(rec):
                if f == 2 and wt == 0:
                    step = v
                elif f == 5 and wt == 2:
                    summary = v
            if summary is None:
                continue
            for f, wt, val in _fields(summary):
                if f != 1 or wt != 2:
                    continue
                tag, value = None, None
                for g, gwt, gv in _fields(val):
                    if g == 1 and gwt == 2:
                        tag = gv.decode("utf-8", "replace")
                    elif g == 2 and gwt == 5:
                        value = struct.unpack("<f", gv)[0]
                    elif g == 8 and gwt == 2 and value is None:
                        value = _tensor_value(gv)
                if tag is not None and value is not None:
                    series.setdefault(tag, {})[step] = value
    out = {}
    for tag, pts in series.items():
        items = sorted(pts.items())
        if len(items) > max_points:
            k = len(items) / max_points
            items = [items[int(i * k)] for i in range(max_points)] + [items[-1]]
        out[tag] = [[s, v] for s, v in items]
    return out
