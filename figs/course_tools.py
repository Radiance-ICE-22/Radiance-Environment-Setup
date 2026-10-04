#!/usr/bin/env python3
"""
course_tools.py — numeric helpers for Galley's course editor, run in the kitchen env.

    source figs_env.sh
    course_tools.py geometry --project-root R --scene backroom [--margin 0.5]
    course_tools.py preview  --project-root R --scene backroom --pilot Viper --frame carl \
                             [--mode fixed|expert] [--body-radius 0.19] < course.json
    course_tools.py splat    --ckpt <run>/nerfstudio_models/step-*.ckpt --out scene.splat \
                             [--max-splats 1000000] [--min-opacity 0.05]

Both print exactly one line `GALLEY_JSON {...}` on stdout (anything else is log noise).

Everything returned is in the COURSE frame, course = (x, -y, -z): z points down, so
altitude is negative z. The conversion and the inside-the-capture test are imported from
figs_pipeline.py, so the editor and the `course` step cannot disagree.

`splat` converts the trained splatfacto checkpoint into the 32-byte-per-Gaussian ".splat"
format the browser renderer reads (position, scale, RGBA, rotation), in the SPLAT frame — the
frame of the checkpoint, which is transforms.json's frame because figs_pipeline.py trains with
orientation/center "none" and no auto-scale (and the frame FiGS renders in: GSplat.Tw2g).
The editor turns it into the course frame with the same 180° rotation about x. Gaussians are
written most-visible first (volume × opacity) so a capped file keeps the ones that matter.
It needs only torch on the CPU.

`preview` runs FiGS's own MinTimeSnap and TsFO_to_tXU — the two calls VehicleRateMPC makes
before it builds the MPC — with the expert's plan settings (kT, use_l2_time) and hz, and the
frame's mass and thrust coefficient. It needs only numpy/scipy: no GPU, no nerfstudio, no
acados, so it is safe to run next to a training job.

Note: with kT set (Viper: 10.0), MinTimeSnap re-optimises the segment durations. The
keyframe `t` values in the file are only the starting guess; `t_solved` is what flies.
That optimisation (SLSQP, finite differences) is slow — about 100 s for circuit on a 2-core
VM — so `--mode fixed` (the default) solves minimum snap with the file's times (kT=None,
~2 s) for live editing, and `--mode expert` reproduces the expert's re-timed trajectory.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figs_pipeline import course_inside, course_int_cells, splat_to_course  # noqa: E402


def emit(obj, code=0):
    print("GALLEY_JSON " + json.dumps(obj, separators=(",", ":"), allow_nan=False), flush=True)
    sys.exit(code)


def r(a, nd=4):
    """Round and turn NaN into None for JSON."""
    a = np.round(np.asarray(a, dtype=float), nd)
    return [None if not np.isfinite(v) else float(v) for v in a.ravel()] if a.ndim == 1 else \
        [r(x, nd) for x in a]


# ── scene files ──────────────────────────────────────────────────────────────────

_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
              "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def read_ply(path):
    """Vertices (and colours if present) of a PLY point cloud. Handles the ASCII files
    nerfstudio writes and the binary ones open3d writes (FiGS's aligned sparse_pc.ply)."""
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt, props, n, in_vertex = None, [], 0, False
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: header has no end_header")
            w = line.decode("ascii", "replace").split()
            if not w or w[0] in ("comment", "obj_info"):
                continue
            if w[0] == "format":
                fmt = w[1]
            elif w[0] == "element":
                in_vertex = w[1] == "vertex"
                if in_vertex:
                    n = int(w[2])
            elif w[0] == "property" and in_vertex:
                if w[1] == "list":
                    raise ValueError("list properties in the vertex element are not supported")
                props.append((w[2], _PLY_TYPES[w[1]]))
            elif w[0] == "end_header":
                break
        names = [p[0] for p in props]
        if fmt == "ascii":
            data = np.loadtxt(fh, max_rows=n, ndmin=2)
            col = {nm: data[:, i] for i, nm in enumerate(names)}
        elif fmt in ("binary_little_endian", "binary_big_endian"):
            end = "<" if fmt == "binary_little_endian" else ">"
            dt = np.dtype([(nm, end + t) for nm, t in props])
            data = np.frombuffer(fh.read(dt.itemsize * n), dtype=dt, count=n)
            col = {nm: data[nm] for nm in names}
        else:
            raise ValueError(f"unsupported PLY format {fmt}")
    xyz = np.stack([col["x"], col["y"], col["z"]], axis=1).astype(float)
    rgb = None
    if all(k in col for k in ("red", "green", "blue")):
        rgb = np.stack([col["red"], col["green"], col["blue"]], axis=1)
        if rgb.dtype.kind == "f":          # some writers store 0..1 floats
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return xyz, rgb


def scene_files(root, scene):
    ws = Path(root) / "SousVide" / "gsplats" / "workspace" / scene
    return ws / "transforms.json", ws / "sparse_pc.ply"


def camera_positions(tf_path):
    tf = json.loads(Path(tf_path).read_text())
    frames = sorted(tf["frames"], key=lambda f: f.get("file_path", ""))   # extraction order
    P = np.array([f["transform_matrix"] for f in frames], dtype=float)[:, :3, 3]
    return P


def to_course(P):
    """Nx3 splat-frame points → course frame, with the pipeline's own conversion."""
    return np.asarray(splat_to_course(np.asarray(P, dtype=float).T)).T


def course_box(lo, hi):
    a, b = splat_to_course(lo), splat_to_course(hi)
    return np.minimum(a, b), np.maximum(a, b)


# ── geometry ─────────────────────────────────────────────────────────────────────

def cmd_geometry(a):
    tf_path, ply_path = scene_files(a.project_root, a.scene)
    if not tf_path.exists():
        emit({"ok": False, "error": f"{tf_path} not found: run the sfm step first"}, 2)
    P = camera_positions(tf_path)
    lo, hi = P.min(axis=0), P.max(axis=0)
    c_lo, c_hi = course_box(lo, hi)
    out = {
        "ok": True, "scene": a.scene, "frame": "course (x, -y, -z), z down",
        "camera_path": r(to_course(P), 3),
        "camera_box": {"lo": r(c_lo, 3), "hi": r(c_hi, 3)},
        "waypoint_box": {"lo": r(c_lo + a.margin, 3), "hi": r(c_hi - a.margin, 3), "margin": a.margin},
        "bounds_splat": {"lo": r(lo, 3), "hi": r(hi, 3)},
        "points": None, "colors": None, "n_points": 0, "n_points_sent": 0,
    }
    if ply_path.exists():
        xyz, rgb = read_ply(ply_path)
        xyz_c = to_course(xyz)
        n = len(xyz_c)
        idx = np.arange(n)
        if n > a.max_points:
            idx = np.sort(np.random.default_rng(0).choice(n, a.max_points, replace=False))
        out["n_points"], out["n_points_sent"] = int(n), int(len(idx))
        out["points"] = [float(v) for v in np.round(xyz_c[idx], 3).ravel()]
        if rgb is not None:
            out["colors"] = [int(v) for v in rgb[idx].ravel()]
        # robust extent (1st-99th percentile): SfM outliers make the raw box useless for framing
        q = np.percentile(xyz_c, [1, 99], axis=0)
        out["points_box"] = {"lo": r(q[0], 3), "hi": r(q[1], 3)}
    else:
        out["warning"] = f"{ply_path.name} not found: no point cloud to show"
    emit(out)


# ── splat export ─────────────────────────────────────────────────────────────────

SH_C0 = 0.28209479177387814


def _param(state, name):
    """splatfacto parameter from a checkpoint's pipeline state dict, across nerfstudio versions
    (`_model.gauss_params.means` in ≥1.0, `_model.means` before)."""
    for k, v in state.items():
        if k.endswith(f"gauss_params.{name}") or k.endswith(f"_model.{name}"):
            return v
    raise KeyError(f"checkpoint has no '{name}' parameter (not a splatfacto model?)")


def cmd_splat(a):
    import torch
    t0 = time.time()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)   # our own training output
    st = ck.get("pipeline", ck)
    means = _param(st, "means").float().numpy()
    scales = _param(st, "scales").float().numpy()             # log scale
    quats = _param(st, "quats").float().numpy()               # w, x, y, z (unnormalised)
    opac = _param(st, "opacities").float().numpy().reshape(-1)   # logits
    dc = _param(st, "features_dc").float().numpy().reshape(len(means), -1)[:, :3]   # SH degree 0
    n_total = len(means)

    alpha = 1.0 / (1.0 + np.exp(-opac))
    keep = (alpha >= a.min_opacity) & np.isfinite(means).all(1) & np.isfinite(scales).all(1)
    idx = np.nonzero(keep)[0]
    importance = np.exp(scales[idx].sum(1)) * alpha[idx]      # same order as antimatter15's converter
    idx = idx[np.argsort(-importance)][: a.max_splats]

    q = quats[idx]
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    rgb = np.clip(0.5 + SH_C0 * dc[idx], 0, 1)
    rec = np.zeros(len(idx), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"] = means[idx]
    rec["s"] = np.exp(scales[idx])
    rec["c"][:, :3] = np.round(rgb * 255)
    rec["c"][:, 3] = np.round(alpha[idx] * 255)
    rec["r"] = np.clip(np.round(q * 128 + 128), 0, 255)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    tmp.write_bytes(rec.tobytes())
    tmp.replace(out)
    lo, hi = np.percentile(means[idx], [1, 99], axis=0) if len(idx) else (np.zeros(3), np.zeros(3))
    emit({"ok": True, "n_total": int(n_total), "n_written": int(len(idx)), "min_opacity": a.min_opacity,
          "bytes": out.stat().st_size, "step": int(ck.get("step", -1)), "seconds": round(time.time() - t0, 1),
          "frame": "splat (transforms.json); course = (x, -y, -z)",
          "box_course": {"lo": r(np.minimum(splat_to_course(lo), splat_to_course(hi)), 3),
                         "hi": r(np.maximum(splat_to_course(lo), splat_to_course(hi)), 3)}})


# ── preview ──────────────────────────────────────────────────────────────────────

def load_cfg(root, family, name):
    p = Path(root) / "SousVide" / "configs" / family / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"configs/{family}/{name}.json not found")
    return json.loads(p.read_text())


def as_float_course(course):
    """Integer cells → float, so the preview shows the course as intended. Files on disk
    are checked separately (course_int_cells): FiGS itself would misread them."""
    c = json.loads(json.dumps(course))
    for k in c["waypoints"]["keyframes"].values():
        k["fo"] = [[(float(v) if isinstance(v, int) and not isinstance(v, bool) else v) for v in row]
                   for row in k["fo"]]
        k["t"] = float(k["t"])
    return c


def intervals(mask, t):
    """[[t_start, t_end], ...] where mask is true."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if m else i - 1
            out.append([round(float(t[start]), 3), round(float(t[end]), 3)])
            start = None
    return out


def cmd_preview(a):
    from figs.tsplines.min_time_snap import MinTimeSnap
    import figs.utilities.transform_helper as th
    from figs.dynamics.external_forces import ExternalForces

    raw = json.loads(sys.stdin.read())
    int_cells = course_int_cells(raw["waypoints"]["keyframes"])
    course = as_float_course(raw)
    pilot = load_cfg(a.project_root, "pilots", a.pilot)
    frame = load_cfg(a.project_root, "frames", a.frame)
    if "plan" not in pilot:
        raise ValueError(f"pilot {a.pilot} is not an expert (no 'plan' block)")
    kT, l2 = pilot["plan"]["kT"], pilot["plan"]["use_l2_time"]
    hz = pilot["track"]["hz"]
    lb = np.array(pilot["track"]["bounds"]["lower"], dtype=float)
    ub = np.array(pilot["track"]["bounds"]["upper"], dtype=float)
    m, kt = float(frame["mass"]), float(frame["motor_thrust_coeff"])

    kf = course["waypoints"]["keyframes"]
    names = list(kf)
    t_file = np.array([k["t"] for k in kf.values()])

    # exactly what VehicleRateMPC.__init__ does
    t0 = time.time()
    mts = MinTimeSnap(course["waypoints"], hz, kT if a.mode == "expert" else None, l2)
    fex = ExternalForces(course.get("forces"))
    Tsd, FOd = mts.get_desired_trajectory()
    tXUd = th.TsFO_to_tXU(Tsd, FOd, m, kt, fex)
    solve_s = time.time() - t0

    pos, vel, acc = FOd[:, 0:3, 0], FOd[:, 0:3, 1], FOd[:, 0:3, 2]
    yaw = FOd[:, 3, 0]
    # attitude FiGS derives from the flat outputs (fo_to_xu): body FRD in the course frame,
    # scipy order [x, y, z, w]. tXU columns: t, p(3), v(3), q(4), u(4).
    quat = tXUd[:, 7:11]
    speed = np.linalg.norm(vel, axis=1)
    accn = np.linalg.norm(acc, axis=1)
    U = tXUd[:, 11:15]
    finite = np.all(np.isfinite(U), axis=1)
    # usage: how far each input goes toward the bound it approaches (1.0 = at the bound)
    span_hi = np.where(ub > 0, ub, np.nan)
    span_lo = np.where(lb < 0, -lb, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        use = np.fmax(U / span_hi, -U / span_lo)
    over = (U < lb) | (U > ub)
    channels = ["thrust", "wx", "wy", "wz"]
    inputs = {
        "names": channels, "lower": r(lb), "upper": r(ub),
        "u": r(U.T, 4),
        "max_use": [None if not np.isfinite(np.nanmax(use[:, i])) else round(float(np.nanmax(use[:, i])), 3)
                    for i in range(4)],
        "violations": {channels[i]: intervals(over[:, i], Tsd) for i in range(4) if over[:, i].any()},
    }

    Tkf = np.asarray(mts.Tkf, dtype=float)
    # position each keyframe actually gets (free cells resolved by the solver)
    kidx = [int(np.argmin(np.abs(Tsd - tk))) for tk in Tkf]
    out = {
        "ok": True, "mode": a.mode, "pilot": a.pilot, "frame": a.frame, "hz": hz, "kT": kT, "use_l2_time": l2,
        "solve_s": round(solve_s, 2), "int_cells": int_cells,
        "keyframes": [{"name": n, "t_file": round(float(tf), 4), "t_solved": round(float(ts), 4),
                       "pos": r(pos[i], 4), "yaw": round(float(yaw[i]), 4)}
                      for n, tf, ts, i in zip(names, t_file, Tkf, kidx)],
        "duration_file": round(float(t_file[-1] - t_file[0]), 3),
        "duration_solved": round(float(Tkf[-1] - Tkf[0]), 3),
        "t": r(Tsd, 4), "pos": r(pos, 4), "vel": r(vel, 4), "acc": r(acc, 4), "yaw": r(yaw, 4),
        "speed": r(speed, 4), "acc_norm": r(accn, 4), "quat": r(quat, 4),
        "stats": {"v_max": round(float(speed.max()), 3), "v_mean": round(float(speed.mean()), 3),
                  "a_max": round(float(accn.max()), 3),
                  "length_m": round(float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()), 2),
                  "nonfinite_inputs": int((~finite).sum())},
        "inputs": inputs,
        "clearance": None, "inside": None,
    }

    if a.scene:
        tf_path, ply_path = scene_files(a.project_root, a.scene)
        if tf_path.exists():
            P = camera_positions(tf_path)
            lo, hi = P.min(axis=0), P.max(axis=0)
            kin = []
            for n, k in kf.items():
                pc = np.array([(row[0] if row and row[0] is not None else np.nan) for row in k["fo"][:3]])
                kin.append({"name": n, "inside": course_inside(pc, lo, hi)[0]})
            samp = np.array([course_inside(p, lo, hi)[0] for p in pos])
            out["inside"] = {"keyframes": kin, "outside_intervals": intervals(~samp, Tsd),
                             "outside_frac": round(float((~samp).mean()), 3)}
        if ply_path.exists():
            from scipy.spatial import cKDTree
            xyz, _ = read_ply(ply_path)
            # Distance to the k-th nearest sparse point, not the nearest: a lone SfM outlier
            # floating in the room is one point, a real surface is many. On dummy's backroom the
            # plain nearest-point distance put circuit (a known-good flight) 2.7 cm from "something".
            # With --body-radius R the numbers are gaps: centre distance minus R, R being the
            # drone's bounding sphere (prop guards included). The sphere is conservative for a
            # flat airframe: above or below a surface it overstates the drone by R - half height.
            k = max(1, a.clearance_k)
            body = max(0.0, a.body_radius)
            dk, _ = cKDTree(to_course(xyz)).query(pos, k=k)
            dk = dk.reshape(len(pos), -1) - body
            d, d1 = dk[:, -1], dk[:, 0]
            i, i1 = int(np.argmin(d)), int(np.argmin(d1))
            out["clearance"] = {
                "threshold": a.clearance, "k": k, "body_radius": body,
                "d": r(d, 3), "min": round(float(d[i]), 3), "min_centre": round(float(d[i] + body), 3),
                "at_t": round(float(Tsd[i]), 3), "at_pos": r(pos[i], 3),
                "below": intervals(d < a.clearance, Tsd), "n_points": int(len(xyz)),
                "nearest_min": round(float(d1[i1]), 3), "nearest_at_t": round(float(Tsd[i1]), 3),
                "note": (f"gap between the drone's {body:g} m bounding sphere and " if body > 0 else "distance to ")
                        + (f"the {k}th-nearest SfM sparse point, so isolated outliers are ignored; "
                           if k > 1 else "the nearest SfM sparse point; ")
                        + "sparse points miss textureless surfaces (plain walls, floors), so a "
                          "clear reading is not proof of free space",
            }
    emit(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geometry")
    g.add_argument("--project-root", required=True)
    g.add_argument("--scene", required=True)
    g.add_argument("--margin", type=float, default=0.5)
    g.add_argument("--max-points", type=int, default=60000)
    p = sub.add_parser("preview")
    p.add_argument("--project-root", required=True)
    p.add_argument("--scene")
    p.add_argument("--pilot", default="Viper")
    p.add_argument("--frame", default="carl")
    p.add_argument("--clearance", type=float, default=0.3)
    p.add_argument("--body-radius", type=float, default=0.0,
                   help="drone bounding-sphere radius (m) subtracted from every distance (0 = treat it as a point)")
    p.add_argument("--clearance-k", type=int, default=5,
                   help="clearance = distance to the k-th nearest sparse point (1 = nearest)")
    p.add_argument("--mode", choices=["fixed", "expert"], default="fixed",
                   help="fixed: the file's keyframe times (fast); expert: re-timed with the pilot's kT")
    sp = sub.add_parser("splat")
    sp.add_argument("--ckpt", required=True, help="splatfacto checkpoint (nerfstudio_models/step-*.ckpt)")
    sp.add_argument("--out", required=True)
    sp.add_argument("--max-splats", type=int, default=1_000_000)
    sp.add_argument("--min-opacity", type=float, default=0.05)
    a = ap.parse_args()
    try:
        {"geometry": cmd_geometry, "preview": cmd_preview, "splat": cmd_splat}[a.cmd](a)
    except SystemExit:
        raise
    except Exception as e:  # report, don't crash: the UI shows the message
        tb = traceback.format_exc().strip().splitlines()
        emit({"ok": False, "error": f"{type(e).__name__}: {e}", "trace": tb[-8:]}, 1)


if __name__ == "__main__":
    main()
