"""Ground-truth query annotations: gsplats/workspace/<scene>/semantics/queries.json.

Positions are in Galley's COURSE frame (x, −y, −z; z down) — what the course editor shows when
you place the Goal marker — so they can be typed straight from the editor.

    python -m radiance_semantics.annotations --scene backroom init          # the Phase 1 gate set
    python -m radiance_semantics.annotations --scene backroom set "red tool chest" 1.2 -0.4 -0.6
    python -m radiance_semantics.annotations --scene backroom list

Phase 3 writes the same file from the splat editor's Annotate mode.

Positions can also be placed from the training photos, on the CPU (Phase 5):

    python -m radiance_semantics.annotations --scene backroom locate frame_00141 1650 520 frame_00261 470 560
    python -m radiance_semantics.annotations --scene backroom set "recycling bin" X Y Z --set phase5 --instance

`locate` projects the SfM sparse points into each named photo (its transforms.json pose), takes the nearest
depth cluster around the pixel and unprojects it; several (frame, u, v) picks of one object are averaged and
their spread printed as a consistency check. A query with several instances of a class keeps every instance
in "instances" ("position" stays the first one, for the readers that take one).
"""

import argparse
import json
import sys

import numpy as np
from datetime import datetime
from pathlib import Path

# Chosen 5 Oct from the Phase 0 image strips: single instances, varied size and colour.
GATE_SET = {
    "backroom": [
        ("red tool chest", "big, saturated colour, one of a kind — the easy case (frames 0060, 0090)"),
        ("shop vacuum", "medium cylinder among similar-coloured clutter (0209)"),
        ("green foam mats", "flat leaning panels; colour + material phrase (0090)"),
        ("garden cart", "black, low, partly occluded — the hardest (0209)"),
        ("whiteboard", "large, seen from many angles; assumed to be one rolling board (0030, 0149, 0179, 0239)"),
    ],
}


def path_for(run_or_scene_dir):
    d = getattr(run_or_scene_dir, "semantics_dir", None)
    return (Path(d) if d is not None else Path(run_or_scene_dir) / "semantics") / "queries.json"


def load(p):
    p = Path(p)
    if not p.exists():
        return {"version": 1, "frame": "course (x, -y, -z), z down", "queries": []}
    return json.loads(p.read_text())


def save(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.part")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(p)


def init(data, scene):
    have = {q["text"] for q in data["queries"]}
    for text, note in GATE_SET.get(scene, []):
        if text not in have:
            data["queries"].append({"text": text, "position": None, "set": "phase1_gate", "note": note})
    return data


def set_position(data, text, xyz, author=None, set_name=None, instance=False, note=None):
    """Set a query's position; with instance=True add xyz as one more instance of the class."""
    for q in data["queries"]:
        if q["text"] == text:
            break
    else:
        q = {"text": text, "set": set_name or "extra"}
        data["queries"].append(q)
    xyz = [float(v) for v in xyz]
    if instance:
        q.setdefault("instances", [q["position"]] if q.get("position") is not None else [])
        q["instances"].append(xyz)
        q["position"] = q["instances"][0]
    else:
        q["position"] = xyz
        q.pop("instances", None)
    if set_name:
        q["set"] = set_name
    if note:
        q["note"] = note
    q["updated"] = datetime.now().isoformat(timespec="seconds")
    if author:
        q["author"] = author
    return q


def instances(q):
    """Every annotated position of a query (one unless it lists several instances)."""
    if q.get("instances"):
        return [list(p) for p in q["instances"]]
    return [list(q["position"])] if q.get("position") is not None else []


def frame_camera(tf, stem):
    """(c2w [4, 4] OpenGL, fx, fy, cx, cy) of the transforms.json frame whose file stem is `stem`."""
    for f in tf["frames"]:
        if Path(f["file_path"]).stem == stem:
            g = lambda k: float(f.get(k, tf.get(k)))          # noqa: E731  (per-frame intrinsics override)
            return np.array(f["transform_matrix"], float), g("fl_x"), g("fl_y"), g("cx"), g("cy")
    raise KeyError(f"no frame {stem} in transforms.json")


def locate(tf, xyz, stem, u, v, radius=20.0, band=0.25):
    """Splat-frame point under pixel (u, v) of a training photo, from the SfM sparse points xyz [p, 3]:
    the points projecting within `radius` px, their nearest depth cluster (the 20th-percentile depth
    ± band m; foreground in front of the background), unprojected at that cluster's median depth.
    Returns (xyz [3], number of supporting points)."""
    c2w, fx, fy, cx, cy = frame_camera(tf, stem)
    R, t = c2w[:3, :3], c2w[:3, 3]
    pc = (np.asarray(xyz, float) - t) @ R                     # world → camera (OpenGL: looks along −z, y up)
    d = -pc[:, 2]
    front = d > 0.1
    pu = cx + fx * pc[:, 0] / np.where(front, d, 1)
    pv = cy - fy * pc[:, 1] / np.where(front, d, 1)
    near = front & (np.hypot(pu - u, pv - v) <= radius)
    if near.sum() < 3:
        raise ValueError(f"{stem} ({u:.0f}, {v:.0f}): only {int(near.sum())} sparse points within {radius:.0f} px")
    dn = d[near]
    d0 = np.percentile(dn, 20)
    sel = dn[np.abs(dn - d0) <= band]
    z = float(np.median(sel))
    pcam = np.array([(u - cx) / fx * z, -(v - cy) / fy * z, -z])
    return R @ pcam + t, int(len(sel))


def locate_depth(run, picks, width=960, window=2, log=print):
    """Like locate, but from the frozen splat's own rendered depth (dense, so thin panels and far walls
    work): each (stem, u, v) is read from the expected depth (camera-space z of the Gaussians, alpha-
    normalised) rendered from that photo's REFINED pose, median over a (2·window+1)² patch with alpha ≥ 0.5,
    and unprojected. GPU. Returns [(xyz splat frame, depth m)]."""
    import torch
    from .cameras import in_workspace, load_pipeline, train_views
    from .lift import gaussians_from_model, render_size, view_K, views_from_pipeline
    from .render import render_features, viewmat_from_c2w
    with in_workspace(run):
        _, pipeline, _, _ = load_pipeline(run)
        cams, files, _ = train_views(pipeline)
        views = {v.stem: v for v in views_from_pipeline(pipeline.model, cams, files)}
        g = gaussians_from_model(pipeline.model)
        del pipeline
    dev = g.means.device
    out = []
    for stem, u, v in picks:
        if stem not in views:
            raise KeyError(f"{stem} is not a training view (held out, or dropped as a pose outlier)")
        vw = views[stem]
        W, H = render_size(vw, width)
        vm, K = viewmat_from_c2w(vw.c2w.to(dev).float()), view_K(vw, W, H).to(dev)
        z = (g.means @ vm[:3, :3].T + vm[:3, 3])[:, 2:3]
        with torch.no_grad():
            img, alpha = render_features(g, z.contiguous(), vm, K, W, H)
        depth = (img / alpha.clamp_min(1e-6))[..., 0].cpu().numpy()
        a = alpha[..., 0].cpu().numpy()
        su, sv = W / vw.width, H / vw.height                          # photo pixels → render pixels
        x, y = int(round(u * su)), int(round(v * sv))
        ys, xs = slice(max(0, y - window), y + window + 1), slice(max(0, x - window), x + window + 1)
        dz = depth[ys, xs][a[ys, xs] >= 0.5]
        if len(dz) == 0:
            raise ValueError(f"{stem} ({u:.0f}, {v:.0f}): nothing rendered there")
        zz = float(np.median(dz))
        fx, fy, cx, cy = vw.fx, vw.fy, vw.cx, vw.cy
        pc = np.array([(u - cx) / fx * zz, -(v - cy) / fy * zz, -zz])
        c2w = vw.c2w.cpu().numpy()
        out.append((c2w[:3, :3] @ pc + c2w[:3, 3], zz))
    return out


def ready(data, set_name=None):
    """(annotated, missing) queries, optionally only one set."""
    qs = [q for q in data["queries"] if set_name is None or q.get("set") == set_name]
    return [q for q in qs if q.get("position") is not None], [q for q in qs if q.get("position") is None]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root")
    ap.add_argument("--scene", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="add the scene's Phase 1 gate objects (positions empty)")
    sp = sub.add_parser("set", help="set a query's position (course frame, metres)")
    sp.add_argument("text")
    sp.add_argument("xyz", nargs=3, type=float)
    sp.add_argument("--set", dest="set_name", help="query set name (e.g. phase5)")
    sp.add_argument("--instance", action="store_true", help="add as one more instance of the class")
    sp.add_argument("--note")
    lp = sub.add_parser("locate", help="3D position (course frame) from (frame, u, v) picks in training photos")
    lp.add_argument("picks", nargs="+", help="frame_stem u v [frame_stem u v …] (pixels of the full-size photo)")
    lp.add_argument("--radius", type=float, default=20.0)
    lp.add_argument("--depth", action="store_true", help="use the splat's rendered depth (GPU) instead of the SfM points")
    sub.add_parser("list")
    a = ap.parse_args(argv)

    from .paths import SemanticsError, find_scene_run, resolve_project_root
    try:
        run = find_scene_run(resolve_project_root(a.project_root), a.scene)
    except SemanticsError as e:
        print(f"  ✗ {e}", file=sys.stderr)
        return 1
    p = path_for(run)
    data = load(p)
    if a.cmd == "locate":
        from .query import to_course
        ws = run.project_root / "SousVide" / "gsplats" / "workspace" / a.scene
        tf = json.loads((ws / "transforms.json").read_text())
        xyz = _read_ply_xyz(ws / "sparse_pc.ply")
        if len(a.picks) % 3:
            print("  ✗ picks come in threes: frame_stem u v", file=sys.stderr)
            return 1
        pts = []
        triples = [(a.picks[i], float(a.picks[i + 1]), float(a.picks[i + 2])) for i in range(0, len(a.picks), 3)]
        if a.depth:
            for (stem, u, v), (P, z) in zip(triples, locate_depth(run, triples)):
                pts.append(P)
                print(f"   {stem} ({u:.0f}, {v:.0f}): course {np.round(to_course(P), 3).tolist()}  (splat depth {z:.2f} m)")
        for stem, u, v in ([] if a.depth else triples):
            P, n = locate(tf, xyz, stem, u, v, a.radius)
            pts.append(P)
            print(f"   {stem} ({u:.0f}, {v:.0f}): course {np.round(to_course(P), 3).tolist()}  ({n} points)")
        P = np.mean(pts, 0)
        spread = max(np.linalg.norm(q - P) for q in pts)
        print(f"  → course {np.round(to_course(P), 3).tolist()}  (spread {spread:.2f} m over {len(pts)} pick(s))")
        return 0
    if a.cmd == "init":
        init(data, a.scene)
        save(p, data)
    elif a.cmd == "set":
        q = set_position(data, a.text, a.xyz, set_name=a.set_name, instance=a.instance, note=a.note)
        save(p, data)
        print(f"  ✔ {q['text']}: {q['position']} (course frame)")
    done, missing = ready(data)
    print(f"  {p}")
    for q in data["queries"]:
        pos = q.get("position")
        print(f"   {'✔' if pos else '·'} {q['text']:<22} {pos if pos else '(no position yet)'}  [{q.get('set', '')}]")
    if missing:
        print(f"  {len(missing)} without a position: place Galley's Goal marker on each object (course editor, "
              f"Show ▸ Splat) and run `set`.")
    return 0


def _read_ply_xyz(path):
    """x, y, z of a binary or ASCII PLY's vertices (the SfM sparse_pc.ply)."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header") + len(b"end_header")
    end = raw.index(b"\n", end) + 1
    header = raw[:end].decode("ascii", "replace").splitlines()
    n = next(int(l.split()[2]) for l in header if l.startswith("element vertex"))
    props = [l.split() for l in header if l.startswith("property")]
    fmt = next(l.split()[1] for l in header if l.startswith("format"))
    types = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1", "int": "i4", "uint": "u4",
             "short": "i2", "ushort": "u2", "char": "i1"}
    if fmt == "ascii":
        rows = np.loadtxt(raw[end:].decode().splitlines()[:n], ndmin=2)
        names = [p[2] for p in props]
        return rows[:, [names.index(k) for k in ("x", "y", "z")]]
    dt = np.dtype([(p[2], ("<" if "little" in fmt else ">") + types[p[1]]) for p in props])
    a = np.frombuffer(raw, dt, count=n, offset=end)
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(float)


if __name__ == "__main__":
    sys.exit(main())
