#!/usr/bin/env python3
"""
drone_model.py — turn the team's CAD export (Fusion OBJ + MTL) into the small GLB that
Galley draws in the course editor, plus drone.json with the sizes the clearance check uses.

    python ui/tools/drone_model.py "D:/Projects/FYP/Drone Model/Drone-obj_mtl/Drone/Drone_5_2205.obj"
    # defaults: --unit cm --forward +y --up +z --ratio 0.12; needs numpy and Node (npx gltfpack)

Output (committed, so target machines need neither the 229 MB OBJ nor Node):
    ui/frontend/public/models/drone.glb    ~2-4 MB, meshopt-compressed, BODY frame
    ui/frontend/public/models/drone.json   extents, radius, rotor centres, provenance

Frames. The GLB is in FiGS's body frame, which is FRD: x forward, y right, z down, metres,
origin at the vehicle's reference point. That is the frame of the quaternion in FiGS's state
(transform_helper.fo_to_xu: body z = cross(x, y) with gravity +z in a z-down world;
scipy's [x, y, z, w]), so the editor can place the model with the solver's own attitude
and no further conversion. The CAD export is assumed right-handed with --forward and --up
naming the model axes that point forward (the depth camera looks along it) and up.

Origin. FiGS's position is the centre of mass; the CAD file has no mass properties, so the
origin defaults to the centre of the four rotors in x/y and the middle of the airframe's
height (props and guards excluded) in z. Pass --origin X Y Z (CAD units) to override.

Size for clearance. `radius` is the farthest point of the model from the origin, prop
guards included — the sphere the clearance check subtracts. `half_extent` gives the box.

Simplification and compression are done by gltfpack (meshoptimizer), run through npx.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

UNITS = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254}
AXES = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1), "+z": (2, 1), "-z": (2, -1)}
TRANSLUCENT = ("translucent", "glass", "clear")
PROP_HINTS = ("translucent",)   # Fusion appearance used for the propellers in Drone_5_2205
GUARD_HINTS = ("nylon",)        # the printed prop guards


# ── OBJ / MTL ────────────────────────────────────────────────────────────────────

def read_mtl(path: Path) -> dict[str, tuple[float, float, float]]:
    kd, cur = {}, None
    if not path.exists():
        return kd
    for line in path.read_text(errors="replace").splitlines():
        w = line.split()
        if not w:
            continue
        if w[0] == "newmtl":
            cur = line.split(None, 1)[1].strip()
            kd[cur] = (0.7, 0.7, 0.7)
        elif w[0] == "Kd" and cur is not None:
            kd[cur] = tuple(float(x) for x in w[1:4])
    return kd


def read_obj(path: Path):
    """Positions, normals and per-material triangles as (v, n) corner index pairs."""
    V, N, tris, cur, mtllib = [], [], {}, None, None
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            t = line[:2]
            if t == "v ":
                V.append(line[2:])
            elif t == "vn":
                N.append(line[3:])
            elif t == "f ":
                c = []
                for p in line.split()[1:]:
                    s = p.split("/")
                    c.append((int(s[0]), int(s[2]) if len(s) > 2 and s[2] else 0))
                lst = tris.setdefault(cur, [])
                for i in range(1, len(c) - 1):
                    lst.append((c[0], c[i], c[i + 1]))
            elif line.startswith("usemtl"):
                cur = line.split(None, 1)[1].strip()
            elif line.startswith("mtllib"):
                mtllib = line.split(None, 1)[1].strip()
    V = np.array(" ".join(V).split(), dtype=np.float64).reshape(-1, 3)
    N = np.array(" ".join(N).split(), dtype=np.float32).reshape(-1, 3) if N else None
    out = {}
    for m, lst in tris.items():
        a = np.array(lst, dtype=np.int64).reshape(-1, 3, 2)
        a[..., 0] = np.where(a[..., 0] < 0, a[..., 0] + len(V), a[..., 0] - 1)
        if N is not None:
            a[..., 1] = np.where(a[..., 1] < 0, a[..., 1] + len(N), a[..., 1] - 1)
        out[m] = a
    return V, N, out, mtllib


# ── frames ───────────────────────────────────────────────────────────────────────

def cad_to_frd(forward: str, up: str) -> np.ndarray:
    """3x3 matrix taking CAD coordinates to body FRD (x fwd, y right, z down)."""
    fi, fs = AXES[forward]
    ui, us = AXES[up]
    if fi == ui:
        raise SystemExit("--forward and --up must be different axes")
    f = np.zeros(3); f[fi] = fs
    u = np.zeros(3); u[ui] = us
    r = np.cross(f, u)                       # right = forward × up (right-handed CAD)
    R = np.stack([f, r, -u])                 # rows: body x, y, z expressed in CAD axes
    assert abs(np.linalg.det(R) - 1) < 1e-9
    return R


def rotor_centres(P: np.ndarray) -> np.ndarray:
    """Centres of the four props: prop vertices split by quadrant, then averaged. The mean,
    not the bounding box: a two-blade prop modelled at an angle has a lopsided box (on
    Drone_5_2205 the box centres sit 1.5 cm off the motor axes; the means are within 1 mm)."""
    c = np.median(P, axis=0)
    out = []
    for sx in (1, -1):
        for sy in (1, -1):
            q = P[((P[:, 0] - c[0]) * sx > 0) & ((P[:, 1] - c[1]) * sy > 0)]
            if len(q) == 0:
                return np.empty((0, 3))
            out.append(q.mean(0))
    return np.array(out)


# ── GLB writer (positions, normals, vertex colours, uint32 indices) ──────────────

def write_glb(path: Path, prims: list[dict]):
    """prims: {name, pos (n,3) f32, nrm (n,3) f32|None, col (n,4) u8|None, idx (m,) u32,
    color (r,g,b,a) base colour, blend bool}."""
    bin_ = bytearray()
    views, accs, meshes, mats, nodes = [], [], [], [], []

    def add(arr: np.ndarray, target, comp, typ, norm=False, minmax=False):
        while len(bin_) % 4:
            bin_.append(0)
        off = len(bin_)
        b = arr.tobytes()
        bin_.extend(b)
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(b), "target": target})
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": int(arr.shape[0]), "type": typ}
        if norm:
            acc["normalized"] = True
        if minmax:
            acc["min"] = [float(x) for x in arr.min(0)]
            acc["max"] = [float(x) for x in arr.max(0)]
        accs.append(acc)
        return len(accs) - 1

    for p in prims:
        attrs = {"POSITION": add(p["pos"].astype(np.float32), 34962, 5126, "VEC3", minmax=True)}
        if p.get("nrm") is not None:
            attrs["NORMAL"] = add(p["nrm"].astype(np.float32), 34962, 5126, "VEC3")
        if p.get("col") is not None:
            attrs["COLOR_0"] = add(p["col"].astype(np.uint8), 34962, 5121, "VEC4", norm=True)
        ind = add(p["idx"].astype(np.uint32), 34963, 5125, "SCALAR")
        mat = {"name": p["name"], "pbrMetallicRoughness": {"baseColorFactor": list(p["color"]),
               "metallicFactor": 0.1, "roughnessFactor": 0.6}, "doubleSided": True}
        if p.get("blend"):
            mat["alphaMode"] = "BLEND"
        mats.append(mat)
        meshes.append({"name": p["name"], "primitives": [{"attributes": attrs, "indices": ind,
                                                          "material": len(mats) - 1, "mode": 4}]})
        nodes.append({"name": p["name"], "mesh": len(meshes) - 1})
    while len(bin_) % 4:
        bin_.append(0)
    gltf = {"asset": {"version": "2.0", "generator": "galley drone_model.py"},
            "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes,
            "meshes": meshes, "materials": mats, "accessors": accs, "bufferViews": views,
            "buffers": [{"byteLength": len(bin_)}]}
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_)))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A)); fh.write(js)
        fh.write(struct.pack("<II", len(bin_), 0x004E4942)); fh.write(bin_)


def glb_triangles(path: Path) -> int | None:
    """Triangle count from a GLB's index accessors (works on gltfpack output too)."""
    try:
        data = path.read_bytes()
        jl = struct.unpack_from("<I", data, 12)[0]
        g = json.loads(data[20:20 + jl])
        return sum(g["accessors"][p["indices"]]["count"] // 3 for m in g["meshes"] for p in m["primitives"])
    except Exception:
        return None


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obj", type=Path)
    ap.add_argument("--mtl", type=Path, help="default: the OBJ's mtllib")
    ap.add_argument("--out-dir", type=Path, default=here.parent / "frontend" / "public" / "models")
    ap.add_argument("--name", default="drone")
    ap.add_argument("--unit", choices=UNITS, default="cm", help="unit of the OBJ coordinates (Fusion: cm)")
    ap.add_argument("--forward", choices=AXES, default="+y", help="CAD axis the drone flies along")
    ap.add_argument("--up", choices=AXES, default="+z", help="CAD axis pointing up")
    ap.add_argument("--origin", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="reference point in CAD units (default: rotor centre, mid-height)")
    ap.add_argument("--ratio", type=float, default=0.12, help="fraction of triangles gltfpack keeps")
    ap.add_argument("--gltfpack", default=None, help="gltfpack command (default: gltfpack on PATH, else npx)")
    a = ap.parse_args()

    obj = a.obj
    print(f"reading {obj} ({obj.stat().st_size / 1e6:.0f} MB)…", flush=True)
    sha = hashlib.sha256()
    with open(obj, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            sha.update(chunk)
    V, N, tris, mtllib = read_obj(obj)
    mtl = a.mtl or (obj.parent / mtllib if mtllib else obj.with_suffix(".mtl"))
    kd = read_mtl(mtl)
    n_in = sum(len(t) for t in tris.values())
    print(f"  {len(V):,} vertices, {n_in:,} triangles, {len(tris)} materials", flush=True)

    s = UNITS[a.unit]
    R = cad_to_frd(a.forward, a.up)
    low = {m: (m or "").lower() for m in tris}
    is_prop = {m: any(h in low[m] for h in PROP_HINTS) for m in tris}
    is_guard = {m: any(h in low[m] for h in GUARD_HINTS) for m in tris}

    def verts(sel):
        ids = np.unique(np.concatenate([tris[m][..., 0].ravel() for m in tris if sel(m)] or [np.empty(0, int)]))
        return V[ids]

    props = verts(lambda m: is_prop[m])
    rot = rotor_centres(props) if len(props) else np.empty((0, 3))
    if a.origin:
        origin = np.array(a.origin, dtype=float)
    else:
        frame = verts(lambda m: not is_prop[m] and not is_guard[m])
        if len(rot) == 4:
            origin = rot.mean(0)
        else:
            origin = (V.min(0) + V.max(0)) / 2
        up_i = AXES[a.up][0]
        origin[up_i] = (frame[:, up_i].min() + frame[:, up_i].max()) / 2
    to_body = lambda P: ((np.asarray(P, float) - origin) * s) @ R.T

    # one primitive for everything opaque (vertex colours), one per translucent material
    prims, opaque = [], {"pos": [], "nrm": [], "col": [], "idx": []}
    base = 0
    for m, t in sorted(tris.items(), key=lambda kv: str(kv[0])):
        corners = t.reshape(-1, 2)
        key, inv = np.unique(corners, axis=0, return_inverse=True)
        pos = to_body(V[key[:, 0]]).astype(np.float32)
        nrm = (N[key[:, 1]] @ R.T).astype(np.float32) if N is not None else None
        idx = inv.reshape(-1).astype(np.uint32)
        rgb = kd.get(m, (0.7, 0.7, 0.7))
        if any(h in low[m] for h in TRANSLUCENT):
            prims.append({"name": m or "default", "pos": pos, "nrm": nrm, "col": None, "idx": idx,
                          "color": (*rgb, 0.55), "blend": True})
            continue
        opaque["pos"].append(pos)
        opaque["nrm"].append(nrm)
        c = np.empty((len(pos), 4), np.uint8)
        c[:, :3] = np.round(np.clip(rgb, 0, 1) * 255)
        c[:, 3] = 255
        opaque["col"].append(c)
        opaque["idx"].append(idx + base)
        base += len(pos)
    if opaque["pos"]:
        prims.insert(0, {"name": "airframe", "pos": np.concatenate(opaque["pos"]),
                         "nrm": None if N is None else np.concatenate(opaque["nrm"]),
                         "col": np.concatenate(opaque["col"]), "idx": np.concatenate(opaque["idx"]),
                         "color": (1, 1, 1, 1), "blend": False})

    a.out_dir.mkdir(parents=True, exist_ok=True)
    glb = a.out_dir / f"{a.name}.glb"
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw.glb"
        write_glb(raw, prims)
        cmd = a.gltfpack.split() if a.gltfpack else (["gltfpack"] if shutil.which("gltfpack") else ["npx", "-y", "gltfpack"])
        # -si: keep this fraction of triangles; -sa: allow it to reach the target on CAD meshes
        # full of tiny disconnected parts; -cc: meshopt compression (three.js MeshoptDecoder)
        run = cmd + ["-i", str(raw), "-o", str(glb), "-si", str(a.ratio), "-sa", "-cc"]
        print("  " + " ".join(run[-9:]), flush=True)
        subprocess.run(run, check=True)

    P = to_body(V)
    he = np.abs(P).max(0)
    guards = to_body(verts(lambda m: is_guard[m])) if any(is_guard.values()) else np.empty((0, 3))
    rb = to_body(rot) if len(rot) else rot
    prop_r = None
    if len(rot) == 4:
        pb = to_body(props)
        prop_r = float(np.median([np.linalg.norm((pb - c)[:, :2], axis=1)[np.argmin(np.linalg.norm(pb[:, None, :2] - rb[None, :, :2], axis=2), axis=1) == i].max()
                                  for i, c in enumerate(rb)]))
    meta = {
        "name": a.name,
        "file": f"{a.name}.glb",
        "frame": "FiGS body FRD: x forward, y right, z down; metres; origin = reference point",
        "source": {"obj": obj.name, "sha256": sha.hexdigest(), "unit": a.unit,
                   "forward": a.forward, "up": a.up, "origin_cad": [round(float(x), 4) for x in origin],
                   "origin_rule": "given" if a.origin else "rotor centre (x/y), mid airframe height (z)"},
        "triangles": {"source": n_in, "glb": glb_triangles(glb), "ratio": a.ratio},
        "bytes": glb.stat().st_size,
        "half_extent": [round(float(x), 4) for x in he],
        "size": [round(float(x), 4) for x in (P.max(0) - P.min(0))],
        "radius": round(float(np.linalg.norm(P, axis=1).max()), 4),
        "radius_xy": round(float(np.linalg.norm(P[:, :2], axis=1).max()), 4),
        "guards": bool(len(guards)),
        "rotors": [[round(float(x), 4) for x in c] for c in rb],
        "prop_radius": None if prop_r is None else round(prop_r, 4),
    }
    (a.out_dir / f"{a.name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("triangles", "bytes", "size", "radius", "radius_xy", "prop_radius")}))


if __name__ == "__main__":
    sys.exit(main())
