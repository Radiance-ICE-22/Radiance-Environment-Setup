#!/usr/bin/env python3
# Usage (GPU, kitchen env, from semantics/): python scripts/diag_teachers.py out.json
"""Where is the signal lost for a query? Per annotated instance, in 2-3 training photos where it is visible:
 (a) relevancy of the cached teacher grid (scale-averaged pyramid) at the object's pixel (max of 3x3 cells)
 (d) relevancy of a CLIP crop centred on the object at each of the 7 pyramid scales (best scale reported)
 (c) relevancy of the lift table's Gaussians within 0.25 m of the annotation (max)
plus the eval outcome (L-C). Same LERF relevancy formula as the query."""
import json, sys, glob
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from radiance_semantics.paths import find_scene_run
from radiance_semantics.teachers import load_maps, load_image, TeacherSettings, CLIP_MEAN, CLIP_STD, common_grid
from radiance_semantics.models import load_clip
from radiance_semantics.annotations import load, path_for, frame_camera, instances
from radiance_semantics.store import read_table
from radiance_semantics.query import TextEncoder, NEGATIVES, relevancy, to_splat

dev = "cuda"
clip, _, _ = load_clip(dev)
enc = TextEncoder()
ts = TeacherSettings()

def rel_of(vecs, E):
    v = np.asarray(vecs, np.float32); v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)
    return relevancy(v, np.ones(len(v), np.float32), E[0], E[1:])

out = []
for scene in ("backroom", "flightroom"):
    run = find_scene_run(Path.home() / "Radiance/figs", scene)
    ws = run.scene_dir
    tf = json.loads((ws / "transforms.json").read_text())
    tdir = sorted((run.semantics_dir / "teachers").glob("*/meta.json"), key=lambda p: p.stat().st_mtime)[-1].parent
    table = read_table(run.backend_dir("lift")); geom = np.asarray(table.geom, np.float32)
    ev = {r["text"]: r for r in json.loads((run.semantics_dir / run.run / "eval" / "L-C.json").read_text())["records"]}
    qs = [q for q in load(path_for(run))["queries"] if q.get("set") == "phase5" and not q.get("negative")]
    frames = {Path(f["file_path"]).stem: f for f in tf["frames"]}
    for q in qs:
        E = enc.encode([q["text"]] + NEGATIVES)
        best = {"a": 0.0, "d": 0.0, "d_scale": None, "views": 0}
        # (c) 3D: Gaussians near any instance
        c3 = 0.0
        for p in instances(q):
            ps = to_splat(np.asarray(p, float))
            near = np.linalg.norm(geom[:, :3] - ps, axis=1) < 0.25
            if near.any():
                c3 = max(c3, float(rel_of(np.asarray(table.clip[np.flatnonzero(near)], np.float32), E).max()))
        # views: instance projects inside (10 % margin), depth 0.8-6 m; nearest 3
        cands = []
        for stem in frames:
            c2w, fx, fy, cx, cy = frame_camera(tf, stem)
            W, H = int(frames[stem].get("w", tf.get("w"))), int(frames[stem].get("h", tf.get("h")))
            for p in instances(q):
                ps = to_splat(np.asarray(p, float)); pc = (ps - c2w[:3, 3]) @ c2w[:3, :3]; z = -pc[2]
                if not (0.8 < z < 6): continue
                u, v = cx + fx * pc[0] / z, cy - fy * pc[1] / z
                if 0.1 * W < u < 0.9 * W and 0.1 * H < v < 0.9 * H:
                    cands.append((z, stem, u, v, W, H))
        for z, stem, u, v, W, H in sorted(cands)[:3]:
            maps = load_maps(tdir, stem)
            if "clip" not in maps: continue
            g = maps["clip"]; gh, gw = g.shape[:2]
            i, j = int(v / H * gh), int(u / W * gw)
            cell = g[max(0, i - 1):i + 2, max(0, j - 1):j + 2].reshape(-1, g.shape[-1])
            a = float(rel_of(cell, E).max())
            img = load_image(ws / frames[stem]["file_path"], dev)
            _, Hi, Wi = img.shape; su, sv = Wi / W, Hi / H
            crops, scs = [], []
            for sc in ts.scales:
                t = int(round(sc * min(Hi, Wi)))
                y0 = int(np.clip(v * sv - t / 2, 0, Hi - t)); x0 = int(np.clip(u * su - t / 2, 0, Wi - t))
                crops.append(F.interpolate(img[None, :, y0:y0 + t, x0:x0 + t], size=(224, 224), mode="bilinear", antialias=True)[0]); scs.append(sc)
            x = (torch.stack(crops) - torch.tensor(CLIP_MEAN, device=dev).view(1, 3, 1, 1)) / torch.tensor(CLIP_STD, device=dev).view(1, 3, 1, 1)
            with torch.no_grad():
                e = clip.encode_image(x).float().cpu().numpy()
            r = rel_of(e, E); k = int(r.argmax())
            best["views"] += 1
            best["a"] = max(best["a"], a)
            if r[k] > best["d"]: best["d"], best["d_scale"] = float(r[k]), scs[k]
        rec = ev.get(q["text"], {})
        out.append({"scene": scene, "text": q["text"], "hit": rec.get("hit"), "failure": rec.get("failure"),
                    "peak3d": rec.get("peak"), "c_near": round(c3, 3), "a_grid": round(best["a"], 3),
                    "d_bestscale": round(best["d"], 3), "d_scale": best["d_scale"], "views": best["views"]})
        r = out[-1]
        print(f"{scene:<10} {r['text']:<26} {'HIT ' if r['hit'] else 'MISS'} {str(r['failure'] or ''):<13} "
              f"(a) grid {r['a_grid']:.3f}  (d) best-scale {r['d_bestscale']:.3f} @{r['d_scale']}  (c) 3D near {r['c_near']:.3f}  peak3D {r['peak3d']}  views {r['views']}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
