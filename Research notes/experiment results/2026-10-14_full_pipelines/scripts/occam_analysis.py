"""Extra measurements for the Occam's-full-chain report (kitchen env, CPU):

    CUDA_VISIBLE_DEVICES= python occam_analysis.py

1. Per Phase 5 query, for Occam's full table, Occam's lift on our A teachers (13 Oct) and our A table: the level /
   scale group the query chose, the peak of every group, the top candidate's size → ../raw/occam/levels.json
2. LangSplat feature statistics per image and level (masks kept, pixel coverage) → ../raw/occam/masks.csv
3. Per-image extraction time from the run logs → ../raw/occam/timing.csv
4. Per-level table coverage (rows with a feature) → ../raw/occam/table_levels.json
5. A figure of the three SAM levels on one image per scene → ../report/fig_levels_<scene>.png
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path.home() / "Radiance/Radiance-Environment-Setup"
sys.path.insert(0, str(REPO / "semantics"))
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "raw" / "occam"
TP = Path.home() / "Radiance/third_party"
TABLES = {"Occam-full": "lift_occamfull", "Occam-ctrl": "lift_occamms", "Ours-A": "lift"}
FIG_FRAMES = {"backroom": "frame_00141", "flightroom": None}


def queries_and_levels():
    from radiance_semantics.annotations import load, path_for
    from radiance_semantics.evaluate import load_scene
    from radiance_semantics.paths import find_scene_run
    from radiance_semantics.query import Settings, TextEncoder, run_query
    from radiance_semantics.store import read_table
    out = {}
    enc = TextEncoder()
    for scene in ("backroom", "flightroom"):
        run = find_scene_run(Path.home() / "Radiance/figs", scene)
        qs = [q for q in load(path_for(run))["queries"] if q.get("set") == "phase5"]
        sc = load_scene(run.scene_dir)
        out[scene] = {}
        for cfg, t in TABLES.items():
            table = read_table(run.backend_dir("lift", t[4:] if t != "lift" else ""))
            rows = []
            for q in qs:
                r = run_query(table, sc, q["text"], enc, Settings())
                top = r["candidates"][0] if r["candidates"] else None
                rows.append({"text": q["text"], "negative": bool(q.get("negative")), "scale_group": r["scale_group"],
                             "scale_peaks": r["scale_peaks"], "peak": r["peak"], "tau": r["tau"],
                             "n_selected": r["n_selected"], "n_candidates": len(r["candidates"]),
                             "top_n": top["n"] if top else None})
            out[scene][cfg] = rows
            print(scene, cfg, "levels chosen:", [x["scale_group"] for x in rows], flush=True)
    (OUT / "levels.json").write_text(json.dumps(out, indent=1) + "\n")


def mask_stats():
    rows = []
    for scene in ("backroom", "flightroom"):
        for f in sorted((TP / "langsplat_features" / scene).glob("*_s.npy")):
            s = np.load(f)                                     # [4, H, W], ids offset across levels, -1 = none
            r = {"scene": scene, "image": f.name[:-6], "H": s.shape[1], "W": s.shape[2],
                 "n_features": int(np.load(str(f)[:-6] + "_f.npy", mmap_mode="r").shape[0])}
            for li, name in enumerate(("default", "s", "m", "l")):
                lv = s[li]
                ids = np.unique(lv[lv >= 0])
                r[f"masks_{name}"] = int(len(ids))
                r[f"cover_{name}"] = round(float((lv >= 0).mean()), 4)
            rows.append(r)
        print(scene, len(rows), "images", flush=True)
    with open(OUT / "masks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def timing():
    rows = []
    pat = re.compile(r"(\d+)/(\d+) (\S+): ([\d.]+) s, peak ([\d.]+) GB")
    for log in ("full_pipelines.log", "occam_full.log"):
        p = TP / "logs" / log
        if not p.exists():
            continue
        for m in pat.finditer(p.read_text(errors="ignore")):
            n = int(m.group(2))
            rows.append({"scene": "backroom" if n == 270 else "flightroom", "image": m.group(3),
                         "seconds": float(m.group(4)), "peak_gb": float(m.group(5)), "log": log})
    seen = {}
    for r in rows:                                              # keep the run that produced the file (last one)
        seen[(r["scene"], r["image"])] = r
    with open(OUT / "timing.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scene", "image", "seconds", "peak_gb", "log"])
        w.writeheader()
        w.writerows(seen.values())
    print("timing rows", len(seen))


def table_levels():
    from radiance_semantics.paths import find_scene_run
    from radiance_semantics.store import read_table
    out = {}
    for scene in ("backroom", "flightroom"):
        run = find_scene_run(Path.home() / "Radiance/figs", scene)
        t = read_table(run.backend_dir("lift", "_occamfull"))
        cs = t.clip_scales
        n = cs.shape[0]
        lv = {}
        for g, name in enumerate(("s", "m", "l")):
            nz = 0
            for i in range(0, n, 200_000):
                nz += int((np.abs(np.asarray(cs[i:i + 200_000, g, :8], np.float32)).sum(1) > 0).sum())
            lv[name] = {"rows_with_feature": nz, "frac": round(nz / n, 4)}
        out[scene] = {"rows": n, "seen": int((np.asarray(t.weight) > 0).sum()), "levels": lv}
    (OUT / "table_levels.json").write_text(json.dumps(out, indent=1) + "\n")
    print(out)


def figure():
    from PIL import Image
    rng = np.random.default_rng(0)
    for scene, frame in FIG_FRAMES.items():
        d = TP / "langsplat_features" / scene
        frame = frame or sorted(d.glob("*_s.npy"))[len(list(d.glob("*_s.npy"))) // 2].name[:-6]
        cams = json.loads((TP / "exports" / scene / "cameras.json").read_text())
        img_path = next(c["image"] for c in cams if c["stem"] == frame)
        s = np.load(d / f"{frame}_s.npy")
        H, W = s.shape[1:]
        img = np.asarray(Image.open(img_path).convert("RGB").resize((W, H))).astype(np.float32)
        panels = [img]
        for li in (1, 2, 3):
            lv = s[li].astype(np.int64)
            pal = rng.uniform(40, 255, (int(lv.max()) + 2, 3))
            col = np.where(lv[..., None] >= 0, pal[lv], 0.0)
            panels.append(0.45 * img + 0.55 * col)
        top = np.concatenate(panels[:2], 1)
        bot = np.concatenate(panels[2:], 1)
        im = Image.fromarray(np.concatenate([top, bot], 0).clip(0, 255).astype(np.uint8))
        im.thumbnail((1600, 1600))
        im.save(ROOT / "report" / f"fig_levels_{scene}.png")
        print("figure", scene, frame)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    timing()
    table_levels()
    figure()
    mask_stats()
    queries_and_levels()
