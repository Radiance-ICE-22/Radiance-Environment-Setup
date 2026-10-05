#!/usr/bin/env python3
"""
semantic_query.py — resolve a phrase to ranked 3D candidates in a scene's semantic table.

    source <prefix>/figs_env.sh
    ./semantic_query.py --scene backroom "red tool chest"
    ./semantic_query.py --scene backroom "shop vacuum" "whiteboard" --standoff 1.2
    ./semantic_query.py --scene backroom --eval            # the annotated set (queries.json) → hits
    ./semantic_query.py --scene backroom "garden cart" --json

Reads the table semantic_pipeline.py exported (gsplats/workspace/<scene>/semantics/<run>/<backend>/),
encodes the text with CLIP on the CPU (never competes with a GPU job), and prints, per candidate:
score, size, centroid and box, the approach point `--standoff` m away and its gap to the nearest
sparse points minus the drone's 0.19 m radius. Positions are in Galley's COURSE frame
(x, −y, −z; z down), the frame of `semantic_goal` and of the course editor.

--eval scores every annotated query in queries.json (radiance_semantics.annotations): a hit when
the TOP candidate's box (grown 0.3 m) contains the annotation or its centroid is within 0.75 m.
Exit code 0 when at least --need of them hit (Phase 1 gate: 4 of 5), 3 when annotations are missing.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from course_tools import read_ply, scene_files  # noqa: E402
from figs_pipeline import fail, info, ok, resolve_project_root, section, warn  # noqa: E402

MARK = "GALLEY_JSON "


def load_scene(project_root, scene):
    from radiance_semantics.query import Scene
    tf_path, ply_path = scene_files(project_root, scene)
    tf = json.loads(Path(tf_path).read_text())
    frames = sorted(tf["frames"], key=lambda f: f.get("file_path", ""))
    c2w = np.array([f["transform_matrix"] for f in frames], dtype=float)
    xyz = read_ply(ply_path)[0] if Path(ply_path).exists() else None
    return Scene(cam_c2w=c2w, sparse_xyz=xyz)


def show(res):
    c = res["candidates"]
    head = f"\"{res['text']}\": {len(c)} candidate(s) from {res['n_selected']:,} Gaussians ≥ {res['threshold']}" \
           f" (max relevancy {res['rel_max']}), {res['ms']['total']} ms"
    (ok if c else warn)(head + ("  — AMBIGUOUS" if res.get("ambiguous") else ""))
    for d in c:
        gap = "—" if d["gap"] is None else f"{d['gap']:+.2f} m {'ok' if d['gap_ok'] else 'TOO CLOSE'}"
        size = np.subtract(d["box"]["hi"], d["box"]["lo"])
        info(f"#{d['rank']} score {d['score']:.3f}  n={d['n']:<6} centroid {d['centroid']}  "
             f"box {size.round(2).tolist()} m  approach {d['approach']}  gap {gap}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="one or more phrases")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--project-root")
    ap.add_argument("--backend", default="lift")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--standoff", type=float, help="approach distance from the object (m, default 1.0)")
    ap.add_argument("--margin", type=float, help="waypoint box inset from the camera box (m, default 0.5)")
    ap.add_argument("--negatives", help="comma-separated (default: object,things,stuff,texture)")
    ap.add_argument("--top", type=int)
    ap.add_argument("--eval", action="store_true", help="score the annotated queries in queries.json")
    ap.add_argument("--need", type=int, default=4, help="--eval passes with at least this many hits")
    ap.add_argument("--device", default="cpu", help="text encoder device (default cpu)")
    ap.add_argument("--json", action="store_true", help="also print one GALLEY_JSON line")
    a = ap.parse_args(argv)
    if not a.text and not a.eval:
        ap.error("give a phrase, or --eval")

    from radiance_semantics import annotations
    from radiance_semantics.paths import SemanticsError, find_scene_run
    from radiance_semantics.query import Settings, TextEncoder, run_query, score_against
    from radiance_semantics.store import read_table
    try:
        root = resolve_project_root(a.project_root).resolve()
        run = find_scene_run(root, a.scene)
    except SemanticsError as e:
        fail(str(e))
        return 1
    tdir = run.backend_dir(a.backend)
    if not (tdir / "index.json").exists():
        fail(f"no {a.backend} table for {a.scene} ({tdir}): run semantic_pipeline.py --scene {a.scene}")
        return 1
    table = read_table(tdir)
    if table.index.get("key") != run.key:
        warn(f"table was built for {table.index.get('key')}, the active checkpoint is {run.key}: re-run the "
             "pipeline (results below may not match the splat)")
    s = Settings()
    for k in ("threshold", "standoff", "margin", "top"):
        if getattr(a, k) is not None:
            setattr(s, k, getattr(a, k))
    if a.negatives:
        s.negatives = [x.strip() for x in a.negatives.split(",") if x.strip()]
    scene = load_scene(root, a.scene)
    enc = TextEncoder(a.device)
    section(f"Semantic query — {a.scene} {run.run} ({a.backend}, {table.n:,} rows, teacher {table.index.get('teacher_tag')})")
    t0 = time.time()

    out = {"scene": a.scene, "run": run.run, "backend": a.backend, "settings": s.__dict__, "results": []}
    for text in a.text:
        res = run_query(table, scene, text, enc, s)
        show(res)
        out["results"].append(res)

    rc = 0
    if a.eval:
        data = annotations.load(annotations.path_for(run))
        done, missing = annotations.ready(data)
        if missing:
            warn(f"{len(missing)} annotated queries have no position yet: "
                 + ", ".join(q["text"] for q in missing))
            info(f"set them with: python -m radiance_semantics.annotations --scene {a.scene} set \"<text>\" X Y Z "
                 "(course frame, from Galley's Goal marker)")
        hits = []
        for q in done:
            res = run_query(table, scene, q["text"], enc, s)
            sc = score_against(res, q["position"])
            show(res)
            (ok if sc["hit"] else fail)(f"{q['text']}: {'HIT' if sc['hit'] else 'MISS'} — top centroid "
                                        f"{sc['error']} m from the annotation" + (f", right object at rank {sc['rank_of_hit']}"
                                        if sc["rank_of_hit"] and not sc["hit"] else ""))
            hits.append({"text": q["text"], "annotation": q["position"], **sc,
                         "top": res["candidates"][0] if res["candidates"] else None, "margin": res["margin"]})
        n_hit = sum(h["hit"] for h in hits)
        out["eval"] = {"hits": n_hit, "annotated": len(done), "missing": [q["text"] for q in missing],
                       "need": a.need, "items": hits}
        section(f"Eval: {n_hit} of {len(done)} annotated queries hit (need {a.need})")
        if missing and len(done) < a.need:
            rc = 3
        else:
            rc = 0 if n_hit >= a.need else 1
        run.runs_dir.mkdir(parents=True, exist_ok=True)
        rec = run.runs_dir / f"semantics_p1_queries_{a.scene}_{datetime.now():%Y-%m-%d_%H%M}.json"
        rec.write_text(json.dumps(out, indent=2, default=str) + "\n")
        info(f"record: {rec}")
    info(f"{time.time() - t0:.1f} s")
    if a.json:
        print(MARK + json.dumps(out, default=str), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
