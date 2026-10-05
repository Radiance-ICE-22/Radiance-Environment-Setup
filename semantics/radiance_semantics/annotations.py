"""Ground-truth query annotations: gsplats/workspace/<scene>/semantics/queries.json.

Positions are in Galley's COURSE frame (x, −y, −z; z down) — what the course editor shows when
you place the Goal marker — so they can be typed straight from the editor.

    python -m radiance_semantics.annotations --scene backroom init          # the Phase 1 gate set
    python -m radiance_semantics.annotations --scene backroom set "red tool chest" 1.2 -0.4 -0.6
    python -m radiance_semantics.annotations --scene backroom list

Phase 3 writes the same file from the splat editor's Annotate mode.
"""

import argparse
import json
import sys
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


def set_position(data, text, xyz, author=None):
    for q in data["queries"]:
        if q["text"] == text:
            break
    else:
        q = {"text": text, "set": "extra"}
        data["queries"].append(q)
    q["position"] = [float(v) for v in xyz]
    q["updated"] = datetime.now().isoformat(timespec="seconds")
    if author:
        q["author"] = author
    return q


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
    if a.cmd == "init":
        init(data, a.scene)
        save(p, data)
    elif a.cmd == "set":
        q = set_position(data, a.text, a.xyz)
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


if __name__ == "__main__":
    sys.exit(main())
