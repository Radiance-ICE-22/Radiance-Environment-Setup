"""Phase 5 evaluation: every variant × the frozen query set of a scene → per-query records and metrics.

Variants (docs/SEMANTICS_PLAN.md, Phase 5):
    L-C   lift table, CLIP query
    L-CD  lift table, CLIP query + DINO at query time (relevancy diffused over the DINO-weighted kNN
          graph, candidates split where their DINO features form two groups; query.Settings)
    F-C   FMGS trained on CLIP alone (DINO loss and pixel alignment off) — the fmgs_c table
    F-CD  FMGS with its paper weights — the fmgs table

Per positive query: the top candidate is scored against EVERY annotated instance (hit = query.score_against
for any instance), the error is the distance from its centroid to the nearest instance. Failures are sorted
into "no candidate", "bad position" (missed, but the top centroid within `near` m of an instance: the right
object, badly localised) and "wrong object".
Negatives (objects not in the scene): the relative threshold always keeps the best-matching Gaussians, so
a fixed false-positive rate says little; we report it at the fixed relevancy floor (any candidate = false
positive) AND a threshold-free separation: the AUROC of the query's peak relevancy, positives vs negatives.

    python -m radiance_semantics.evaluate --scene backroom                    # all four variants, set phase5
    python -m radiance_semantics.evaluate --report out/ --scenes backroom GTN_lab_v1   # CSV + LaTeX table

Writes semantics/<run>/eval/<variant>.json (records + metrics) per scene. Settings are the query defaults,
never tuned on this set (the five Phase 1 queries were the development set).
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from .query import Scene, Settings, run_query, score_against

VARIANTS = {
    "L-C": ("lift", {}),
    "L-CD": ("lift", {"dino_diffuse": True, "dino_split": True}),
    "F-C": ("fmgs_c", {}),
    "F-CD": ("fmgs", {}),
}
def parse_variant(v):
    """"L-C" → ("lift", {}, ""); "L-C@480" → the same variant on the lift_w480 table (feature-width sweep)."""
    name, _, width = v.partition("@")
    if name not in VARIANTS or (width and not width.isdigit()):
        raise ValueError(f"unknown variant {v!r} ({', '.join(VARIANTS)}, optionally @<feature width>)")
    backend, over = VARIANTS[name]
    return backend, over, f"_w{width}" if width else ""


NEAR = 1.5          # m: a miss whose top centroid is this close to an instance is "bad position"


def score_instances(result, instances):
    """Best score_against over the annotated instances, plus the nearest-instance error."""
    if not result["candidates"]:
        return {"hit": False, "error": None, "rank_of_hit": None, "instance": None}
    best = None
    for i, p in enumerate(instances):
        sc = score_against(result, p)
        key = (sc["hit"], -(sc["error"] if sc["error"] is not None else 1e9))
        if best is None or key > best[0]:
            best = (key, dict(sc, instance=i))
    sc = best[1]
    ranks = [r for r in (score_against(result, p)["rank_of_hit"] for p in instances) if r is not None]
    sc["rank_of_hit"] = min(ranks) if ranks else None
    return sc


def failure_kind(rec):
    if rec["hit"]:
        return None
    if rec["error"] is None:
        return "no candidate"
    return "bad position" if rec["error"] <= NEAR else "wrong object"


def auroc(pos, neg):
    """P(score of a random positive > score of a random negative), ties count half. None if a side is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()
    return float(gt / (len(pos) * len(neg)))


def aggregate(records):
    pos = [r for r in records if not r["negative"]]
    neg = [r for r in records if r["negative"]]
    errs = [r["error"] for r in pos if r["error"] is not None]
    hits = [r for r in pos if r["hit"]]
    kinds = {}
    for r in pos:
        k = failure_kind(r)
        if k:
            kinds[k] = kinds.get(k, 0) + 1
    m = {
        "n_pos": len(pos), "n_neg": len(neg),
        "hit_rate": round(len(hits) / len(pos), 3) if pos else None,
        "hits": len(hits),
        "error_median_m": round(float(np.median(errs)), 3) if errs else None,
        "error_p90_m": round(float(np.percentile(errs, 90)), 3) if errs else None,
        "ambiguous_rate": round(sum(r["ambiguous"] for r in pos) / len(pos), 3) if pos else None,
        "feasible_rate": round(sum(bool(r["gap_ok"]) for r in hits) / len(hits), 3) if hits else None,
        "neg_fp_rate": round(sum(r["n_candidates"] > 0 for r in neg) / len(neg), 3) if neg else None,
        "neg_auroc": auroc([r["peak"] for r in pos], [r["peak"] for r in neg]),
        "query_ms_median": int(np.median([r["ms"] for r in records])) if records else None,
        "failures": kinds,
    }
    if m["neg_auroc"] is not None:
        m["neg_auroc"] = round(m["neg_auroc"], 3)
    return m


def build_cost(table, teachers_meta=None):
    """Build time (min), peak VRAM (GB), table size (MB) of a table, from its index.json metrics."""
    met = table.index.get("metrics", {})
    part = met.get("fmgs") or met.get("lift") or {}
    secs = part.get("seconds")
    if part.get("steps") and part.get("it_per_s"):          # FMGS: a run resumed after an interruption
        secs = part["steps"] / part["it_per_s"]                # records only its last leg in "seconds"
    vram = part.get("peak_vram_mib_device") or part.get("peak_vram_mib")
    size = sum(f.stat().st_size for f in Path(table.path).iterdir() if f.is_file()) / 2 ** 20
    out = {"build_min": round(secs / 60, 1) if secs else None, "peak_vram_gb": round(vram / 1024, 2) if vram else None,
           "table_mb": round(size, 1)}
    if teachers_meta:
        out["teachers_min"] = round(teachers_meta.get("seconds", 0) / 60, 1) or None
    return out


def load_scene(scene_dir):
    from .annotations import _read_ply_xyz
    tf = json.loads((Path(scene_dir) / "transforms.json").read_text())
    c2w = np.array([f["transform_matrix"] for f in sorted(tf["frames"], key=lambda f: f.get("file_path", ""))], float)
    ply = Path(scene_dir) / "sparse_pc.ply"
    return Scene(cam_c2w=c2w, sparse_xyz=_read_ply_xyz(ply) if ply.exists() else None)


def evaluate(table, scene, queries, encoder, settings=None, log=print):
    """Run every query; return the records (one per query)."""
    s = settings or Settings()
    recs = []
    for q in queries:
        t0 = time.time()
        res = run_query(table, scene, q["text"], encoder, s)
        ms = int((time.time() - t0) * 1000)
        top = res["candidates"][0] if res["candidates"] else None
        rec = {"text": q["text"], "negative": bool(q.get("negative")), "n_candidates": len(res["candidates"]),
               "peak": res["peak"], "tau": res["tau"], "ambiguous": bool(res["ambiguous"]), "margin": res["margin"],
               "top": top and {k: top[k] for k in ("centroid", "box", "approach", "gap", "gap_ok", "n", "score")},
               "gap_ok": top and top["gap_ok"], "ms": ms}
        if not rec["negative"]:
            from .annotations import instances
            rec.update(score_instances(res, instances(q)))
            rec["failure"] = failure_kind(rec)
        recs.append(rec)
        tag = ("NEG " + ("fp" if rec["n_candidates"] else "ok")) if rec["negative"] else \
              ("HIT " if rec["hit"] else "MISS") + (f" {rec['error']:.2f} m" if rec.get("error") is not None else "")
        log(f"    {q['text']:<28} {tag:<14} peak {res['peak']:.3f}  {ms} ms")
    return recs


def run_variants(project_root, scene, variants, set_name="phase5", log=print):
    from .annotations import load, path_for
    from .paths import find_scene_run
    from .query import TextEncoder
    from .store import read_table
    run = find_scene_run(project_root, scene)
    data = load(path_for(run))
    queries = [q for q in data["queries"] if q.get("set") == set_name and (q.get("negative") or q.get("position") is not None)]
    if not queries:
        raise SystemExit(f"no annotated '{set_name}' queries in {path_for(run)}")
    sc = load_scene(run.scene_dir)
    enc = TextEncoder()
    outdir = run.semantics_dir / run.run / "eval"
    outdir.mkdir(parents=True, exist_ok=True)
    tmeta = None
    tdirs = sorted((run.semantics_dir / "teachers").glob("*/meta.json"), key=lambda p: p.stat().st_mtime)
    if tdirs:
        tmeta = json.loads(tdirs[-1].read_text())
    summary = {}
    for v in variants:
        backend, over, suffix = parse_variant(v)
        bdir = run.backend_dir(backend, suffix)
        if not (bdir / "index.json").exists():
            log(f"  {v}: no {backend} table at {bdir} — skipped")
            continue
        table = read_table(bdir)
        if table.index.get("key") != run.key:
            log(f"  {v}: the {backend} table is for another checkpoint — skipped (rebuild it)")
            continue
        s = replace(Settings(), **over)
        log(f"  {v} ({backend}{', DINO at query time' if over else ''}): {len(queries)} queries")
        recs = evaluate(table, sc, queries, enc, s, log)
        m = aggregate(recs)
        m.update(build_cost(table, tmeta))
        doc = {"scene": scene, "run": run.run, "key": run.key, "variant": v, "backend": backend, "set": set_name,
               "settings": asdict(s), "metrics": m, "records": recs,
               "evaluated": datetime.now().isoformat(timespec="seconds")}
        (outdir / f"{v}.json").write_text(json.dumps(doc, indent=2) + "\n")
        summary[v] = m
        log(f"  {v}: hits {m['hits']}/{m['n_pos']}, median error {m['error_median_m']} m, "
            f"negatives fp {m['neg_fp_rate']}, AUROC {m['neg_auroc']} → {outdir / (v + '.json')}")
    return summary


SWEEPS = {"threshold": [0.50, 0.525, 0.55, 0.575, 0.60],
          "rel_alpha": [0.0, 0.25, 0.5, 0.75],
          "voxel": [0.05, 0.1, 0.2, 0.3]}


def sweep(project_root, scene, variant="L-C", set_name="phase5", sweeps=None, log=print):
    """Sensitivity of one variant to the query settings, one parameter at a time around the defaults.
    Reported as sensitivity only: the defaults were fixed on the development set and stay fixed."""
    from .annotations import load, path_for
    from .paths import find_scene_run
    from .query import TextEncoder
    from .store import read_table
    run = find_scene_run(project_root, scene)
    queries = [q for q in load(path_for(run))["queries"]
               if q.get("set") == set_name and (q.get("negative") or q.get("position") is not None)]
    backend, over = VARIANTS[variant]
    table, sc, enc = read_table(run.backend_dir(backend)), load_scene(run.scene_dir), TextEncoder()
    base = replace(Settings(), **over)
    rows = []
    for param, values in (sweeps or SWEEPS).items():
        for val in values:
            m = aggregate(evaluate(table, sc, queries, enc, replace(base, **{param: val}), log=lambda _: None))
            rows.append({"scene": scene, "variant": variant, "param": param, "value": val,
                         "default": getattr(base, param) == val, **{k: m[k] for k in (
                             "hit_rate", "hits", "n_pos", "error_median_m", "ambiguous_rate", "neg_fp_rate")}})
            log(f"  {variant} {param}={val}{' (default)' if rows[-1]['default'] else ''}: hits {m['hits']}/{m['n_pos']}, "
                f"median {m['error_median_m']} m, neg FP {m['neg_fp_rate']}, ambiguous {m['ambiguous_rate']}")
    out = run.semantics_dir / run.run / "eval" / f"sweep_{variant}.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


COLUMNS = [("scene", "Scene"), ("variant", "Variant"), ("hit_rate", "Top-1 hit"), ("error_median_m", "Err. med. (m)"),
           ("error_p90_m", "Err. p90 (m)"), ("ambiguous_rate", "Ambig."), ("feasible_rate", "Feasible"),
           ("neg_fp_rate", "Neg. FP"), ("neg_auroc", "Neg. AUROC"), ("build_min", "Build (min)"),
           ("peak_vram_gb", "VRAM (GB)"), ("table_mb", "Table (MB)"), ("query_ms_median", "Query (ms)")]


def report_rows(project_root, scenes, variants=tuple(VARIANTS)):
    from .paths import find_scene_run
    rows = []
    for scene in scenes:
        run = find_scene_run(project_root, scene)
        for v in variants:
            p = run.semantics_dir / run.run / "eval" / f"{v}.json"
            if p.exists():
                m = json.loads(p.read_text())["metrics"]
                rows.append({"scene": scene, "variant": v, **m})
    return rows


def write_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([k for k, _ in COLUMNS] + ["n_pos", "n_neg", "failures"])
        for r in rows:
            w.writerow([r.get(k) for k, _ in COLUMNS] + [r.get("n_pos"), r.get("n_neg"), json.dumps(r.get("failures", {}))])


def _tex(v):
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v).replace("_", r"\_")


def write_latex(rows, path, caption="Goal localisation by variant (frozen query sets).", label="tab:sem-variants"):
    cols = COLUMNS
    lines = [r"\begin{table}[ht]", r"\centering\small", r"\begin{tabular}{@{}ll" + "r" * (len(cols) - 2) + "@{}}",
             r"\toprule", " & ".join(h for _, h in cols) + r" \\", r"\midrule"]
    last = None
    for r in rows:
        if last is not None and r["scene"] != last:
            lines.append(r"\midrule")
        last = r["scene"]
        lines.append(" & ".join(_tex(r.get(k)) for k, _ in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    from .paths import resolve_project_root
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root")
    ap.add_argument("--scene", help="evaluate this scene's variants")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), help="L-C L-CD F-C F-CD, optionally @<width> (e.g. L-C@480)")
    ap.add_argument("--set", default="phase5", help="query set in queries.json (default phase5)")
    ap.add_argument("--report", metavar="DIR", help="write variants.csv and variants.tex from the eval files")
    ap.add_argument("--sweep", metavar="VARIANT", choices=list(VARIANTS),
                    help="sensitivity of VARIANT to threshold / rel_alpha / voxel (with --scene)")
    ap.add_argument("--scenes", nargs="+", help="scenes for --report")
    a = ap.parse_args(argv)
    root = resolve_project_root(a.project_root)
    if a.scene and a.sweep:
        sweep(root, a.scene, a.sweep, a.set)
    elif a.scene:
        run_variants(root, a.scene, a.variants, a.set)
    if a.report:
        rows = report_rows(root, a.scenes or ([a.scene] if a.scene else []), a.variants)
        out = Path(a.report)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out / "variants.csv")
        write_latex(rows, out / "variants.tex")
        print(f"  {len(rows)} rows → {out / 'variants.csv'}, {out / 'variants.tex'}")
    if a.report and a.sweep:
        import csv
        rows = []
        for sc_ in a.scenes or [a.scene]:
            from .paths import find_scene_run
            r = find_scene_run(root, sc_)
            p = r.semantics_dir / r.run / "eval" / f"sweep_{a.sweep}.json"
            rows += json.loads(p.read_text()) if p.exists() else []
        with open(Path(a.report) / f"sweep_{a.sweep}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["scene"])
            w.writeheader(); w.writerows(rows)
        print(f"  {len(rows)} sweep rows → {Path(a.report) / f'sweep_{a.sweep}.csv'}")
    if not (a.scene or a.report):
        ap.error("give --scene and/or --report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
