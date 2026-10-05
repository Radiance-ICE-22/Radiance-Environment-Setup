"""Stand-in for figs/semantic_worker.py: same JSON-lines contract, relevancy from colour words
over the synthetic room (synth.py), candidates by voxel connected components."""
import argparse, base64, json, math, os, sys, time
sys.path.insert(0, os.environ["E2E_DIR"])
import synth

ap = argparse.ArgumentParser(); ap.add_argument("--project-root"); ap.add_argument("--idle", type=float); ap.parse_args()
SC = synth.scene()
print(json.dumps({"ready": True, "pid": os.getpid()}), flush=True)

def rel_for(text, backend="lift"):
    """Relevancy by colour words; the 'fmgs' stand-in is sharper (narrower kernel, higher peak)."""
    words = [w for w in text.lower().replace(",", " ").split() if w in synth.COLOURS]
    if not words: return [0.5] * len(SC)
    tgt = [sum(synth.COLOURS[w][a] for w in words) / len(words) for a in range(3)]
    s, top = (30, 0.48) if backend == "fmgs" else (45, 0.4)
    return [0.45 + top * math.exp(-sum((c[a] - tgt[a]) ** 2 for a in range(3)) / (2 * s ** 2)) for (_, _, c) in SC]

def clusters(sel, vox=0.1):
    cell = {}
    for i in sel:
        p = SC[i][0]; cell.setdefault(tuple(int(math.floor(v / vox)) for v in p), []).append(i)
    seen, out = set(), []
    for k in cell:
        if k in seen: continue
        stack, members = [k], []; seen.add(k)
        while stack:
            c = stack.pop(); members += cell[c]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nb = (c[0] + dx, c[1] + dy, c[2] + dz)
                        if nb in cell and nb not in seen: seen.add(nb); stack.append(nb)
        out.append(members)
    return out

def query(req):
    t0 = time.time(); s = req.get("settings") or {}
    rel = rel_for(req["text"], req.get("backend", "lift"))
    top = sorted(rel, reverse=True)[:100]; peak = sum(top) / len(top)
    thr = s.get("threshold", 0.55); a = s.get("rel_alpha", 0.5); tau = thr + a * max(0.0, peak - thr)
    sel = [i for i, r in enumerate(rel) if r >= tau]
    cands = []
    for m in clusters(sel):
        if len(m) < 20: continue
        score = sum((rel[i] - thr) * SC[i][2][3] / 255 for i in m)
        lo = [min(SC[i][0][a] for i in m) for a in range(3)]; hi = [max(SC[i][0][a] for i in m) for a in range(3)]
        c = [sum(SC[i][0][a] for i in m) / len(m) for a in range(3)]
        d = math.hypot(c[0], c[1]) or 1; st = s.get("standoff", 1.0)
        app = [c[0] - c[0] / d * st, c[1] - c[1] / d * st, -1.0]
        cands.append({"score": round(score, 4), "n": len(m), "large": math.dist(lo, hi) > 4,
                      "centroid": [round(v, 3) for v in c], "box": {"lo": [round(v, 3) for v in lo], "hi": [round(v, 3) for v in hi]},
                      "approach": [round(v, 3) for v in app], "cameras": 42, "gap": 0.31, "gap_ok": True,
                      "centroid_splat": [c[0], -c[1], -c[2]]})
    cands.sort(key=lambda x: -x["score"]); cands = cands[: s.get("top", 5)]
    for r, c in enumerate(cands): c["rank"] = r + 1
    # as radiance_semantics.query: (top − runner-up) / top, 1.0 with a single cluster; ambiguous below 0.25
    margin = round((cands[0]["score"] - cands[1]["score"]) / cands[0]["score"], 3) if len(cands) > 1 else (1.0 if cands else None)
    res = {"text": req["text"], "negatives": s.get("negatives", ["object", "things", "stuff", "texture"]), "threshold": thr, "tau": round(tau, 4),
           "peak": round(peak, 4), "rel_alpha": a, "rel_pct": {"p50": 0.45, "p90": 0.5, "p99": 0.8}, "n_selected": len(sel),
           "rel_max": max(rel), "rel_p99": 0.8, "voxel": 0.1, "candidates": cands, "margin": margin,
           "ambiguous": margin is not None and margin < 0.25, "frame": "course (x, -y, -z), z down",
           "ms": {"encode": 1, "relevancy": 1, "total": round((time.time() - t0) * 1000)}}
    out = {"ok": True, "ms": res["ms"]["total"], "result": res, "table": {"key": "e2e", "n": len(SC), "backend": req.get("backend", "lift")}, "stale": False}
    if req.get("relevancy"): out["relevancy_b64"] = base64.b64encode(bytes(round(r * 255) for r in rel)).decode()
    return out

for line in sys.stdin:
    req = json.loads(line); rid = req.get("id")
    try:
        if req["op"] == "ping": o = {"ok": True, "pid": os.getpid()}
        elif req["op"] == "query": o = query(req)
        elif req["op"] == "labels":
            c = SC[req["index"]][2]
            def cos(l):
                w = [x for x in l.lower().split() if x in synth.COLOURS]
                if not w: return 0.18
                t = synth.COLOURS[w[0]]; return round(0.18 + 0.12 * math.exp(-sum((c[a] - t[a]) ** 2 for a in range(3)) / (2 * 45 ** 2)), 4)
            o = {"ok": True, "scores": sorted([[l, cos(l)] for l in req["labels"]], key=lambda x: -x[1]), "seen": True,
                 "position_splat": [0, 0, 0], "table": {}, "stale": False}
        else: o = {"ok": False, "code": "bad_request", "error": "unknown op"}
    except Exception as e:
        o = {"ok": False, "code": "error", "error": repr(e)}
    o["id"] = rid; print(json.dumps(o), flush=True)
