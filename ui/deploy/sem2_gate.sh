#!/usr/bin/env bash
# Semantic embeddings, Phase 2 gate (docs/SEMANTICS_PLAN.md): Galley backend.
#
#   1. pull; backend venv + tests (incl. tests/test_semantics.py)
#   2. ./run_ui.sh restart — Galley with the new code (refuses while a job runs)
#   3. through the API, as the splat editor will call it:
#        GET  /api/scenes/SCENE/semantics            steps, table fresh (not stale), annotations
#        POST /api/jobs/semantics                    queued, every step cached → succeeds in seconds
#        POST /api/scenes/SCENE/semantics/query      cold start, then the annotated queries warm
#        GET  …/relevancy/{id}, …/lift/pca           one / three bytes per row
#        POST …/labels                               a picked Gaussian's best label
#        GET  /api/semantics/worker                  running, CUDA hidden
#   4. verdict: warm queries under 1 s, same hits as the Phase 1 gate
#
# Needs the Phase 1 table for SCENE (sem1_gate.sh). Takes about a minute. Galley stays running
# afterwards with the new code. Run it while the queue is idle:
#   tmux new -d -s sem2 'bash ~/Radiance/Radiance-Environment-Setup/ui/deploy/sem2_gate.sh > ~/sem2_gate.log 2>&1'
#
# Overrides: SCENE=… SKIP_PULL=1 BASE=http://127.0.0.1:8800
set -u
source "$(dirname "$(readlink -f "$0")")/host.sh"
REPO=$RADIANCE_REPO
SCENE=${SCENE:-backroom}
stamp() { date '+%F %T'; }

echo "=== $(stamp) sync $REPO"
if [ -z "${SKIP_PULL:-}" ]; then
  cd "$REPO" && git pull --ff-only && git log --oneline -3 || exit 1
fi
[ -f "$REPO/figs/semantic_worker.py" ] || { echo "figs/semantic_worker.py missing: push the Phase 2 commit first"; exit 1; }

echo; echo "=== $(stamp) backend venv + tests"
cd "$REPO/ui/backend" || exit 1
UV=$(command -v uv || echo ~/.local/bin/uv)
[ -d .venv ] || $UV venv -q .venv
$UV pip install -q -p .venv -e '.[test]'
.venv/bin/python -m pytest -q 2>&1 | tail -2
TESTS=${PIPESTATUS[0]}

echo; echo "=== $(stamp) restart Galley"
bash "$REPO/run_ui.sh" restart || { echo "run_ui.sh restart failed (a job running?)"; exit 1; }
PORT=$(.venv/bin/python -c 'from galley.settings import load; print(load().port)')
BASE=${BASE:-http://127.0.0.1:$PORT}

echo; echo "=== $(stamp) API checks against $BASE"
.venv/bin/python - "$BASE" "$SCENE" <<'EOF'
import json, sys, time, urllib.request, urllib.error
base, scene = sys.argv[1], sys.argv[2]
api = base + "/api"
fails = []

def call(method, path, body=None, raw=False, timeout=300):
    req = urllib.request.Request(api + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        return data if raw else json.loads(data)

def check(ok, msg):
    print(f"  {'PASS' if ok else 'FAIL'}  {msg}", flush=True)
    if not ok:
        fails.append(msg)

st = call("GET", f"/scenes/{scene}/semantics")
fresh = [t for t in st["tables"] if t["active_run"] and not t["stale"]]
check(bool(fresh), f"status: run {st['run']}, steps done {[x['step'] for x in st['steps'] if x['done']]}, "
                   f"fresh tables {[t['backend'] for t in fresh]}, annotations {st['queries']}")
if not fresh:
    print("  no fresh lift table for this checkpoint: run sem1_gate.sh first"); sys.exit(1)
rows = fresh[0]["rows"]

jid = call("POST", "/jobs/semantics", {"scene": scene})["id"]
for _ in range(600):
    j = call("GET", f"/jobs/{jid}")
    if j["status"] not in ("queued", "running"):
        break
    time.sleep(1)
log = [l["line"] for l in call("GET", f"/jobs/{jid}/log")]
skipped = sum("already done" in l for l in log)
check(j["status"] == "succeeded", f"semantics job {jid}: {j['status']}, {skipped} steps skipped as already done")

call("POST", "/semantics/worker/stop")
t0 = time.time()
q = call("POST", f"/scenes/{scene}/semantics/query", {"text": "red tool chest"})
cold = time.time() - t0
check(q["stale"] is False and q["result"]["candidates"], f"cold query {cold:.1f} s (worker {q['worker_ms']} ms), "
      f"{len(q['result']['candidates'])} candidates, table key {q['table']['key']}")

ann = call("GET", f"/scenes/{scene}/semantics/queries")["queries"]
done = [a for a in ann if a.get("position")]
warm, hits = [], 0
for a in done:
    t0 = time.time()
    r = call("POST", f"/scenes/{scene}/semantics/query", {"text": a["text"]})
    warm.append(time.time() - t0)
    c = r["result"]["candidates"]
    if c:
        import math
        e = math.dist(c[0]["centroid"], a["position"])
        lo = [v - 0.3 for v in c[0]["box"]["lo"]]; hi = [v + 0.3 for v in c[0]["box"]["hi"]]
        hit = e <= 0.75 or all(l <= p <= h for l, p, h in zip(lo, a["position"], hi))
    else:
        e, hit = None, False
    hits += hit
    print(f"        {a['text']:<18} {warm[-1] * 1000:5.0f} ms  {'HIT ' if hit else 'MISS'} "
          f"{'' if e is None else f'{e:.3f} m'}  approach {c[0]['approach'] if c else None}  gap {c[0]['gap'] if c else None}")
if warm:
    check(max(warm) < 1.0, f"warm queries: max {max(warm) * 1000:.0f} ms, mean {sum(warm) / len(warm) * 1000:.0f} ms (need < 1000)")
    check(hits >= min(4, len(done)), f"{hits} of {len(done)} annotated queries hit (Phase 1 gate: 5 of 5)")

rel = call("GET", f"/scenes/{scene}/semantics/relevancy/{q['relevancy_id']}", raw=True)
check(len(rel) == rows, f"relevancy: {len(rel)} bytes for {rows} rows, max {max(rel)}")
pca = call("GET", f"/scenes/{scene}/semantics/lift/pca", raw=True)
check(len(pca) == 3 * rows, f"pca: {len(pca)} bytes = 3 × {rows}")
lab = call("POST", f"/scenes/{scene}/semantics/labels",
           {"index": 0, "labels": ["red tool chest", "floor", "wall", "ceiling", "table"]})
check(bool(lab["scores"]), f"labels for row 0 (seen {lab['seen']}): {lab['scores'][:3]}")
w = call("GET", "/semantics/worker")
check(w["running"], f"worker pid {w['pid']}, idle timeout {w['idle_s']} s, restarts {w['restarts']}, log {w['log']}")
splat = call("POST", f"/scenes/{scene}/splat")
check(splat.get("n_written") in (None, rows), f".splat records {splat.get('n_written')} vs table rows {rows}")
print()
print("  RESULT:", "PASSED" if not fails else f"FAILED ({len(fails)})")
sys.exit(1 if fails else 0)
EOF
API=$?

echo; echo "=== $(stamp) Phase 2 gate"
printf '  %s  backend tests\n' "$([ "$TESTS" -eq 0 ] && echo PASS || echo FAIL)"
printf '  %s  API checks\n' "$([ "$API" -eq 0 ] && echo PASS || echo FAIL)"
if [ "$TESTS" -eq 0 ] && [ "$API" -eq 0 ]; then
  echo "  Phase 2 gate PASSED — send this log back. Galley is running with the new code."
  exit 0
fi
echo "  Phase 2 gate FAILED — send this log back (and ~/.local/share/galley/semantic_worker.log)."
exit 1
