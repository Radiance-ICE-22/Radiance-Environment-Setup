#!/usr/bin/env bash
# Semantic embeddings, Phase 3 gate (docs/SEMANTICS_PLAN.md): the splat editor, query → course → flight.
#
#   bash sem3_gate.sh                 automated half (below), then do the browser half
#   bash sem3_gate.sh --check NAME    after the browser half: check course NAME and its flight
#
# Automated half:
#   1. pull; backend venv + tests; the committed frontend build contains the splat editor
#   2. ./run_ui.sh restart (Galley with the new code and frontend)
#   3. through the API, the way the splat editor calls it:
#        status (lift table fresh) · the browser .splat (records = table rows, 32 bytes each)
#        query "red tool chest" (QUERY=…) · relevancy bytes = rows
#   4. Send to course, as the editor does it for a new course (course/model.ts courseToGoal):
#        start = first camera position kept in the waypoint box, at rest; final keyframe = the
#        candidate's approach point, at rest, facing the object; semantic_goal with query, backend,
#        score, box, approach → configs/courses/sem_gate_<query>.json
#   5. preview (clearance), then Save and fly through the queue: course → simulate → validate → record
#   6. verdict: flight succeeded, tracking error small, the course ends at the approach point
#
# Browser half (through the port forward): Explorer ▸ Scenes ▸ backroom ▸ Semantics (splat editor) →
# type "red tool chest", Enter → the chest lights up (note "recoloured in … ms" in the legend) →
# click the chest: Properties lists its best labels → Goal ▸ Send to course → New course (keep the
# name) → Send and open → in the course editor Fly (F5). When the flight finishes:
#   bash sem3_gate.sh --check sem_red_tool_chest
#
# Takes ~5 min (mostly the flight). Run it while the queue is idle:
#   tmux new -d -s sem3 'bash ~/Radiance/Radiance-Environment-Setup/ui/deploy/sem3_gate.sh > ~/sem3_gate.log 2>&1'
#
# Overrides: SCENE=backroom QUERY="red tool chest" SKIP_PULL=1 BASE=http://127.0.0.1:8800
set -u
source "$(dirname "$(readlink -f "$0")")/host.sh"
REPO=$RADIANCE_REPO
SCENE=${SCENE:-backroom}
QUERY=${QUERY:-red tool chest}
CHECK=""
[ "${1:-}" = "--check" ] && CHECK=${2:?usage: sem3_gate.sh --check COURSE}
stamp() { date '+%F %T'; }
cd "$REPO/ui/backend" || exit 1
PORT=$(.venv/bin/python -c 'from galley.settings import load; print(load().port)' 2>/dev/null || echo 8800)
BASE=${BASE:-http://127.0.0.1:$PORT}
TESTS=0

if [ -z "$CHECK" ]; then
  echo "=== $(stamp) sync $REPO"
  if [ -z "${SKIP_PULL:-}" ]; then
    cd "$REPO" && git pull --ff-only && git log --oneline -3 || exit 1
  fi
  ls "$REPO"/ui/frontend/dist/assets/SplatEditor-*.js >/dev/null 2>&1 || { echo "ui/frontend/dist has no SplatEditor chunk: push the Phase 3 commit (with dist/) first"; exit 1; }
  echo "  frontend build: $(ls "$REPO"/ui/frontend/dist/assets/ | grep -E '^(index|SplatEditor)-' | tr '\n' ' ')"

  echo; echo "=== $(stamp) backend venv + tests"
  cd "$REPO/ui/backend" || exit 1
  UV=$(command -v uv || echo ~/.local/bin/uv)
  [ -d .venv ] || $UV venv -q .venv
  $UV pip install -q -p .venv -e '.[test]'
  .venv/bin/python -m pytest -q 2>&1 | tail -2
  TESTS=${PIPESTATUS[0]}

  echo; echo "=== $(stamp) restart Galley"
  bash "$REPO/run_ui.sh" restart || { echo "run_ui.sh restart failed (a job running?)"; exit 1; }
fi

echo; echo "=== $(stamp) checks against $BASE"
.venv/bin/python - "$BASE" "$SCENE" "$QUERY" "$CHECK" <<'EOF'
import json, math, re, sys, time, urllib.request
base, scene, query, check = sys.argv[1:5]
api = base + "/api"
fails = []

def call(method, path, body=None, raw=False, timeout=600):
    req = urllib.request.Request(api + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        return data if raw else json.loads(data)

def ok(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}", flush=True)
    if not cond:
        fails.append(msg)
    return cond

def fly(name):
    job = call("POST", "/jobs/figs", {"scene": scene, "course": name, "from_step": "course", "stop_after": "record",
                                      "redo": ["course", "simulate", "validate"]})["id"]
    print(f"        job {job} queued (course → simulate → validate → record)", flush=True)
    return wait(job)

def wait(job):
    t0 = time.time()
    while (j := call("GET", f"/jobs/{job}"))["status"] in ("queued", "running"):
        time.sleep(5)
    print(f"        job {job}: {j['status']} after {time.time() - t0:.0f} s", flush=True)
    for l in call("GET", f"/jobs/{job}/log")[-30:]:
        if any(k in l["line"] for k in ("✔", "✗", "tracking", "failed", "Error", "outside")):
            print("         ", l["line"][:160])
    return j

def check_flight(name, j):
    c = call("GET", f"/configs/courses/{name}")
    g, kfs = c.get("semantic_goal") or {}, list(c["waypoints"]["keyframes"].items())
    last = [kfs[-1][1]["fo"][a][0] for a in range(3)]
    ok(bool(g.get("query")) and g.get("backend") and g.get("approach"), f"{name}: semantic_goal “{g.get('query')}” · {g.get('backend')} · score {g.get('score')} · approach {g.get('approach')}")
    ok(g.get("approach") is not None and math.dist(last, g["approach"]) < 1e-3, f"final keyframe {kfs[-1][0]} {last} = the approach point")
    yaw = kfs[-1][1]["fo"][3][0]
    face = math.atan2(g["position"][1] - last[1], g["position"][0] - last[0]) if g.get("position") else None
    ok(face is not None and abs(math.atan2(math.sin(yaw - face), math.cos(yaw - face))) < 0.01, f"final yaw {yaw:.3f} faces the goal ({face:.3f})" if face is not None else "final yaw: no goal position")
    ok(j["status"] == "succeeded", f"flight job {j['id']} {j['status']}")
    r = call("GET", f"/scenes/{scene}")["results"]
    sim = r.get("sim") or {}
    rc = r.get("course") or {}
    ok(rc.get("name") == name, f"results.course: {rc}")
    ok(sim.get("track_err_max_m") is not None and sim["track_err_max_m"] < 0.3,
       f"tracking error mean {sim.get('track_err_mean_m')} / max {sim.get('track_err_max_m')} m (< 0.3) · {sim.get('frames')} frames · pixel std {sim.get('pixel_std')} · dark {sim.get('dark_frames')}")
    print(f"        video: {base}/api/scenes/{scene}/flight")

if check:
    jobs = [j for j in call("GET", f"/jobs?scene={scene}") if j["kind"] == "figs" and (j.get("params") or {}).get("course") == check]
    if not ok(bool(jobs), f"a flight job for {check} (course editor ▸ Fly)"):
        sys.exit(1)
    j = jobs[0] if jobs[0]["status"] not in ("queued", "running") else wait(jobs[0]["id"])
    check_flight(check, j)
    print("\n  RESULT:", "PASSED" if not fails else f"FAILED ({len(fails)})"); sys.exit(1 if fails else 0)

st = call("GET", f"/scenes/{scene}/semantics")
fresh = [t for t in st["tables"] if t["active_run"] and not t["stale"] and t["backend"] == "lift"]
if not ok(bool(fresh), f"lift table fresh for run {st['run']} ({fresh[0]['rows'] if fresh else 0} rows)"):
    print("  run sem1_gate.sh / Build features first"); sys.exit(1)
rows = fresh[0]["rows"]
sp = call("POST", f"/scenes/{scene}/splat")
size = len(call("GET", f"/scenes/{scene}/splat/{sp['file']}", raw=True))
ok(sp.get("n_written") == rows and size == 32 * rows, f".splat {sp['file']}: {sp.get('n_written')} records, {size / 2**20:.1f} MB = 32 × {rows} rows (record i = table row i)")

t0 = time.time()
q = call("POST", f"/scenes/{scene}/semantics/query", {"text": query, "standoff": 1.0})
dt = time.time() - t0
c = q["result"]["candidates"]
if not ok(bool(c) and not q["stale"], f"query “{query}”: {len(c)} candidates in {dt:.1f} s, τ {q['result']['tau']}, margin {q['result']['margin']}"):
    sys.exit(1)
rel = call("GET", f"/scenes/{scene}/semantics/relevancy/{q['relevancy_id']}", raw=True)
ok(len(rel) == rows, f"relevancy {len(rel)} bytes = rows · {sum(b >= q['result']['tau'] * 255 for b in rel)} at or above τ (lit)")
top = c[0]
print(f"        top: centroid {top['centroid']} · box {top['box']} · approach {top['approach']} · gap {top['gap']} ({'ok' if top['gap_ok'] else 'TOO CLOSE'})")
ann = {a["text"]: a for a in call("GET", f"/scenes/{scene}/semantics/queries")["queries"]}
if ann.get(query, {}).get("position"):
    p = ann[query]["position"]
    e = math.dist(top["centroid"], p)
    ok(e <= 0.75 or all(top["box"]["lo"][a] - 0.3 <= p[a] <= top["box"]["hi"][a] + 0.3 for a in range(3)), f"top candidate is the annotated {query} ({e:.2f} m)")

# Send to course, as the editor does for a new course (course/model.ts courseToGoal)
geo = call("GET", f"/scenes/{scene}/geometry")
w = geo["waypoint_box"]
s0 = geo["camera_path"][0]
s = [min(max(s0[a], w["lo"][a]), w["hi"][a]) if w["lo"][a] <= w["hi"][a] else s0[a] for a in range(3)]
app, goal = top["approach"], top["centroid"]
yaw_to = lambda f, t: math.atan2(t[1] - f[1], t[0] - f[0])
wrap = lambda a: math.atan2(math.sin(a), math.cos(a))
ys, ye = yaw_to(s, app), yaw_to(app, goal)
r3 = lambda v: round(v, 3)
rest = lambda p, yaw: [[r3(p[0]), 0.0], [r3(p[1]), 0.0], [r3(p[2]), 0.0], [r3(yaw), 0.0]]
name = "sem_gate_" + re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")[:40]
course = {"waypoints": {"Nco": 6, "keyframes": {
    "start": {"t": 0.0, "fo": rest(s, ys)},
    "goal": {"t": round(max(3.0, math.dist(s, app) / 0.8), 2), "fo": rest(app, ys + wrap(ye - ys))}}}, "forces": None,
    "semantic_goal": {"label": query, "position": goal, "query": query, "backend": "lift", "score": top["score"],
                      "extent": top["box"], "approach": app}}
print("        saved", call("PUT", f"/configs/courses/{name}", course)["path"])
pv = call("POST", "/courses/preview", {"course": course, "scene": scene, "body_radius": 0.19, "clearance": 0.15})
cl = pv.get("clearance") or {}
ok(True, f"preview: {pv['duration_solved']} s, {pv['stats']['length_m']} m, v_max {pv['stats']['v_max']} m/s, min gap {cl.get('min')} m at {cl.get('at_t')} s"
          + (" (under 0.15 m: the straight path brushes the scenery; add a keyframe in the editor if the flight fails)" if cl.get("min") is not None and cl["min"] < 0.15 else ""))
check_flight(name, fly(name))
print("\n  RESULT:", "PASSED" if not fails else f"FAILED ({len(fails)})")
sys.exit(1 if fails else 0)
EOF
API=$?

echo; echo "=== $(stamp) Phase 3 gate (${CHECK:+browser half: }${CHECK:-automated half})"
[ -z "$CHECK" ] && printf '  %s  backend tests\n' "$([ "$TESTS" -eq 0 ] && echo PASS || echo FAIL)"
printf '  %s  query → course → flight\n' "$([ "$API" -eq 0 ] && echo PASS || echo FAIL)"
if [ "$TESTS" -eq 0 ] && [ "$API" -eq 0 ]; then
  [ -z "$CHECK" ] && echo "  Automated half PASSED. Now the browser half (see the top of this script), then: bash $0 --check sem_red_tool_chest" \
                  || echo "  Browser half PASSED — send both logs back, with the recolour time from the legend."
  exit 0
fi
echo "  FAILED — send this log back."
exit 1
