#!/usr/bin/env bash
# Phase 3 gate on dummy: the course editor's backend, end to end, without a browser.
#
#   1. pull, backend tests
#   2. course_tools.py directly in kitchen: geometry, fixed and expert previews of circuit,
#      timed; the expert duration must match the last recorded backroom+circuit flight
#   3. through the API: geometry, preview, lint of every course on this machine
#   4. build a loop from backroom's recommended box and send it the way a browser does
#      (0.0 written as 0), save it, fly it through the queue with the Viper expert
#
# Then the manual half of the gate: in the browser, open Course editor → backroom → New loop,
# move a keyframe, Save and fly. Changes: ~/FYP-Radiance pulled; configs/courses/gate3_loop.json
# (also mirrored to the overlay); one flight of backroom (outputs/flights/backroom_flight.mp4,
# a runs/*.json record). kitchen is untouched.
#
#   bash ~/FYP-Radiance/ui/deploy/phase3_gate.sh 2>&1 | tee ~/phase3_gate.log
#
# Paths per machine (dummy, intellisense08) come from host.sh next to this script.
set -u
source "$(dirname "$(readlink -f "$0")")/host.sh"
ROOT=$FIGS_ROOT
UI=$RADIANCE_REPO/ui
TOOLS=$RADIANCE_REPO/figs/course_tools.py
SCENE=${SCENE:-backroom}
PORT=8800
API=http://127.0.0.1:$PORT/api
j() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1]))" "$1"; }
gj() { sed -n 's/^GALLEY_JSON //p'; }

echo "=== sync FYP-Radiance"
cd $RADIANCE_REPO && git pull --ff-only && git log --oneline -3 || exit 1
[ -f $TOOLS ] || { echo "figs/course_tools.py missing: push the Phase 3 commit first"; exit 1; }

echo; echo "=== backend tests (venv refreshed: test extras gained tensorboardX in Phase 2)"
cd $UI/backend
UV=$(command -v uv || echo ~/.local/bin/uv)
[ -d .venv ] || $UV venv -q .venv
$UV pip install -q -p .venv -e '.[test]'
GALLEY_TEST_CONFIGS=$ROOT/SousVide/configs .venv/bin/python -m pytest -q 2>&1 | tail -3

echo; echo "=== course_tools.py in kitchen (CPU only)"
(
  source $ROOT/figs_env.sh >/dev/null 2>&1 || { echo "cannot source figs_env.sh"; exit 1; }
  export CUDA_VISIBLE_DEVICES=
  C=$ROOT/SousVide/configs/courses/circuit.json
  /usr/bin/time -f "  geometry: %e s, %M KB" python $TOOLS geometry --project-root $ROOT --scene $SCENE | gj \
    | j "f\"  {d['n_points']} sparse points ({d['n_points_sent']} sent), {len(d['camera_path'])} cameras; camera box {d['camera_box']}; waypoint box {d['waypoint_box']}\""
  for mode in fixed expert; do
    /usr/bin/time -f "  $mode: %e s wall" python $TOOLS preview --project-root $ROOT --scene $SCENE --mode $mode < $C | gj \
      | j "f\"  {d['mode']}: duration {d['duration_solved']} s (file {d['duration_file']}), v_max {d['stats']['v_max']}, thrust use {d['inputs']['max_use'][0]}, min clearance {d['clearance']['min'] if d['clearance'] else None} m, outside {d['inside']['outside_frac'] if d['inside'] else None}\""
  done
  REC=$(ls -t $ROOT/SousVide/runs/${SCENE}_*.json 2>/dev/null | head -1)
  [ -n "$REC" ] && python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print('  last recorded flight', sys.argv[1].split('/')[-1], ': course', r.get('course',{}).get('name'), ', sim duration', r.get('sim',{}).get('duration_s'), 's  <- expert duration above should match for circuit')" "$REC"
)

echo; echo "=== start server"
cd $UI/backend
.venv/bin/python -m galley > /tmp/galley_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 30); do curl -sf $API/health >/dev/null && break; sleep 0.5; done

echo; echo "=== API: geometry (gzip), preview, lint"
curl -s --compressed -o /tmp/geo.json -w "  geometry: %{http_code}, %{size_download} bytes gzipped, %{time_total} s\n" $API/scenes/$SCENE/geometry
j "d['waypoint_box']" < /tmp/geo.json
python3 - "$API" "$SCENE" <<'EOF'
import json, os, sys, urllib.request
api, scene = sys.argv[1], sys.argv[2]
course = json.load(open(os.path.join(os.environ["FIGS_ROOT"], "SousVide/configs/courses/circuit.json")))
req = urllib.request.Request(f"{api}/courses/preview", json.dumps({"course": course, "scene": scene}).encode(), {"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req))
print(f"  preview via API: {d['mode']} {d['duration_solved']} s, {len(d['t'])} samples, solved in {d['solve_s']} s")
for c in json.load(urllib.request.urlopen(f"{api}/configs/courses")):
    bad = json.load(urllib.request.urlopen(f"{api}/courses/{c['name']}/lint"))["int_cells"]
    print(f"  lint {c['name']:<20} {'OK' if not bad else 'INTEGER CELLS (FiGS misreads these; re-save from the editor): ' + ', '.join(bad)}")
EOF

echo; echo "=== build gate3_loop from the recommended box, send it like a browser, save, fly"
python3 - "$API" "$SCENE" <<'EOF'
import json, sys, time, urllib.request
api, scene = sys.argv[1], sys.argv[2]
geo = json.load(open("/tmp/geo.json"))
w, c = geo["waypoint_box"], geo["camera_box"]
lo = [w["lo"][a] if w["lo"][a] <= w["hi"][a] else (c["lo"][a] + c["hi"][a]) / 2 for a in range(3)]
hi = [w["hi"][a] if w["lo"][a] <= w["hi"][a] else (c["lo"][a] + c["hi"][a]) / 2 for a in range(3)]
z = round(min(max(-1.2, lo[2]), hi[2]), 2)
x0, x1 = round(lo[0] + .2 * (hi[0] - lo[0]), 2), round(hi[0] - .2 * (hi[0] - lo[0]), 2)
y0, y1 = round(lo[1] + .2 * (hi[1] - lo[1]), 2), round(hi[1] - .2 * (hi[1] - lo[1]), 2)
P = lambda x, y, yaw: [[x, None, None, None], [y, None, None, None], [z, None, None, None], [yaw, None, None, None]]
R = lambda x, y, yaw: [[x, 0.0], [y, 0.0], [z, 0.0], [yaw, 0.0]]
course = {"waypoints": {"Nco": 6, "keyframes": {
    "start": {"t": 0.0, "fo": R(x0, y0, 0.0)}, "k1": {"t": 3.0, "fo": P(x1, y0, 1.571)},
    "k2": {"t": 6.0, "fo": P(x1, y1, 3.142)}, "k3": {"t": 9.0, "fo": P(x0, y1, 4.712)},
    "end": {"t": 12.0, "fo": R(x0, y0, 6.283)}}}, "forces": None,
    "semantic_goal": {"label": "gate test", "position": [round((x0 + x1) / 2, 2), round((y0 + y1) / 2, 2), z]}}
browser = json.dumps(course).replace(".0,", ",").replace(".0]", "]")     # JSON.stringify(0.0) == "0"
def call(method, path, body=None):
    req = urllib.request.Request(api + path, body.encode() if isinstance(body, str) else None, {"Content-Type": "application/json"}, method=method)
    return json.load(urllib.request.urlopen(req))
print("  saved:", call("PUT", "/configs/courses/gate3_loop", browser)["path"])
print("  lint after save:", call("GET", "/courses/gate3_loop/lint"))
pv = call("POST", "/courses/preview", json.dumps({"course": course, "scene": scene}))
print(f"  preview: {pv['duration_solved']} s, v_max {pv['stats']['v_max']} m/s, min clearance {pv['clearance'] and pv['clearance']['min']} m, outside {pv['inside'] and pv['inside']['outside_frac']}")
job = call("POST", "/jobs/figs", json.dumps({"scene": scene, "course": "gate3_loop", "from_step": "course", "stop_after": "record",
                                              "redo": ["course", "simulate", "validate"]}))["id"]
print(f"  job {job} queued")
while (s := call("GET", f"/jobs/{job}")["status"]) in ("queued", "running"):
    time.sleep(3)
print(f"  job {job}: {s}")
for l in call("GET", f"/jobs/{job}/log")[-40:]:
    t = l["line"]
    if any(k in t for k in ("✔", "✗", "!", "tracking", "keyframe", "start", "k1", "k2", "k3", "end ", "pixel", "run record", "failed")):
        print("   ", t)
r = call("GET", f"/scenes/{scene}")["results"]
print("  results.course:", r.get("course"), "\n  results.sim:", {k: r.get("sim", {}).get(k) for k in ("frames", "duration_s", "track_err_max_m", "pixel_std", "dark_frames", "peak_vram_mib")})
EOF
echo "=== done: now do the browser half (Course editor → $SCENE → New loop → move a keyframe → Save and fly)"
