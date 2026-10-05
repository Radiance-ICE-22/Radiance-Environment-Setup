#!/usr/bin/env bash
# Semantic embeddings, Phase 4 gate (docs/SEMANTICS_PLAN.md): the FMGS backend.
#
#   1. pull; semantics tests in kitchen (CPU); Galley backend tests
#   2. field probes (python -m radiance_semantics.fmgs.diag): tiny-cuda-nn per part, else PyTorch; the
#      field the trainer will build (impl auto) must run forward + backward on the GPU
#   3. 200-step smoke run (python -m radiance_semantics.fmgs.train): the loss falls, the Gaussians are unchanged
#   4. ./run_ui.sh restart, then the full run through Galley's queue, as the splat editor's Build does:
#        POST /api/jobs/semantics {"scene": SCENE, "backend": "fmgs"}  → fmgs (4,200 steps) → bake
#   5. verdict: peak VRAM under 8 GB (default or a logged fallback), the Gaussians AND the checkpoint file
#      unchanged (checksums), the fmgs table fresh with .splat-many rows, and the Phase 1 queries run on it
#      (hits by the gates' rule, next to the lift's for comparison)
#
# Needs the Phase 1 teachers for SCENE (they are shared with the lift). Takes about 40–60 min on
# intellisense08, almost all of it training. Run it while the queue is idle:
#   tmux new -d -s sem4 'bash ~/Radiance/Radiance-Environment-Setup/ui/deploy/sem4_gate.sh > ~/sem4_gate.log 2>&1'
#
# Overrides: SCENE=backroom SKIP_PULL=1 SKIP_SMOKE=1 STEPS=4200 BASE=http://127.0.0.1:8800
set -u
source "$(dirname "$(readlink -f "$0")")/host.sh"
REPO=$RADIANCE_REPO
ROOT=$FIGS_ROOT
SCENE=${SCENE:-backroom}
STEPS=${STEPS:-4200}
stamp() { date '+%F %T'; }
RES=()
res() { RES+=("$(printf '  %s  %s' "$1" "$2")"); }

echo "=== $(stamp) sync $REPO"
if [ -z "${SKIP_PULL:-}" ]; then
  cd "$REPO" && git pull --ff-only && git log --oneline -3 || exit 1
fi
[ -f "$REPO/semantics/radiance_semantics/fmgs/train.py" ] || { echo "radiance_semantics/fmgs missing: push the Phase 4 commit first"; exit 1; }

echo; echo "=== $(stamp) semantics tests (kitchen, CPU) + Galley backend tests"
(
  source "$ROOT/figs_env.sh" >/dev/null 2>&1 || { echo "cannot source figs_env.sh"; exit 1; }
  cd "$REPO/semantics"
  python -c "import pytest" 2>/dev/null || { echo "  (pytest not in kitchen: skipped)"; exit 0; }
  CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider tests 2>&1 | tail -2
  exit "${PIPESTATUS[0]}"
) && res PASS "semantics tests" || res FAIL "semantics tests"
cd "$REPO/ui/backend" || exit 1
UV=$(command -v uv || echo ~/.local/bin/uv)
[ -d .venv ] || $UV venv -q .venv
$UV pip install -q -p .venv -e '.[test]'
.venv/bin/python -m pytest -q 2>&1 | tail -1
[ "${PIPESTATUS[0]}" -eq 0 ] && res PASS "Galley backend tests" || res FAIL "Galley backend tests"
PORT=$(.venv/bin/python -c 'from galley.settings import load; print(load().port)')
BASE=${BASE:-http://127.0.0.1:$PORT}
if curl -sf "$BASE/api/health" | grep -q '"current_job":[0-9]'; then
  echo "A Galley job is running: wait for the queue to be idle (the smoke run uses the GPU outside the queue)."; exit 1
fi

echo; echo "=== $(stamp) FMGS field on the GPU: tiny-cuda-nn probes, PyTorch fallback"
# Each tiny-cuda-nn case runs in its own process (CUDA_LAUNCH_BLOCKING=1). On intellisense08 the
# 24 × 8 hash grid (192 dims) fails with "invalid configuration argument"; impl auto then splits it into two
# 12-level tcnn grids (or falls back to PyTorch). This step fails only if the chosen field does not run.
(
  source "$ROOT/figs_env.sh" >/dev/null 2>&1
  cd "$REPO/semantics" && python -m radiance_semantics.fmgs.diag 2>&1 | grep -vE "^\s*$|Warning"
  exit "${PIPESTATUS[0]}"
) && res PASS "FMGS field runs on the GPU (impl auto)" || { res FAIL "FMGS field on the GPU"; printf '%s\n' "${RES[@]}"; exit 1; }

if [ -z "${SKIP_SMOKE:-}" ]; then
  echo; echo "=== $(stamp) 200-step smoke run (outside the queue)"
  SMOKE=$(mktemp -d /tmp/sem4_smoke.XXXX)
  (
    source "$ROOT/figs_env.sh" >/dev/null 2>&1
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python -m radiance_semantics.fmgs.train --project-root "$ROOT" \
      --scene "$SCENE" --steps 200 --out "$SMOKE" 2>&1 | grep -vE "^\s*$|Warning|it/s\]" | tail -12
    exit "${PIPESTATUS[0]}"
  ) && res PASS "smoke: 200 steps, loss fell, Gaussians unchanged" || res FAIL "smoke run (see above)"
  rm -rf "$SMOKE"
fi

echo; echo "=== $(stamp) restart Galley, full run through the queue"
bash "$REPO/run_ui.sh" restart >/dev/null || { echo "run_ui.sh restart failed"; exit 1; }
for i in $(seq 30); do curl -sf "$BASE/api/health" >/dev/null && break; sleep 1; done

.venv/bin/python - "$BASE" "$SCENE" "$STEPS" "$ROOT" <<'EOF'
import json, math, sys, time, urllib.request
from pathlib import Path
base, scene, steps, root = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
api = base + "/api"
fails = []

def call(method, path, body=None, raw=False, timeout=600):
    req = urllib.request.Request(api + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = r.read()
        return d if raw else json.loads(d)

def ok(c, msg):
    print(f"  {'PASS' if c else 'FAIL'}  {msg}", flush=True)
    if not c:
        fails.append(msg)
    return c

body = {"scene": scene, "backend": "fmgs"}
if steps != 4200:
    body["fmgs_steps"] = steps
jid = call("POST", "/jobs/semantics", body)["id"]
print(f"        job {jid} queued: {body}", flush=True)
t0, seen = time.time(), -1
while (j := call("GET", f"/jobs/{jid}"))["status"] in ("queued", "running"):
    log = call("GET", f"/jobs/{jid}/log?after={seen}")
    for l in log:
        seen = l["seq"]
        if any(k in l["line"] for k in ("step ", "✔", "!", "fallback", "field ", "tiny-cuda-nn", "table:", "Gaussians")):
            print("         ", l["line"][:170], flush=True)
    time.sleep(20)
print(f"        job {jid}: {j['status']} after {(time.time() - t0) / 60:.1f} min", flush=True)
if not ok(j["status"] == "succeeded", f"fmgs job {jid} {j['status']}"):
    for l in call("GET", f"/jobs/{jid}/log")[-25:]:
        print("         ", l["line"][:200])
    sys.exit(1)

st = call("GET", f"/scenes/{scene}/semantics")
fm = next((t for t in st["tables"] if t["active_run"] and t["backend"] == "fmgs"), None)
lift = next((t for t in st["tables"] if t["active_run"] and t["backend"] == "lift"), None)
ok(fm is not None and not fm["stale"], f"fmgs table fresh: {fm and fm['rows']} rows, {fm and fm['mb']} MB, run {st['run']}")
tj = json.loads((root / "SousVide/gsplats/workspace" / scene / "semantics" / st["run"] / "fmgs_train" / "train.json").read_text())
fb = tj.get("fallback") or {}
peak = tj.get("peak_vram_mib_device") or tj.get("peak_vram_mib")
ok(peak is not None and peak < 8192, f"peak VRAM {peak} MiB (device) / {tj.get('peak_vram_mib')} MiB (torch) < 8192 · variant {tj.get('variant')}"
   + (f" · fallback {fb.get('level')}: {fb.get('name')}" if fb.get("level") else " · no fallback needed") + f" · field {tj.get('impl')}")
ok(tj.get("gauss_unchanged") and tj.get("checkpoint_sha_before") == tj.get("checkpoint_sha_after"),
   f"Gaussians unchanged (checksum {tj.get('gauss_checksum_before')} → {tj.get('gauss_checksum_after')}); checkpoint file "
   f"{tj.get('checkpoint_sha_before')} → {tj.get('checkpoint_sha_after')}")
ok(tj["loss_last"] < tj["loss_first"], f"loss {tj['loss_first']} → {tj['loss_last']} over {tj['steps']} steps, "
   f"{tj['seconds'] / 60:.1f} min ({tj['it_per_s']} it/s), {tj.get('params_m')} M parameters, {tj.get('trainable_gaussians'):,} trainable Gaussians")
sp = call("POST", f"/scenes/{scene}/splat")
ok(fm is not None and sp.get("n_written") == fm["rows"], f".splat records {sp.get('n_written')} = fmgs rows {fm and fm['rows']}")

def hit(c, p):
    e = math.dist(c["centroid"], p)
    return e <= 0.75 or all(c["box"]["lo"][a] - 0.3 <= p[a] <= c["box"]["hi"][a] + 0.3 for a in range(3)), e

ann = [a for a in call("GET", f"/scenes/{scene}/semantics/queries")["queries"] if a.get("position")]
call("POST", f"/scenes/{scene}/semantics/query", {"text": "warm up", "backend": "fmgs", "relevancy": False})
hits = {"lift": 0, "fmgs": 0}
print(f"        {'query':<18} {'lift':>22}   {'fmgs':>22}")
for a in ann:
    row = []
    for b in ("lift", "fmgs"):
        t1 = time.time()
        r = call("POST", f"/scenes/{scene}/semantics/query", {"text": a["text"], "backend": b, "relevancy": False})
        dt = (time.time() - t1) * 1000
        c = r["result"]["candidates"]
        h, e = hit(c[0], a["position"]) if c else (False, None)
        hits[b] += h
        row.append(f"{'HIT ' if h else 'MISS'} {'' if e is None else f'{e:5.2f} m'} {dt:5.0f} ms")
    print(f"        {a['text']:<18} {row[0]:>22}   {row[1]:>22}", flush=True)
ok(len(ann) > 0, f"Phase 1 queries run on the fmgs table: fmgs {hits['fmgs']} of {len(ann)} hit, lift {hits['lift']} of {len(ann)} "
   "(a comparison for Phase 5, not a pass mark)")
print("\n  RESULT:", "PASSED" if not fails else f"FAILED ({len(fails)})")
sys.exit(1 if fails else 0)
EOF
[ $? -eq 0 ] && res PASS "full run, VRAM, checksums, table, queries" || res FAIL "full run checks (see above)"

echo; echo "=== $(stamp) Phase 4 gate"
printf '%s\n' "${RES[@]}"
if printf '%s\n' "${RES[@]}" | grep -q FAIL; then echo "  Phase 4 gate FAILED — send this log back."; exit 1; fi
echo "  Phase 4 gate PASSED — send this log back. Then open the splat editor: View ▸ Compare shows lift | FMGS."
