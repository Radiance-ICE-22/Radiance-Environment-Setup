#!/usr/bin/env bash
# Phase 4 gate on dummy: SOUS-VIDE's learning half, unchanged upstream code, end to end.
#
#   1. pull, refresh the backend venv, tests
#   2. svnet_pipeline.py preflight in kitchen (imports, configs, scene, disk estimate)
#   3. through Galley's queue: cohort p4_smoke = backroom + circuit, data_alpha, Maverick,
#      notebook epochs (histNet 200, commNet 300), in-loop and final evaluation eval_single
#   4. print rollout, training and evaluation results
#
# Long: rollouts, two trainings and evaluations — expect a few hours on the RTX 3050 Ti.
# Run it detached and check the log (or the SV-Net page, through the usual tunnel):
#   tmux new -d -s p4 'bash ~/FYP-Radiance/ui/deploy/phase4_gate.sh > ~/phase4_gate.log 2>&1'
#
# Changes: ~/FYP-Radiance pulled; SousVide/cohorts/p4_smoke/ (~6 GB with data_alpha on
# circuit); ~/projects/figs_validation/.svnet_pipeline_state/p4_smoke/; cohorts/ added to
# SousVide/.git/info/exclude. kitchen is untouched. Galley serves on :8800 while it runs, so
# stop any other Galley first.
#
# Overrides: COHORT=… SCENE=… COURSE=… METHOD=… HIST=… COMM=… EVAL=…
#
# Paths per machine (dummy, intellisense08) come from host.sh next to this script.
set -u
source "$(dirname "$(readlink -f "$0")")/host.sh"
ROOT=$FIGS_ROOT
UI=$RADIANCE_REPO/ui
PIPE=$RADIANCE_REPO/figs/svnet_pipeline.py
COHORT=${COHORT:-p4_smoke}; SCENE=${SCENE:-backroom}; COURSE=${COURSE:-circuit}; METHOD=${METHOD:-data_alpha}
HIST=${HIST:-200}; COMM=${COMM:-300}; EVAL=${EVAL:-eval_single}
API=http://127.0.0.1:8800/api
stamp() { date '+%F %T'; }

echo "=== $(stamp) sync FYP-Radiance"
cd $RADIANCE_REPO && git pull --ff-only && git log --oneline -3 || exit 1
[ -f $PIPE ] || { echo "figs/svnet_pipeline.py missing: push the Phase 4 commit first"; exit 1; }

echo; echo "=== backend venv + tests"
cd $UI/backend
UV=$(command -v uv || echo ~/.local/bin/uv)
[ -d .venv ] || $UV venv -q .venv
$UV pip install -q -p .venv -e '.[test]'
GALLEY_TEST_CONFIGS=$ROOT/SousVide/configs .venv/bin/python -m pytest -q 2>&1 | tail -2

echo; echo "=== $(stamp) preflight in kitchen (imports, configs, scene, size estimate)"
( source $ROOT/figs_env.sh >/dev/null 2>&1 && python $PIPE --project-root $ROOT --cohort $COHORT --scene $SCENE --courses $COURSE \
    --method $METHOD --hist-epochs $HIST --comm-epochs $COMM --comm-eval $EVAL --deploy-method $EVAL --only preflight ) \
  || { echo "preflight failed: fix the above first"; exit 1; }

echo; echo "=== $(stamp) start Galley"
.venv/bin/python -m galley > /tmp/galley_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 30); do curl -sf $API/health >/dev/null && break; sleep 0.5; done

echo; echo "=== $(stamp) queue the cohort through the API (what the SV-Net page's 'Queue the full run' sends)"
python3 - "$API" "$COHORT" <<'EOF'
import json, sys, time, urllib.request
api, cohort = sys.argv[1], sys.argv[2]
def call(method, path, body=None):
    req = urllib.request.Request(api + path, json.dumps(body).encode() if body else None,
                                 {"Content-Type": "application/json"}, method=method)
    return json.load(urllib.request.urlopen(req))
jid = call("POST", "/jobs/svnet", {"cohort": cohort})        # settings were saved by the preflight
print(f"  job {jid['id']} queued; following its log (progress redraws are not stored)")
seen, t0 = -1, time.time()
while True:
    for l in call("GET", f"/jobs/{jid['id']}/log?after={seen}"):
        seen = l["seq"]
        line = l["line"]
        if " epoch " in line and not any(f" epoch {k}/" in line for k in (1, 50, 100, 150, 200, 250, 300)):
            continue                                      # keep the log short: a few epochs per network
        print("   ", line, flush=True)
    s = call("GET", f"/jobs/{jid['id']}")["status"]
    if s not in ("queued", "running"):
        break
    time.sleep(10)
print(f"  job {jid['id']}: {s} after {(time.time() - t0) / 60:.0f} min")
c = call("GET", f"/cohorts/{cohort}")
r = c["results"]
print("  steps done:", [x["step"] for x in c["steps"] if x["done"]], "  disk GB:", c["disk_gb"])
if "estimate" in r: print("  estimate:", r["estimate"])
if "rollout" in r: print("  rollout:", json.dumps(r["rollout"]))
for net in ("histNet", "commNet"):
    t = r.get(f"train_{net}")
    if t:
        for p, v in t["pilots"].items():
            print(f"  {net} {p}: {v['epochs']} epochs, train {v['loss_train'][-1][1] if v['loss_train'] else None}, "
                  f"test {v['loss_test'][-1][1] if v['loss_test'] else None}, {v['n_train']}/{v['n_test']} samples, "
                  f"{t['wallclock']}, peak {t['peak_vram_mib']} MiB, upstream eval {v['eval_tte_upstream']}")
if "deploy" in r:
    for p, v in r["deploy"]["pilots"].items():
        print(f"  deploy {p:<10} {v['role']:<7} tracking mean {v['tte']['mean_m']} m, max {v['tte']['max_m']} m, "
              f"{v['tte']['within_0.3m']*100:.0f}% within 0.3 m | upstream TTE {v['upstream_tte_mean']} PP {v['upstream_pp']} | "
              f"{v['hz_mean']} Hz | {v['video']}")
    print(f"  deploy: {r['deploy']['wallclock']}, peak {r['deploy']['peak_vram_mib']} MiB")
EOF
echo "=== $(stamp) done"
