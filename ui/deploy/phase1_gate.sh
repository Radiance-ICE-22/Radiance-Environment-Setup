#!/usr/bin/env bash
# Phase 1 gate on dummy: install the backend in its own venv, run its tests, start it,
# drive it with curl, stop it. Changes: ~/FYP-Radiance updated to origin/main,
# ~/FYP-Radiance/ui/backend/.venv, ~/.local/share/galley/galley.db. kitchen is untouched.
set -u
UI=~/FYP-Radiance/ui
UV=$(command -v uv || echo ~/.local/bin/uv)
PORT=8800
API=http://127.0.0.1:$PORT/api
export GALLEY_MACHINE=$UI/machines/dummy.toml
j() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1]))" "$1"; }

echo "=== sync FYP-Radiance with GitHub (drops the two local patches, now committed upstream)"
cd ~/FYP-Radiance && git checkout -- figs/figs_pipeline.py && git pull --ff-only && git log --oneline -3

echo; echo "=== backend venv (uv: $UV)"
cd $UI/backend
$UV venv -q .venv && $UV pip install -q -p .venv -e '.[test]' && .venv/bin/python -c "import fastapi,uvicorn;print('fastapi',fastapi.__version__)"

echo; echo "=== unit tests (+ round-trip every real config on this machine)"
GALLEY_TEST_CONFIGS=~/projects/figs_validation/SousVide/configs .venv/bin/python -m pytest -q 2>&1 | tail -5

echo; echo "=== start server"
.venv/bin/python -m galley > /tmp/galley_server.log 2>&1 &
SRV=$!
for i in $(seq 30); do curl -sf $API/health >/dev/null && break; sleep 0.5; done
curl -s $API/health; echo

echo; echo "=== machine, scenes, configs, runs"
curl -s $API/machine | j "d['gpu'], d['disk']"
curl -s $API/scenes | j "[(s['scene'], s['models'], 'loadable' if s['loadable'] else 'NOT loadable') for s in d]"
curl -s $API/configs | j "{k: [c['name'] for c in v] for k, v in d.items()}"
curl -s "$API/runs?scene=backroom" | j "[(r['_file'], r['sim']['track_err_max_m']) for r in d]"
curl -s $API/scenes/backroom | j "[s['step'] for s in d['steps'] if s['done']]"

echo; echo "=== job 1: figs_pipeline preflight through the queue"
ID=$(curl -s -X POST $API/jobs/figs -H 'content-type: application/json' -d '{"scene":"backroom","only":"preflight"}' | j "d['id']")
for i in $(seq 60); do S=$(curl -s $API/jobs/$ID | j "d['status']"); [ "$S" = running ] || [ "$S" = queued ] || break; sleep 1; done
echo "job $ID: $S"
curl -s "$API/jobs/$ID/log" | j "'\n'.join(l['line'] for l in d[-12:])"

echo; echo "=== job 2: selftest 60 s, cancel after 4 s"
ID=$(curl -s -X POST $API/jobs/selftest -H 'content-type: application/json' -d '{"seconds":60}' | j "d['id']")
sleep 4
curl -s "$API/jobs/$ID/log" | j "[l['line'] for l in d]"
curl -s -X POST $API/jobs/$ID/cancel | j "d['status'], d['returncode']"
sleep 1; curl -s $API/jobs/$ID | j "d['status']"

echo; echo "=== job 3: bad requests are refused"
for body in '{"scene":"../x"}' '{"scene":"x","video":"../../etc/passwd"}' '{"scene":"x","only":"gsplat","from_step":"course"}'; do
  printf '%s -> ' "$body"; curl -s -o /dev/null -w '%{http_code}\n' -X POST $API/jobs/figs -H 'content-type: application/json' -d "$body"
done

echo; echo "=== stop server"
kill $SRV; wait $SRV 2>/dev/null
tail -5 /tmp/galley_server.log
echo "=== done"
