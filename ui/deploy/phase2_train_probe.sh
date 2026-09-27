#!/usr/bin/env bash
# Phase 2 probe on dummy: can splatfacto train on a 4 GB card, and at what settings?
#
# Retrains the shipped `backroom` capture as a separate scene, `backroom_t4`, whose
# workspace is a symlink to backroom's: SfM is reused, and backroom itself is never
# touched. Tries settings from closest-to-reference to most frugal and stops at the first
# that completes, then flies circuit through the result for comparison with backroom.
#
# Long (roughly 1 h per attempt). Run it detached:
#   tmux new -d -s probe 'bash ~/FYP-Radiance/ui/deploy/phase2_train_probe.sh > ~/phase2_probe.log 2>&1'
set -u
ROOT=~/projects/figs_validation
REPO=$ROOT/SousVide
PIPE=~/FYP-Radiance/figs/figs_pipeline.py
SCENE=backroom_t4
LOGDIR=~/phase2_probe_logs; mkdir -p $LOGDIR
stamp() { date '+%F %T'; }

echo "=== $(stamp) sync FYP-Radiance"
cd ~/FYP-Radiance
if [ -d ui ] && ! git ls-files --error-unmatch ui/README.md >/dev/null 2>&1; then
  mv ui /tmp/ui.scp.$$ && echo "moved the scp'd ui/ aside so the committed one can be pulled"
fi
git pull --ff-only || exit 1
[ -d /tmp/ui.scp.$$/backend/.venv ] && [ ! -d ui/backend/.venv ] && mv /tmp/ui.scp.$$/backend/.venv ui/backend/
git log --oneline -3
python3 $PIPE --scene x --list-steps | grep -q "^  train" || { echo "figs_pipeline.py has no train step — push the split first"; exit 1; }

source $ROOT/figs_env.sh || exit 1
export TERM=dumb
echo; echo "=== $(stamp) nerfstudio options available"
ns-train splatfacto --help 2>/dev/null | grep -oE -- "--(max-num-iterations|pipeline\.datamanager\.cache-images|pipeline\.model\.stop-split-at|pipeline\.model\.cull-alpha-thresh|vis) [^ ]*" | sort -u
ns-train splatfacto nerfstudio-data --help 2>/dev/null | grep -oE -- "--downscale-factor [^ ]*" | head -1

echo; echo "=== $(stamp) scene $SCENE -> workspace symlink to backroom"
cd $REPO/gsplats/workspace
[ -e $SCENE ] || ln -s backroom $SCENE
ls -la $SCENE; ls backroom/images | wc -l; python3 -c "import json;t=json.load(open('backroom/transforms.json'));print('frames',len(t['frames']),'size',t.get('w'),'x',t.get('h'))"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

ATTEMPTS=(
  "A_auto_cpucache|--cache-images cpu"
  "B_split10k|--cache-images cpu --train-arg=--pipeline.model.stop-split-at=10000"
  "C_down4|--cache-images cpu --downscale 4"
)
WIN=""
for a in "${ATTEMPTS[@]}"; do
  name=${a%%|*}; opts=${a#*|}
  echo; echo "=== $(stamp) attempt $name: $opts"
  python3 $PIPE --project-root $ROOT --scene $SCENE --only train --redo train --archive-old --train-vis tensorboard $opts 2>&1 \
    | tr '\r' '\n' | grep -vE '^\s*$' > $LOGDIR/$name.log
  rc=${PIPESTATUS[0]}
  grep -E "cmd:|peak VRAM|trained in|out of memory|OutOfMemory|failed|Error" $LOGDIR/$name.log | tail -8
  echo "rc=$rc"
  if [ $rc -eq 0 ]; then WIN=$name; break; fi
  grep -qiE "out of memory|OutOfMemory" $LOGDIR/$name.log || { echo "not an OOM — stopping here, see $LOGDIR/$name.log"; break; }
done

if [ -n "$WIN" ]; then
  echo; echo "=== $(stamp) $WIN trained; verify + fly circuit through $SCENE"
  python3 $PIPE --project-root $ROOT --scene $SCENE --course circuit --from verify 2>&1 \
    | tr '\r' '\n' | grep -vE "ACADOS|Please export|officially supported|incompatibility|currently in use|Warning|warn|pkg_resources|^\s*$" \
    | grep -E "registered|sparse points|checkpoint|tracking error|frames|pixel|near-black|✔|✗" | head -30
  echo; echo "=== comparison (run records)"
  python3 - <<'PY'
import json, glob, os
for f in sorted(glob.glob(os.path.expanduser("~/projects/figs_validation/SousVide/runs/backroom*.json"))):
    r = json.load(open(f)); s = r.get("sim", {}); t = r.get("train", {})
    print(f"{os.path.basename(f):40s} train_peak={r.get('train_peak_vram_mib')} ckpt_mb={r.get('sfm',{}).get('ckpt_mb')} "
          f"track_max={s.get('track_err_max_m')} pixel_std={s.get('pixel_std')} dark={s.get('dark_frames')} train={t.get('wallclock')}")
PY
fi
echo; echo "=== $(stamp) done. Per-attempt logs: $LOGDIR"
