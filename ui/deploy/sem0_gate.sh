#!/usr/bin/env bash
# Semantic embeddings, Phase 0 gate (docs/SEMANTICS_PLAN.md): environment and pose check.
#
#   1. pull the repo
#   2. setup_scripts/install_semantics.sh — OpenCLIP 2.24 + DINOv2 + radiance_semantics into
#      kitchen under a freeze-everything constraints file; weights cached under the prefix
#   3. verify_figs.sh --quick --semantics — the usual imports/GPU checks plus CLIP/DINOv2 smoke
#      and the quick gsplat N-channel gradient probe
#   4. probe on 1.5 M synthetic Gaussians and on SCENE's real splat (480x270 and 960x540):
#      gradient identities (incl. 64 channels with grad via 32-channel chunking), ms per 32-channel forward+backward, peak VRAM, lift-time estimate
#   5. camera check on SCENE: 10 training views rendered with the refined (SO3xR3) pose and the
#      raw transforms.json pose, PSNR against the training images, held-out PSNR, correction size
#   6. verdict
#
# About 5–15 minutes once weights are cached (first run downloads ~0.7 GB). Uses the GPU, so run
# it while Galley's queue is idle:
#   tmux new -d -s sem0 'bash ~/Radiance/Radiance-Environment-Setup/ui/deploy/sem0_gate.sh > ~/sem0_gate.log 2>&1'
#
# Changes: the repo is pulled; kitchen gains open_clip_torch 2.24.0, ftfy, regex,
# huggingface_hub<1, tqdm (if missing) and the editable radiance_semantics — nothing already
# installed moves (checked before/after); weights in <prefix>/.cache; JSON records in
# SousVide/runs/semantics_p0_*.json; image strips in gsplats/workspace/<scene>/semantics/<run>/p0_cameras/.
#
# Overrides: SCENE=… RES=… VIEWS=… SKIP_PULL=1
set -u
source "$(dirname "$(readlink -f "$0")")/host.sh"
ROOT=$FIGS_ROOT
REPO=$RADIANCE_REPO
SCENE=${SCENE:-backroom}; RES=${RES:-480x270,960x540}; VIEWS=${VIEWS:-10}
stamp() { date '+%F %T'; }
declare -A R

echo "=== $(stamp) sync $REPO"
if [ -z "${SKIP_PULL:-}" ]; then
  cd "$REPO" && git pull --ff-only && git log --oneline -3 || exit 1
fi
[ -f "$REPO/semantics/radiance_semantics/probe.py" ] || { echo "semantics/ missing: push the Phase 0 commit first"; exit 1; }

echo; echo "=== $(stamp) install_semantics.sh"
bash "$REPO/setup_scripts/install_semantics.sh" --prefix "$ROOT"; R[install]=$?
[ "${R[install]}" -eq 0 ] || { echo "install failed: fix the above first (nothing else was run)"; exit 1; }

echo; echo "=== $(stamp) verify_figs.sh --quick --semantics"
bash "$REPO/setup_scripts/verify_figs.sh" --prefix "$ROOT" --quick --semantics; R[verify]=$?

set +u
# shellcheck disable=SC1091
source "$ROOT/figs_env.sh" >/dev/null 2>&1
set -u
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$used" ] && [ "$used" -gt 1000 ]; then
  echo "! ${used} MiB of VRAM already in use (Galley job? ns-viewer?): peak numbers below include it"
fi

echo; echo "=== $(stamp) probe: synthetic, 1.5 M Gaussians"
python -m radiance_semantics.probe --project-root "$ROOT" --res "$RES" --record | grep -v '^GALLEY_JSON'
R[probe_synthetic]=${PIPESTATUS[0]}

echo; echo "=== $(stamp) probe: $SCENE's trained splat"
python -m radiance_semantics.probe --project-root "$ROOT" --scene "$SCENE" --res "$RES" --record | grep -v '^GALLEY_JSON'
R[probe_scene]=${PIPESTATUS[0]}

echo; echo "=== $(stamp) cameras: refined vs raw poses on $SCENE"
python -m radiance_semantics.cameras --project-root "$ROOT" --scene "$SCENE" --views "$VIEWS" --save-images --record \
  | grep -v '^GALLEY_JSON'
R[cameras]=${PIPESTATUS[0]}

echo; echo "=== $(stamp) Phase 0 gate"
fails=0
for k in install verify probe_synthetic probe_scene cameras; do
  if [ "${R[$k]}" -eq 0 ]; then printf '  PASS  %s\n' "$k"; else printf '  FAIL  %s (exit %s)\n' "$k" "${R[$k]}"; fails=$((fails + 1)); fi
done
echo "  records: $(ls -1t "$ROOT"/SousVide/runs/semantics_p0_*.json 2>/dev/null | head -3 | tr '\n' ' ')"
if [ "$fails" -eq 0 ]; then
  echo "  Phase 0 gate PASSED — send this log and the three JSON records back; Phase 1 can start."
else
  echo "  Phase 0 gate FAILED ($fails) — send this log back."
fi
exit $(( fails > 0 ))
