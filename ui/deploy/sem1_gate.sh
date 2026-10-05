#!/usr/bin/env bash
# Semantic embeddings, Phase 1 gate (docs/SEMANTICS_PLAN.md): teachers + lift + CLI query.
#
#   1. pull the repo; install_semantics.sh (idempotent; refreshes the editable package)
#   2. semantic_pipeline.py --scene SCENE: preflight, cameras, teachers (CLIP pyramid + DINOv2 for
#      every frame — the long part), lift (refined poses, 32-channel chunks), export (.splat order)
#   3. the five annotated objects (queries.json, course frame): `semantic_query.py --eval`
#   4. verdict: pipeline within 8 GB, and at least 4 of 5 objects hit
#
# Annotations first (once): in Galley's course editor on SCENE, Show ▸ Splat, put the Goal marker on
# each object, then on this machine:
#   source ~/Radiance/figs/figs_env.sh
#   python -m radiance_semantics.annotations --scene backroom init
#   python -m radiance_semantics.annotations --scene backroom set "red tool chest" X Y Z    # ×5
# Without them the pipeline still runs and the gate stops at step 3 with PENDING.
#
# Uses the GPU for tens of minutes (teachers dominate). Run while Galley's queue is idle:
#   tmux new -d -s sem1 'bash ~/Radiance/Radiance-Environment-Setup/ui/deploy/sem1_gate.sh > ~/sem1_gate.log 2>&1'
#
# Changes: kitchen gets the updated editable package (nothing else moves); teacher features in
# gsplats/workspace/<scene>/semantics/teachers/<tag>/ (~1–2 GB), the table in .../semantics/<run>/lift/
# and lift_raw/; records SousVide/runs/semantics_p1_*.json.
#
# Overrides: SCENE=… FEAT_WIDTH=… NEED=… SKIP_PULL=1 EXTRA="--redo teachers"
set -u
source "$(dirname "$(readlink -f "$0")")/host.sh"
ROOT=$FIGS_ROOT
REPO=$RADIANCE_REPO
SCENE=${SCENE:-backroom}; FEAT_WIDTH=${FEAT_WIDTH:-960}; NEED=${NEED:-4}
stamp() { date '+%F %T'; }
declare -A R

echo "=== $(stamp) sync $REPO"
if [ -z "${SKIP_PULL:-}" ]; then
  cd "$REPO" && git pull --ff-only && git log --oneline -3 || exit 1
fi
[ -f "$REPO/figs/semantic_pipeline.py" ] || { echo "figs/semantic_pipeline.py missing: push the Phase 1 commit first"; exit 1; }

echo; echo "=== $(stamp) install_semantics.sh"
bash "$REPO/setup_scripts/install_semantics.sh" --prefix "$ROOT" | tail -4; R[install]=${PIPESTATUS[0]}
[ "${R[install]}" -eq 0 ] || { echo "install failed: run it by hand to see why"; exit 1; }

set +u
# shellcheck disable=SC1091
source "$ROOT/figs_env.sh" >/dev/null 2>&1
set -u
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$used" ] && [ "$used" -gt 1000 ]; then
  echo "! ${used} MiB of VRAM already in use (Galley job? ns-viewer?): peak numbers below include it"
fi

echo; echo "=== $(stamp) semantic_pipeline.py --scene $SCENE"
# shellcheck disable=SC2086
python "$REPO/figs/semantic_pipeline.py" --project-root "$ROOT" --scene "$SCENE" --feat-width "$FEAT_WIDTH" ${EXTRA:-}
R[pipeline]=$?

if [ "${R[pipeline]}" -eq 0 ]; then
  echo; echo "=== $(stamp) annotated queries (semantic_query.py --eval)"
  python -m radiance_semantics.annotations --project-root "$ROOT" --scene "$SCENE" init >/dev/null
  python "$REPO/figs/semantic_query.py" --project-root "$ROOT" --scene "$SCENE" --eval --need "$NEED"
  R[queries]=$?
else
  R[queries]=99
fi

echo; echo "=== $(stamp) Phase 1 gate"
python "$REPO/figs/semantic_pipeline.py" --project-root "$ROOT" --scene "$SCENE" --status | sed 's/^/  /'
fails=0; pending=0
for k in install pipeline queries; do
  case "${R[$k]}" in
    0)  printf '  PASS     %s\n' "$k" ;;
    3)  printf '  PENDING  %s (annotations missing — see above)\n' "$k"; pending=1 ;;
    99) printf '  SKIPPED  %s (pipeline failed)\n' "$k"; fails=$((fails + 1)) ;;
    *)  printf '  FAIL     %s (exit %s)\n' "$k" "${R[$k]}"; fails=$((fails + 1)) ;;
  esac
done
echo "  records: $(ls -1t "$ROOT"/SousVide/runs/semantics_p1_*.json 2>/dev/null | head -3 | tr '\n' ' ')"
if [ "$fails" -eq 0 ] && [ "$pending" -eq 0 ]; then
  echo "  Phase 1 gate PASSED — send this log and the records back."
elif [ "$fails" -eq 0 ]; then
  echo "  Pipeline done; gate PENDING annotations. Set the positions, then re-run with SKIP_PULL=1 (steps are cached)."
else
  echo "  Phase 1 gate FAILED ($fails) — send this log back."
fi
exit $(( fails > 0 ? 1 : (pending > 0 ? 3 : 0) ))
