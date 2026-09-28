#!/usr/bin/env bash
# bringup.sh — bring an existing FiGS install up to date with this repo, and check it end to end.
#
# For a machine where install_figs.sh has already run (intellisense08: ~/Radiance/figs). Paths come
# from host.sh by hostname. It does not reinstall anything and never touches torch or the kitchen
# env's packages. In order:
#
#   1. pull this repo (fast-forward only)
#   2. report the SousVide clone: commit, local changes, distance from the pinned upstream commit
#   3. check the install: tiny-cuda-nn built for this GPU, colmap, verify_figs.sh
#   4. copy our captures/courses from figs/sousvide_overlay into SousVide (never overwrites)
#   5. Galley's backend in its own venv (uv) + its tests against this machine's real configs
#   6. figs_pipeline.py: fly backroom + circuit (the known-good pair) and validate the video
#   7. start Galley once, query it, stop it
#
#   bash ~/Radiance/Radiance-Environment-Setup/ui/deploy/bringup.sh 2>&1 | tee ~/bringup.log
#
# Changes: the repo checkout (pull), SousVide/configs (overlay files that are missing),
# ui/backend/.venv, ~/.local/share/galley/galley.db, one backroom flight (outputs/flights/,
# runs/*.json, .figs_pipeline_state/backroom/).
set -u
source "$(dirname "$(readlink -f "$0")")/host.sh"
ROOT=$FIGS_ROOT
REPO=$RADIANCE_REPO
SV=$ROOT/SousVide
stamp() { date '+%F %T'; }
say() { echo; echo "=== $(stamp) $*"; }
PROBLEMS=()

say "host $(hostname): FIGS_ROOT=$ROOT  REPO=$REPO  GALLEY_MACHINE=$GALLEY_MACHINE"
[ -f "$ROOT/figs_env.sh" ] || { echo "no $ROOT/figs_env.sh — run setup_scripts/install_figs.sh --prefix $ROOT first"; exit 1; }
[ -f "$GALLEY_MACHINE" ] || echo "note: $GALLEY_MACHINE does not exist yet (it arrives with the pull below)"

say "1. pull $REPO"
cd "$REPO" && git pull --ff-only && git log --oneline -3 || { echo "pull failed: resolve it by hand (never merge on this machine)"; exit 1; }
[ -f "$GALLEY_MACHINE" ] || { echo "still no $GALLEY_MACHINE"; exit 1; }

say "2. SousVide clone (report only)"
PIN=$(tr -d '\r\n' < "$REPO/figs/sousvide_overlay/SOUSVIDE_REF")
git -C "$SV" log --oneline -3
echo "  pinned upstream: ${PIN:0:7}"
if git -C "$SV" merge-base --is-ancestor "$PIN" HEAD 2>/dev/null; then
    echo "  HEAD contains the pin; commits on top of it:"; git -C "$SV" log --oneline "$PIN"..HEAD | sed 's/^/    /'
    echo "  files changed since the pin:"; git -C "$SV" diff --stat "$PIN" HEAD | tail -15 | sed 's/^/    /'
    if git -C "$SV" diff --name-only "$PIN" HEAD | grep -q '^src/'; then
        PROBLEMS+=("SousVide has local commits that change src/ (the sousvide package): results may differ from upstream")
    fi
else
    PROBLEMS+=("SousVide HEAD does not contain the pinned commit ${PIN:0:7}")
fi
echo "  uncommitted:"; git -C "$SV" status --short | head -20 | sed 's/^/    /'

say "3. install checks"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
ARCH=$(cat "$ROOT/.figs_install_state/tcnn_arch" 2>/dev/null || echo "?")
echo "  GPU compute capability $CC, tiny-cuda-nn built for $ARCH"
[ "$ARCH" = "$CC" ] || PROBLEMS+=("tiny-cuda-nn built for $ARCH but the GPU is $CC: setup_scripts/install_figs.sh --prefix $ROOT --redo tcnn")
( source "$ROOT/figs_env.sh" >/dev/null 2>&1
  printf '  colmap in kitchen: %s\n' "$(command -v colmap || echo MISSING)"
  printf '  ns-train: %s\n' "$(command -v ns-train || echo MISSING)" )
bash "$REPO/setup_scripts/verify_figs.sh" --prefix "$ROOT" --quick 2>&1 | tail -30   # step 6 flies

say "4. overlay (our captures and courses into SousVide/configs; existing files are never overwritten)"
# apply_overlay.sh assumes the lab layout (repo root with the install at <repo>/figs); here the
# repo and the install are siblings, so copy directly with the same no-overwrite rule.
( cd "$REPO/figs/sousvide_overlay" && find configs -type f | sort ) | while read -r f; do
    src="$REPO/figs/sousvide_overlay/$f"; dst="$SV/$f"
    if [ ! -e "$dst" ]; then mkdir -p "$(dirname "$dst")"; cp "$src" "$dst"; echo "  added     $f"
    elif cmp -s <(tr -d '\r' < "$src") <(tr -d '\r' < "$dst"); then echo "  same      $f"
    else echo "  DIFFERENT $f (left alone; compare by hand)"; fi
done

say "5. Galley backend"
cd "$REPO/ui/backend"
UV=$(command -v uv || echo ~/.local/bin/uv)
[ -d .venv ] || $UV venv -q .venv
$UV pip install -q -p .venv -e '.[test]'
GALLEY_TEST_CONFIGS=$SV/configs .venv/bin/python -m pytest -q 2>&1 | tail -3

say "6. fly backroom + circuit (course → simulate → validate → record)"
( source "$ROOT/figs_env.sh" >/dev/null 2>&1
  python "$REPO/figs/figs_pipeline.py" --project-root "$ROOT" --scene backroom --course circuit \
      --from course --redo course --redo simulate --redo validate 2>&1 \
    | grep -E "✔|✗|!|tracking error|pixel|frames|VRAM|run record|failed|Error" | head -40 )
[ "${PIPESTATUS[0]}" = 0 ] || true

say "7. Galley server smoke"
.venv/bin/python -m galley > /tmp/galley_bringup.log 2>&1 &
SRV=$!
for i in $(seq 30); do curl -sf http://127.0.0.1:8800/api/health >/dev/null && break; sleep 0.5; done
curl -s http://127.0.0.1:8800/api/health; echo
curl -s http://127.0.0.1:8800/api/scenes | python3 -c "import json,sys; print('  scenes:', [(s['scene'], 'ok' if s['loadable'] else 'NOT loadable') for s in json.load(sys.stdin)])"
curl -s http://127.0.0.1:8800/api/machine | python3 -c "import json,sys; d=json.load(sys.stdin); print('  gpu:', d['gpu']['name'], d['gpu']['live'], ' disk:', d['disk'])"
kill $SRV; wait $SRV 2>/dev/null

say "summary"
if [ ${#PROBLEMS[@]} -eq 0 ]; then echo "  no blocking problems found"; else printf '  ! %s\n' "${PROBLEMS[@]}"; fi
echo "  Galley:  tmux new -d -s galley 'cd $REPO/ui/backend && GALLEY_MACHINE=$GALLEY_MACHINE .venv/bin/python -m galley'   (loopback :8800)"
echo "  from the laptop:  ssh -N -L 18800:localhost:8800 $(whoami)@$(hostname)  →  http://localhost:18800"
echo "  (18800 on the laptop side: Windows often reserves 8800, which fails with 'bind … Permission denied')"
