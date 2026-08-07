#!/usr/bin/env bash
# apply_overlay.sh — restore our files into the SousVide clone after an install.
#
# SousVide is upstream's repo (StanfordMSL/SousVide), cloned fresh by
# `install_figs.sh` step `clone`. Our capture configs and courses have to live
# *inside* that clone for FiGS to find them, but they are ours, not upstream's —
# so they are version-controlled in this repo under
#
#     figs/sousvide_overlay/
#
# and copied into place by this script. Keeping them out of the clone is what lets
# the outer repo be a normal git repository instead of one with a foreign checkout
# embedded in it, and it means `install_figs.sh --redo clone` (which wipes and
# re-clones) costs you nothing.
#
#   ./apply_overlay.sh                  # restore
#   ./apply_overlay.sh --dry-run        # show what would be written
#   ./apply_overlay.sh --checkout-pin   # also move the clone to the pinned commit
#
# Safe to re-run: identical files are skipped, differing ones are reported and
# left alone unless --force.

set -euo pipefail

ROOT=""
DRY=0
FORCE=0
PIN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --root)         ROOT="${2:?--root needs a path}"; shift 2 ;;
        --dry-run)      DRY=1; shift ;;
        --force)        FORCE=1; shift ;;
        --checkout-pin) PIN=1; shift ;;
        -h|--help)      sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# Default to the directory above this script (setup_scripts/ lives at the repo root).
if [ -z "$ROOT" ]; then
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

OVERLAY="$ROOT/figs/sousvide_overlay"
REPO="$ROOT/figs/SousVide"

[ -d "$OVERLAY" ] || { echo "no overlay at $OVERLAY" >&2; exit 1; }
[ -d "$REPO" ] || {
    echo "no SousVide clone at $REPO" >&2
    echo "run setup_scripts/install_figs.sh --prefix $ROOT/figs first" >&2
    exit 1; }

echo "overlay: $OVERLAY"
echo "target:  $REPO"
[ "$DRY" -eq 1 ] && echo "mode:    DRY RUN"
echo

# ── pinned upstream commit ───────────────────────────────────────────────────────
# install_figs.sh clones upstream at whatever HEAD happens to be that day. The
# numbers in docs/ were measured against one specific commit, so record it and say
# so loudly when they diverge — a silent upstream change is exactly the kind of
# thing that makes a reproduced run disagree with the documented reference figures.

PINFILE="$OVERLAY/SOUSVIDE_REF"
if [ -f "$PINFILE" ]; then
    want="$(tr -d '[:space:]' < "$PINFILE")"
    have="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    if [ "$want" = "$have" ]; then
        echo "upstream commit: $have (matches pin)"
    else
        echo "upstream commit: $have"
        echo "pinned:          $want"
        if [ "$PIN" -eq 1 ]; then
            echo "checking out pinned commit"
            if [ "$DRY" -eq 0 ]; then
                git -C "$REPO" checkout --quiet "$want"
                git -C "$REPO" submodule update --recursive --init
            fi
        else
            echo "  ! clone is not at the pinned commit. The reference numbers in docs/"
            echo "    were measured against the pin. Re-run with --checkout-pin to match."
        fi
    fi
else
    echo "no SOUSVIDE_REF pin recorded"
fi
echo

# ── copy the overlay in ──────────────────────────────────────────────────────────

copied=0; same=0; conflict=0

while IFS= read -r -d '' src; do
    rel="${src#"$OVERLAY"/}"
    [ "$rel" = "SOUSVIDE_REF" ] && continue          # metadata, not a payload file
    dst="$REPO/$rel"

    if [ -e "$dst" ] && cmp -s "$src" "$dst"; then
        same=$((same + 1))
        continue
    fi

    if [ -e "$dst" ] && [ "$FORCE" -eq 0 ]; then
        # The copy in the clone differs. It may be a hand-edit made on the machine
        # and not yet folded back into the overlay; clobbering it would lose work.
        echo "  ! $rel differs from the overlay — left alone (use --force to overwrite)"
        conflict=$((conflict + 1))
        continue
    fi

    echo "  $rel"
    if [ "$DRY" -eq 0 ]; then
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
    fi
    copied=$((copied + 1))
done < <(find "$OVERLAY" -type f -print0)

echo
echo "restored $copied, already current $same, conflicting $conflict"

if [ "$conflict" -gt 0 ]; then
    echo
    echo "Conflicts mean the machine has a newer version than the repo. To keep it,"
    echo "copy it back into figs/sousvide_overlay/ and commit; to discard it, re-run"
    echo "with --force."
    exit 3
fi
