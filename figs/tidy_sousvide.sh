#!/usr/bin/env bash
# tidy_sousvide.sh — move generated flight renders out of the SousVide repo root.
#
# SousVide is an upstream repo. Rendered MP4s, run records and hand-written driver
# scripts accumulate in its root, where they are indistinguishable from repo content
# and show up as untracked noise in every `git status`.
#
# This script relocates the *renders* only:
#
#     SousVide/<scene>_flight.mp4   →   SousVide/outputs/flights/<scene>_flight.mp4
#
# It moves nothing that git tracks, deletes nothing, and defaults to a dry run.
#
#   ./tidy_sousvide.sh                 # show what would happen
#   ./tidy_sousvide.sh --apply         # do it
#   ./tidy_sousvide.sh --apply --repo ~/somewhere/SousVide
#
# Re-runnable: a second --apply run is a no-op.

set -euo pipefail

# ── locate the repo ──────────────────────────────────────────────────────────────
# Same resolution order as figs_pipeline.py, so the two agree about where things are:
# --repo, then $FIGS_PROJECT_ROOT, then $ACADOS_SOURCE_DIR (figs_env.sh exports it as
# <root>/SousVide/FiGS/acados), then the conventional path.

APPLY=0
REPO=""

while [ $# -gt 0 ]; do
    case "$1" in
        --apply)  APPLY=1; shift ;;
        --repo)   REPO="${2:?--repo needs a path}"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$REPO" ]; then
    if [ -n "${FIGS_PROJECT_ROOT:-}" ]; then
        REPO="$FIGS_PROJECT_ROOT/SousVide"
    elif [ -n "${ACADOS_SOURCE_DIR:-}" ]; then
        REPO="$(cd "$ACADOS_SOURCE_DIR/../.." && pwd)"
    else
        REPO="$HOME/FYP-Radiance/figs/SousVide"
    fi
fi

RESOLVED="$(cd "$REPO" 2>/dev/null && pwd)" || { echo "no such directory: $REPO" >&2; exit 1; }
REPO="$RESOLVED"

# A real SousVide checkout has both of these. Refuse to shuffle files in a directory
# that merely happens to exist — a mistyped --repo should not rearrange someone's home.
[ -d "$REPO/gsplats" ] && [ -d "$REPO/configs" ] || {
    echo "not a SousVide repo (no gsplats/ + configs/): $REPO" >&2; exit 1; }

echo "repo:  $REPO"
[ "$APPLY" -eq 1 ] || echo "mode:  DRY RUN  (re-run with --apply to make changes)"
echo

DEST="$REPO/outputs/flights"

run() {                       # echo in dry-run, execute under --apply
    if [ "$APPLY" -eq 1 ]; then "$@"; else printf '  would: %s\n' "$*"; fi
}

# ── is a path tracked by git? ────────────────────────────────────────────────────
# Anything upstream committed stays exactly where upstream put it.

tracked() {
    git -C "$REPO" ls-files --error-unmatch -- "$1" >/dev/null 2>&1
}

# ── 1. destination ───────────────────────────────────────────────────────────────

if [ -d "$DEST" ]; then
    echo "outputs/flights/ already exists"
else
    echo "creating outputs/flights/"
    run mkdir -p "$DEST"
fi
echo

# ── 2. move the renders ──────────────────────────────────────────────────────────
# Root-level MP4s only (-maxdepth 1). Nothing under gsplats/ is touched: the staged
# captures there are inputs FiGS globs for by substring, and moving one breaks the
# scene lookup.

echo "flight renders in the repo root:"
moved=0
skipped=0

while IFS= read -r -d '' f; do
    base="$(basename "$f")"

    if tracked "$base"; then
        echo "  $base — tracked by git, leaving alone"
        skipped=$((skipped + 1))
        continue
    fi

    if [ -e "$DEST/$base" ]; then
        # Same name in both places. Never clobber: the one in outputs/ may be the
        # render you actually care about.
        if cmp -s "$f" "$DEST/$base"; then
            echo "  $base — identical copy already in outputs/flights/, removing the root one"
            run rm -f "$f"
        else
            ts="$(date -r "$f" +%Y-%m-%d_%H%M 2>/dev/null || date +%Y-%m-%d_%H%M)"
            echo "  $base — differs from the one in outputs/flights/, filing as ${base%.mp4}_$ts.mp4"
            run mv -n "$f" "$DEST/${base%.mp4}_$ts.mp4"
        fi
    else
        echo "  $base → outputs/flights/"
        run mv -n "$f" "$DEST/"
    fi
    moved=$((moved + 1))
done < <(find "$REPO" -maxdepth 1 -type f -name '*.mp4' -print0)

[ "$moved" -eq 0 ] && echo "  (none — root is already clean)"
echo

# ── 3. keep them out of git ──────────────────────────────────────────────────────
# Written to .git/info/exclude, not .gitignore: this is our local layout preference,
# and .gitignore is a tracked upstream file that would show as a modification and
# conflict on the next pull.

EXCLUDE="$REPO/.git/info/exclude"
if [ -d "$REPO/.git" ]; then
    if grep -qxF 'outputs/' "$EXCLUDE" 2>/dev/null; then
        echo "git exclude: outputs/ already listed"
    else
        # Deliberately not `*.mp4`: that would also hide a stray render dropped in the
        # root by something other than the pipeline, which is exactly what the report at
        # the end of this script is for.
        echo "git exclude: adding outputs/ and runs/ to .git/info/exclude"
        if [ "$APPLY" -eq 1 ]; then
            mkdir -p "$(dirname "$EXCLUDE")"
            {
                echo ''
                echo '# figs pipeline artefacts (added by tidy_sousvide.sh)'
                echo 'outputs/'
                echo 'runs/'
            } >> "$EXCLUDE"
        else
            printf '  would: append outputs/ and runs/ to %s\n' "$EXCLUDE"
        fi
    fi
else
    echo "git exclude: $REPO is not a git checkout, skipping"
fi
echo

# ── 4. what is left ──────────────────────────────────────────────────────────────
# Not moved automatically — you have to decide. Driver scripts like
# script_intellisense.py and simulate_and_validate.py are yours, not upstream's, but
# they may be referenced by shell history or notes, so this only reports them.

echo "untracked files still in the repo root:"
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" ls-files --others --exclude-standard --directory \
        | grep -v '/' | sed 's/^/  /' || true
else
    echo "  (not a git checkout — cannot tell yours from upstream's)"
fi
echo

echo "contents of outputs/flights/:"
if [ -d "$DEST" ]; then
    ls -lh "$DEST" 2>/dev/null | tail -n +2 | sed 's/^/  /' || true
else
    echo "  (would exist after --apply)"
fi
echo

if [ "$APPLY" -eq 1 ]; then
    echo "done. Future runs of figs_pipeline.py write straight to outputs/flights/"
    echo "once you have copied over the patched copy."
else
    echo "dry run only — nothing changed. Re-run with --apply."
fi
