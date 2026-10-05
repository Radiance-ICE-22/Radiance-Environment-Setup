#!/usr/bin/env bash
# =============================================================================
#  install_semantics.sh — OpenCLIP + DINOv2 + radiance_semantics into 'kitchen'
# =============================================================================
#  Phase 0 of the semantic-embedding plan (docs/SEMANTICS_PLAN.md). Adds, to the
#  existing kitchen env and WITHOUT moving anything already in it:
#     open_clip_torch 2.24.0 (--no-deps) + ftfy, regex, huggingface_hub<1.0, tqdm
#     semantics/ from this repo as the editable package radiance_semantics
#     OpenCLIP ViT-B-16/laion2b_s34b_b88k and DINOv2 vits14 weights, cached
#     under the prefix (HF_HOME / TORCH_HOME, as figs_env.sh sets them)
#
#  Why so careful: tiny-cuda-nn and gsplat are compiled against torch 2.1.2 and
#  numpy 1.x. Every pip call here runs under a constraints file that freezes
#  every package already installed, so a dependency can be ADDED but never
#  upgraded. Versions of the ABI-sensitive stack are compared before and after.
#  OpenCLIP is pinned at 2.24.0 because 3.x requires timm>=1.0.17 and
#  nerfstudio 1.1.4 pins timm==0.6.7 (2.24 imports timm only optionally).
#
#  Idempotent; safe to re-run. Run it while Galley's queue is idle.
#
#  Usage:
#     ./install_semantics.sh --prefix ~/Radiance/figs
#     ./install_semantics.sh --prefix ~/Radiance/figs --skip-weights   # no downloads
#  Also run by install_figs.sh as its `semantics` step.
# =============================================================================

set -uo pipefail

PROJECT_ROOT="${FIGS_PREFIX:-$HOME/projects/figs_validation}"
OPEN_CLIP_PIN="open_clip_torch==2.24.0"
SKIP_WEIGHTS=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/semantics"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)       PROJECT_ROOT="$2"; shift 2 ;;
    --skip-weights) SKIP_WEIGHTS=1; shift ;;
    --yes|-y)       shift ;;                      # accepted for symmetry with install_figs.sh
    -h|--help)      sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)"; exit 2 ;;
  esac
done
PROJECT_ROOT="${PROJECT_ROOT%/}"
ENV_FILE="$PROJECT_ROOT/figs_env.sh"
STATE_DIR="$PROJECT_ROOT/.figs_install_state"

say()  { printf '%s\n' "$*"; }
ok()   { printf '  ✔ %s\n' "$*"; }
die()  { printf '  ✘ %s\n' "$*"; exit 1; }

[[ -f "$ENV_FILE" ]] || die "$ENV_FILE not found — pass --prefix <FiGS install prefix> (run install_figs.sh first)"
[[ -f "$PKG_DIR/pyproject.toml" ]] || die "$PKG_DIR/pyproject.toml missing — run this from a full checkout of the repo"
mkdir -p "$STATE_DIR"

# conda's activate.d hooks are not 'set -u' safe; figs_env.sh also cd's into SousVide.
set +u
# shellcheck disable=SC1090
source "$ENV_FILE" >/dev/null 2>&1
set -u
[[ "${CONDA_DEFAULT_ENV:-}" == "kitchen" ]] || die "env 'kitchen' did not activate from $ENV_FILE"
say "python: $(command -v python)"

BEFORE="$STATE_DIR/semantics_versions_before.json"
CONS="$STATE_DIR/semantics-constraints.txt"

# env_check.py is stdlib-only, so it runs from the source tree before anything is installed.
say "--- stack before (must already be the validated one) ---"
PYTHONPATH="$PKG_DIR" python -m radiance_semantics.env_check --snapshot "$BEFORE" --constraints "$CONS" \
  || die "the kitchen stack is already off its pins (above) — fix that before adding anything"

pipc() {  # pipc <what> <pip args...>
  local what="$1"; shift
  say "=== pip: $what ==="
  pip install --disable-pip-version-check -c "$CONS" "$@" || die "pip failed: $what (nothing that was installed before was changed)"
}

pipc "$OPEN_CLIP_PIN (no deps)"           --no-deps "$OPEN_CLIP_PIN"
# open_clip's pure-Python needs. Dependencies may be ADDED (e.g. wcwidth for ftfy) but the
# constraints file stops any installed package from moving.
# huggingface_hub < 1.0: 1.x replaced requests with httpx; open_clip 2.24 was written against 0.x.
pipc "ftfy regex huggingface_hub<1 tqdm" ftfy regex "huggingface_hub<1.0" tqdm

say "=== pip: radiance_semantics (editable, no deps) ==="
if ! pip install --disable-pip-version-check -c "$CONS" --no-deps -e "$PKG_DIR"; then
  # Old setuptools without PEP 660 editables: fall back to a .pth pointing at the source tree.
  SITE=$(python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
  echo "$PKG_DIR" > "$SITE/radiance_semantics.pth" || die "could not write $SITE/radiance_semantics.pth"
  ok "editable install unavailable; added $SITE/radiance_semantics.pth"
fi

say "--- stack after ---"
if ! python -m radiance_semantics.env_check --compare "$BEFORE"; then
  say ""
  say "  FATAL: a watched package changed (listed above). Put it back before using FiGS, e.g."
  say "    pip install --no-deps <package>==<version shown as 'before'>"
  say "  then re-run this script. Compiled extensions (tiny-cuda-nn, gsplat) break if torch moved."
  exit 1
fi

python -c 'import open_clip, radiance_semantics as r, importlib.metadata as m; print("import OK: open_clip", m.version("open_clip_torch"), "| radiance_semantics", r.__version__)' \
  || die "imports failed after install"

if (( SKIP_WEIGHTS )); then
  say "--- weights skipped (--skip-weights); fetch later with: python -m radiance_semantics.models --fetch ---"
else
  say "--- weights (cached under $PROJECT_ROOT/.cache) ---"
  python -m radiance_semantics.models --fetch | grep -v '^GALLEY_JSON' \
    || die "weight download failed (network?). Re-run, or use --skip-weights and fetch later"
fi

printf '%s (open_clip 2.24.0, radiance_semantics)\n' "$(date '+%Y-%m-%d %H:%M')" > "$STATE_DIR/semantics.done"
ok "semantic features installed into kitchen"
