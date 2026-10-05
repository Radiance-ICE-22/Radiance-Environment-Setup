#!/usr/bin/env bash
# =============================================================================
#  verify_figs.sh — functional smoke test for an installed FiGS environment
# =============================================================================
#  install_figs.sh --verify-only checks that everything *imports*.
#  This script goes further: it exercises the GPU, patches the example notebook
#  to a scene that actually exists in the downloadable dataset, and runs a full
#  headless flight simulation, watching VRAM the whole time.
#
#  Usage:
#     ./verify_figs.sh                 # imports + GPU + notebook run
#     ./verify_figs.sh --quick         # imports + GPU only, no notebook
#     ./verify_figs.sh --scene backroom --course circuit
#     ./verify_figs.sh --quick --semantics   # + OpenCLIP/DINOv2 smoke and the gsplat
#                                            #   N-channel gradient probe (Phase 0)
# =============================================================================

set -uo pipefail

PROJECT_ROOT="${FIGS_PREFIX:-$HOME/projects/figs_validation}"
REPO_DIR="$PROJECT_ROOT/SousVide"
ENV_FILE="$PROJECT_ROOT/figs_env.sh"
SCENE="backroom"
COURSE="circuit"
QUICK=0
NB_TIMEOUT=1800
# By default, skip pg.generate_gsplat() — that runs COLMAP SfM + splatfacto
# training from scratch (tens of minutes, heavy VRAM). Verification only needs
# the *load an existing splat and fly it* path. Use --with-splat-gen to include.
SPLAT_GEN=0
SEMANTICS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)  QUICK=1; shift ;;
    --with-splat-gen) SPLAT_GEN=1; shift ;;
    --semantics) SEMANTICS=1; shift ;;
    --scene)  SCENE="$2"; shift 2 ;;
    --course) COURSE="$2"; shift 2 ;;
    --prefix) PROJECT_ROOT="$2"; REPO_DIR="$PROJECT_ROOT/SousVide"; ENV_FILE="$PROJECT_ROOT/figs_env.sh"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
  BOLD=$(tput bold); DIM=$(tput dim); RESET=$(tput sgr0)
  RED=$(tput setaf 1); GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); CYAN=$(tput setaf 6)
  IS_TTY=1
else
  BOLD=""; DIM=""; RESET=""; RED=""; GREEN=""; YELLOW=""; CYAN=""; IS_TTY=0
fi

PASS=0; WARN=0; FAILC=0
hr() { printf '%s%s%s\n' "$DIM" "$(printf '─%.0s' $(seq 1 72))" "$RESET"; }
head2() { hr; printf '%s%s  %s%s\n' "$BOLD" "$CYAN" "$1" "$RESET"; hr; }

t() {  # t <label> <command>
  local label="$1"; shift
  printf '  %-40s' "$label"
  local out
  if out=$( eval "$@" 2>&1 ); then
    printf '%s✔%s %s\n' "$GREEN" "$RESET" "$(echo "$out" | tail -1 | cut -c1-42)"; PASS=$((PASS+1)); return 0
  else
    printf '%s✘%s %s\n' "$RED" "$RESET" "$(echo "$out" | tail -1 | cut -c1-60)"; FAILC=$((FAILC+1)); return 1
  fi
}
tw() {
  local label="$1"; shift
  printf '  %-40s' "$label"
  local out
  if out=$( eval "$@" 2>&1 ); then
    printf '%s✔%s %s\n' "$GREEN" "$RESET" "$(echo "$out" | tail -1 | cut -c1-42)"; PASS=$((PASS+1)); return 0
  else
    printf '%s!%s %s\n' "$YELLOW" "$RESET" "$(echo "$out" | tail -1 | cut -c1-60)"; WARN=$((WARN+1)); return 1
  fi
}

# ---------------------------------------------------------------- environment
head2 "Environment"
if [[ ! -f "$ENV_FILE" ]]; then
  printf '  %s✘%s %s not found — run install_figs.sh first\n' "$RED" "$RESET" "$ENV_FILE"; exit 1
fi
# conda's activate.d scripts read unset vars (e.g. MKL_INTERFACE_LAYER), which
# is fatal under 'set -u' — lift nounset across activation, then restore it.
set +u
# shellcheck disable=SC1090
source "$ENV_FILE" >/dev/null 2>&1
set -u
printf '  conda env : %s\n' "${CONDA_DEFAULT_ENV:-<none>}"
printf '  python    : %s\n' "$(command -v python)"
printf '  repo      : %s\n' "$REPO_DIR"
[[ "${CONDA_DEFAULT_ENV:-}" == "kitchen" ]] || { printf '  %s✘%s env "kitchen" not active\n' "$RED" "$RESET"; exit 1; }

# ------------------------------------------------------------------- imports
head2 "Imports"
t  "torch"                 "python -c 'import torch;print(torch.__version__)'"
t  "torch CUDA runtime"    "python -c 'import torch;print(\"cuda\",torch.version.cuda)'"
t  "tinycudann"            "python -c 'import tinycudann;print(\"ok\")'"
t  "nerfstudio"            "python -c 'import nerfstudio,importlib.metadata as m;print(m.version(\"nerfstudio\"))'"
t  "gsplat.rasterization"  "python -c 'from gsplat import rasterization;print(\"ok\")'"
t  "hloc"                  "python -c 'import hloc;print(\"ok\")'"
t  "figs"                  "python -c 'import figs;print(\"ok\")'"
t  "acados_template"       "python -c 'import acados_template;print(\"ok\")'"
tw "sousvide"              "python -c 'import sousvide;print(\"ok\")'"

# ------------------------------------------------------------------ binaries
head2 "Binaries & native libs"
# 'colmap -h' exits non-zero after printing its banner — with pipefail that
# would fail the pipeline, so discard its status and judge on the output.
t  "colmap"                "{ colmap -h 2>&1 || true; } | grep -m1 -F 'COLMAP'"
tw "colmap CUDA support"   "{ colmap -h 2>&1 || true; } | grep -qi cuda && echo 'CUDA-enabled'"
t  "ns-train / splatfacto" "{ ns-train --help 2>&1 || true; } | grep -q splatfacto && echo registered"
t  "libacados.so"          "test -f \"\$ACADOS_SOURCE_DIR/lib/libacados.so\" && echo present"
t  "ffmpeg"                "ffmpeg -version 2>&1 | head -1"

# ----------------------------------------------------------------------- GPU
head2 "GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf '  %s✘%s nvidia-smi missing\n' "$RED" "$RESET"; FAILC=$((FAILC+1))
else
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total \
             --format=csv,noheader | sed 's/^/  /'
  t "torch sees the GPU"   "python -c 'import torch;assert torch.cuda.is_available();print(torch.cuda.get_device_name(0))'"
  t "live CUDA matmul"     "python -c 'import torch;a=torch.randn(1024,1024,device=\"cuda\");print(float((a@a).mean()))'"
  t "tinycudann fwd pass" \
    "python -c 'import torch,tinycudann as tcnn;m=tcnn.Encoding(3,{\"otype\":\"HashGrid\",\"n_levels\":8,\"n_features_per_level\":2,\"log2_hashmap_size\":15,\"base_resolution\":16,\"per_level_scale\":1.5}).cuda();print(m(torch.rand(64,3,device=\"cuda\")).shape)'"
  # soft: gsplat's rasterization() signature has shifted between minor versions,
  # so a failure here may be API drift rather than a broken install
  tw "gsplat rasterize call" \
    "python - <<'PY'
import torch
from gsplat import rasterization
N=100
means=torch.randn(N,3,device='cuda')
quats=torch.nn.functional.normalize(torch.randn(N,4,device='cuda'),dim=-1)
scales=torch.rand(N,3,device='cuda')*0.1
opac=torch.rand(N,device='cuda')
colors=torch.rand(N,3,device='cuda')
K=torch.tensor([[[100.,0,64],[0,100.,64],[0,0,1.]]],device='cuda')
vm=torch.eye(4,device='cuda')[None]
img,_,_=rasterization(means,quats,scales,opac,colors,vm,K,128,128,sh_degree=None)
print('rendered',tuple(img.shape))
PY"
  MAXVRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  if (( MAXVRAM < 6000 )); then
    printf '  %s!%s %s MiB VRAM — inference/simulation fine, splatfacto *training* will likely OOM\n' \
      "$YELLOW" "$RESET" "$MAXVRAM"; WARN=$((WARN+1))
  fi
fi

# ------------------------------------------------------ semantic features (Phase 0)
if (( SEMANTICS )); then
  head2 "Semantic features (docs/SEMANTICS.md, Phase 0)"
  if t "radiance_semantics" "python -c 'import radiance_semantics as r;print(r.__version__)'"; then
    t "open_clip 2.24.0"     "python -c 'import importlib.metadata as m;v=m.version(\"open_clip_torch\");assert v==\"2.24.0\",v;print(v)'"
    # torch 2.1.2 / nerfstudio 1.1.4 / gsplat 1.0.0 / numpy 1.x — the compiled extensions' ABI
    t "pinned stack intact"  "python -m radiance_semantics.env_check >/dev/null && echo 'torch, nerfstudio, gsplat, numpy on their pins'"
    # functional, not just imports: CLIP must tell red from blue; DINOv2 gives 384-d tokens
    t "CLIP + DINOv2 smoke"  "python -m radiance_semantics.models --smoke | grep -v '^GALLEY_JSON'"
    # gsplat renders 64-channel features and its gradients are exact (linearity, finite
    # differences, blend weights, channel chunking) on 20k synthetic Gaussians
    t "gsplat N-ch gradients" "python -m radiance_semantics.probe --quick | grep -E '✔|✗'"
  else
    printf '      %sinstall it: %s/setup_scripts/install_semantics.sh --prefix %s%s\n' \
      "$DIM" "$(cd "$(dirname "$0")/.." && pwd)" "$PROJECT_ROOT" "$RESET"
  fi
fi

# -------------------------------------------------------------- example data
head2 "Example data"
tw "gsplats/capture"   "test -d '$REPO_DIR/gsplats/capture' && ls '$REPO_DIR/gsplats/capture'"
tw "gsplats/workspace" "test -d '$REPO_DIR/gsplats/workspace' && ls '$REPO_DIR/gsplats/workspace'"
# A checkpoint must exist for the SELECTED scene — finding some other scene's
# is not enough, and silently proceeding is what makes the notebook fall through
# to running SfM/training from scratch.
ALL_CKPTS=$(find "$REPO_DIR/gsplats/workspace" -name 'step-*.ckpt' 2>/dev/null)
CKPT=$(printf '%s\n' "$ALL_CKPTS" | grep -m1 -- "/$SCENE/" || true)
if [[ -n "$CKPT" ]]; then
  printf '  %s✔%s checkpoint for "%s": %s\n' "$GREEN" "$RESET" "$SCENE" "${CKPT#"$REPO_DIR"/}"; PASS=$((PASS+1))
else
  printf '  %s!%s no trained checkpoint for scene "%s"\n' "$YELLOW" "$RESET" "$SCENE"; WARN=$((WARN+1))
  if [[ -n "$ALL_CKPTS" ]]; then
    printf '      scenes that DO have one:\n'
    printf '%s\n' "$ALL_CKPTS" | sed -E 's#.*/outputs/([^/]+)/.*#        \1#' | sort -u
    printf '      %sre-run with --scene <name> to use one of those%s\n' "$DIM" "$RESET"
  fi
  if (( SPLAT_GEN == 0 )); then
    printf '  %s!%s skipping notebook (would need to train from scratch; pass --with-splat-gen to allow)\n' \
      "$YELLOW" "$RESET"
    QUICK=1
  fi
fi

# ---------------------------------------------------- headless notebook run
if (( QUICK == 0 )); then
  head2 "Headless flight simulation (notebooks/figs_examples.ipynb)"
  NB="$REPO_DIR/notebooks/figs_examples.ipynb"
  if [[ ! -f "$NB" ]]; then
    printf '  %s!%s %s not found — skipping\n' "$YELLOW" "$RESET" "$NB"; WARN=$((WARN+1))
  else
    PATCHED="$REPO_DIR/notebooks/_verify_figs_examples.ipynb"
    printf '  patching scene -> %s / %s (splat generation: %s)\n' \
      "$SCENE" "$COURSE" "$( ((SPLAT_GEN)) && echo ENABLED || echo skipped )"
    python - "$NB" "$PATCHED" "$SCENE" "$COURSE" "$SPLAT_GEN" <<'PY'
import json,re,sys
src,dst,scene,course,splat_gen=sys.argv[1:6]
splat_gen=int(splat_gen)
nb=json.load(open(src))
n=0; g=0
for c in nb["cells"]:
    if c["cell_type"]!="code": continue
    out=[]; seen=set(); depth=0
    for line in c["source"]:
        # --- comment out splat generation (COLMAP SfM + splatfacto training) ---
        # Without a checkpoint this is a 30-60 min GPU-heavy job, and it is the
        # step that drags in the hloc/pycolmap API surface. Verification only
        # needs the load-and-fly path.
        if not splat_gen:
            if depth == 0 and re.match(r'\s*[\w\.]*\bgenerate_gsplat\s*\(', line):
                out.append("# [verify_figs] skipped: " + line)
                # track bracket depth so a call split across lines stays commented
                depth = line.count("(") - line.count(")")
                g += 1
                continue
            if depth > 0:
                out.append("# [verify_figs] skipped: " + line)
                depth += line.count("(") - line.count(")")
                g += 1
                continue
        kind=None
        if re.match(r'\s*#?\s*capture_name\s*=', line):
            kind="capture"; line=f'capture_name = "{scene}"\n'
        elif re.match(r'\s*#?\s*scene_name\s*,\s*course_name\s*=', line):
            kind="scene"; line=f'scene_name, course_name = "{scene}", "{course}"\n'
        if kind:
            if kind in seen: continue      # drop the commented-out duplicates
            seen.add(kind); n+=1
        out.append(line)
    c["source"]=out
json.dump(nb,open(dst,"w"))
print(f"patched {n} scene line(s); commented out {g} generate_gsplat line(s)")
PY

    LOG="$PROJECT_ROOT/.figs_install_logs/notebook_verify.log"
    mkdir -p "$(dirname "$LOG")"
    printf '  running headless (timeout %ss) — log: %s\n' "$NB_TIMEOUT" "$LOG"

    ( cd "$REPO_DIR" && jupyter nbconvert --to notebook --execute \
        "$PATCHED" --output _executed_figs_examples.ipynb \
        --ExecutePreprocessor.timeout=$NB_TIMEOUT ) >"$LOG" 2>&1 &
    NBPID=$!

    # live progress: elapsed, peak VRAM, last log line
    t0=$(date +%s); peak=0; frames=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏); i=0; lastsize=0; lastchange=$t0
    while kill -0 $NBPID 2>/dev/null; do
      now=$(date +%s); el=$((now-t0))
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      [[ -n "$used" ]] && (( used > peak )) && peak=$used
      size=$(stat -c%s "$LOG" 2>/dev/null || echo 0)
      [[ "$size" != "$lastsize" ]] && { lastsize=$size; lastchange=$now; }
      idle=$((now-lastchange))
      if (( IS_TTY )); then
        if (( idle > 120 )); then
          printf '\r\033[K  %s%s%s %ds elapsed | VRAM %s MiB (peak %s) %s| no output %ds — rendering, still alive%s' \
            "$YELLOW" "${frames[i]}" "$RESET" "$el" "${used:-?}" "$peak" "$YELLOW" "$idle" "$RESET"
        else
          printf '\r\033[K  %s%s%s %ds elapsed | VRAM %s MiB (peak %s)' \
            "$CYAN" "${frames[i]}" "$RESET" "$el" "${used:-?}" "$peak"
        fi
      elif (( el % 30 == 0 )); then
        printf '  … %ds elapsed, VRAM %s MiB (peak %s)\n' "$el" "${used:-?}" "$peak"
      fi
      i=$(( (i+1) % 10 )); sleep 1
    done
    wait $NBPID; NBRC=$?
    (( IS_TTY )) && printf '\r\033[K'
    el=$(( $(date +%s) - t0 ))

    if (( NBRC == 0 )); then
      printf '  %s✔%s notebook executed in %ds — peak VRAM %s MiB\n' "$GREEN" "$RESET" "$el" "$peak"; PASS=$((PASS+1))
      VID=$(find "$REPO_DIR/notebooks" -maxdepth 1 -name '*.mp4' -newermt "-${el} seconds" 2>/dev/null | head -1)
      if [[ -n "$VID" ]]; then
        printf '  %s✔%s render produced: %s (%s)\n' "$GREEN" "$RESET" "$(basename "$VID")" "$(du -h "$VID" | cut -f1)"; PASS=$((PASS+1))
      else
        printf '  %s!%s no .mp4 produced — check %s\n' "$YELLOW" "$RESET" "$LOG"; WARN=$((WARN+1))
      fi
      idlev=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      printf '  %s·%s VRAM after completion: %s MiB (should be near 0)\n' "$DIM" "$RESET" "$idlev"
    else
      printf '  %s✘%s notebook FAILED (exit %d after %ds)\n' "$RED" "$RESET" "$NBRC" "$el"; FAILC=$((FAILC+1))
      printf '  %slast 20 lines:%s\n' "$DIM" "$RESET"
      tail -20 "$LOG" | sed 's/^/    /'
    fi
    rm -f "$PATCHED"
  fi
fi

# ------------------------------------------------------------------- summary
hr
if (( FAILC == 0 )); then
  printf '%s%s  ✔ FiGS environment verified%s — %d passed, %d warnings\n' "$BOLD" "$GREEN" "$RESET" "$PASS" "$WARN"
else
  printf '%s%s  ✘ %d check(s) FAILED%s — %d passed, %d warnings\n' "$BOLD" "$RED" "$FAILC" "$RESET" "$PASS" "$WARN"
fi
hr
exit $(( FAILC > 0 ? 1 : 0 ))
