#!/usr/bin/env bash
# =============================================================================
#  install_figs.sh — FiGS / SOUS-VIDE installer for Ubuntu 22.04 / 24.04
# =============================================================================
#  Installs, in order:
#     Miniconda -> SousVide repo (+FiGS, acados, hloc submodules) -> acados
#     -> conda env 'kitchen' -> tiny-cuda-nn (with the 3 known build fixes)
#     -> nerfstudio + editable installs -> example GSplats -> verification
#
#  Everything is idempotent and resumable: completed steps are recorded in
#  .figs_install_state/ and skipped on re-run. Delete a marker to force a redo.
#
#  Usage:
#     ./install_figs.sh                     # full install
#     ./install_figs.sh --skip-gsplats      # env only, no 5GB example data
#     ./install_figs.sh --redo acados       # force one step to re-run
#     ./install_figs.sh --list-steps        # show step IDs
#     ./install_figs.sh --verify-only       # just run the checks
#     ./install_figs.sh --yes               # no interactive prompts
#     ./install_figs.sh --prefix ~/somewhere
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------- configuration
PROJECT_ROOT="${FIGS_PREFIX:-$HOME/projects/figs_validation}"
REPO_URL="https://github.com/StanfordMSL/SousVide.git"
REPO_DIR_NAME="SousVide"
CONDA_ENV_NAME="kitchen"
CONDA_DIR=""                 # default: <prefix>/miniconda3  (override with --conda-dir)
GSPLAT_GDRIVE_ID="1kW5dzsfD3rbRA3RIQDyJPG6_UJaO9ALP"
STALL_WARN_SECS=180          # warn if a step produces no output for this long
MAKE_JOBS="$(nproc 2>/dev/null || echo 4)"

SKIP_GSPLATS=0
VERIFY_ONLY=0
ASSUME_YES=0
NEW_HOST=0
RELOCATED=0
REDO_STEPS=""

# ------------------------------------------------------------------ arg parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-gsplats) SKIP_GSPLATS=1; shift ;;
    --verify-only)  VERIFY_ONLY=1; shift ;;
    --yes|-y)       ASSUME_YES=1; shift ;;
    --prefix)       PROJECT_ROOT="$2"; shift 2 ;;
    --conda-dir)    CONDA_DIR="$2"; shift 2 ;;
    --use-home-conda) CONDA_DIR="$HOME/miniconda3"; shift ;;
    --redo)         REDO_STEPS="$REDO_STEPS $2"; shift 2 ;;
    --new-host)
      # Moving an existing install (e.g. on a portable SSD) to a different
      # machine. Everything on the drive is reused; only the host-specific
      # pieces are redone: apt packages live on the host, and tiny-cuda-nn is
      # compiled for one GPU architecture. envfile is rewritten in case the
      # mount path changed.
      NEW_HOST=1; SKIP_GSPLATS=1
      REDO_STEPS="$REDO_STEPS apt_deps hostgcc tcnn envfile"
      shift ;;
    --relocated)
      # The install tree was COPIED or MOVED to a different path (e.g. off the
      # portable SSD onto an internal disk). Conda is not relocatable, so conda
      # itself and the env must be rebuilt — but the ~8 GB package cache, the
      # ~5 GB gsplats and the git clone are all path-independent and reused, so
      # the rebuild links from local cache instead of re-downloading.
      RELOCATED=1; NEW_HOST=1; SKIP_GSPLATS=1
      REDO_STEPS="$REDO_STEPS apt_deps miniconda acados hostgcc conda_env tcnn pips envfile"
      shift ;;
    --list-steps)
      cat <<'EOF'
Step IDs (use with --redo):
  preflight     system checks: OS, GPU, disk, RAM
  apt_deps      apt build tools (git cmake build-essential wget unzip ...)
  miniconda     install Miniconda3 into <prefix>/miniconda3
  clone         clone SousVide + init submodules
  acados        cmake build of the acados MPC solver
  hostgcc       ensure a CUDA-11.8-compatible host compiler (gcc-11)
  conda_env     create conda env 'kitchen' from environment_x86.yml
  tcnn          build tiny-cuda-nn (the fragile one)
  pips          nerfstudio + editable FiGS / acados_template / hloc
  gsplats       download + unpack the example GSplat dataset
  envfile       write figs_env.sh with the persistent exports
  verify        full verification suite
EOF
      exit 0 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)"; exit 2 ;;
  esac
done

PROJECT_ROOT="${PROJECT_ROOT%/}"
REPO_DIR="$PROJECT_ROOT/$REPO_DIR_NAME"
LOG_DIR="$PROJECT_ROOT/.figs_install_logs"
STATE_DIR="$PROJECT_ROOT/.figs_install_state"
# (per-step logs live in $LOG_DIR/<step>.log)

# conda lives under the prefix by default, so nothing large lands in $HOME
[[ -z "$CONDA_DIR" ]] && CONDA_DIR="$PROJECT_ROOT/miniconda3"

# ---------------------------------------------------------------------------
#  Keep every large cache/scratch area on the prefix volume too. Left at their
#  defaults these quietly eat 10-20 GB of $HOME and /tmp:
#     pip wheel cache      ~/.cache/pip           (several GB for torch wheels)
#     conda package cache  <conda>/pkgs           (~8 GB)
#     build scratch        /tmp                   (tiny-cuda-nn objects; /tmp is
#                                                  often a small tmpfs in RAM)
#     torch / HF / nerfstudio model caches
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#  Ignore ~/.local/lib/pythonX.Y/site-packages entirely.
#  Python puts user-site AHEAD of the env's site-packages on sys.path, so a
#  stray torch in ~/.local silently shadows the env's torch — builds then link
#  against the wrong libtorch and fail with CUDA/ABI mismatches whose tracebacks
#  point at /home/<user>/.local paths. pip also falls back to installing there
#  when the env directory is not writable, which is how it gets populated by
#  accident (e.g. a UID mismatch on a portable drive).
# ---------------------------------------------------------------------------
export PYTHONNOUSERSITE=1
export PIP_USER=0

export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export CONDA_PKGS_DIRS="$CONDA_DIR/pkgs"
export TMPDIR="$PROJECT_ROOT/.tmp"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$TORCH_HOME" "$HF_HOME" 2>/dev/null || true

# ------------------------------------------------------------------- UI helpers
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && [[ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]]; then
  BOLD=$(tput bold); DIM=$(tput dim); RESET=$(tput sgr0)
  RED=$(tput setaf 1); GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3)
  BLUE=$(tput setaf 4); CYAN=$(tput setaf 6)
  IS_TTY=1
else
  BOLD=""; DIM=""; RESET=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; CYAN=""
  IS_TTY=0
fi

TOTAL_STEPS=12
CHECK_COUNT_EXPECTED=19
STEP_NO=0
declare -a WARNINGS=()
declare -a FAILURES=()
SCRIPT_START=$(date +%s)

hr()    { printf '%s\n' "${DIM}$(printf '─%.0s' $(seq 1 "$(tput cols 2>/dev/null || echo 72)"))${RESET}"; }
say()   { printf '%s\n' "$*"; }
info()  { printf '  %s•%s %s\n' "$BLUE" "$RESET" "$*"; }
ok()    { printf '  %s✔%s %s\n' "$GREEN" "$RESET" "$*"; }
warn()  { printf '  %s!%s %s\n' "$YELLOW" "$RESET" "$*"; WARNINGS+=("$*"); }
fail()  { printf '  %s✘%s %s\n' "$RED" "$RESET" "$*"; FAILURES+=("$*"); }

banner() {
  hr
  printf '%s%s  FiGS / SOUS-VIDE installer%s\n' "$BOLD" "$CYAN" "$RESET"
  printf '  %starget:%s %s\n' "$DIM" "$RESET" "$PROJECT_ROOT"
  printf '  %sconda: %s %s\n' "$DIM" "$RESET" "$CONDA_DIR"
  printf '  %slogs:  %s %s\n' "$DIM" "$RESET" "$LOG_DIR"
  printf '  %shost:  %s %s@%s\n' "$DIM" "$RESET" "$(whoami)" "$(hostname)"
  hr
}

# Pull the newest meaningful line out of a log for the status display.
# Tools like conda/pip/wget draw progress bars with carriage returns and ANSI
# cursor codes. Echoing those raw into our own status line corrupts the display,
# so: split on \r (each progress repaint becomes its own line, newest last),
# strip ANSI escapes, drop non-printables, then truncate to the width given.
last_log_line() {
  local f="$1" width="${2:-60}"
  tail -c 3000 "$f" 2>/dev/null \
    | tr '\r' '\n' \
    | sed -e 's/\x1b\[[0-9;?]*[A-Za-z]//g' -e 's/\x1b[()][A-B0-9]//g' -e 's/\x1b[=>]//g' \
    | tr -cd '[:print:]\n' \
    | grep -a '[^[:space:]]' \
    | tail -1 \
    | cut -c1-"$width"
}

fmt_dur() {
  local s=$1
  if   (( s < 60 ));   then printf '%ds' "$s"
  elif (( s < 3600 )); then printf '%dm%02ds' $((s/60)) $((s%60))
  else                      printf '%dh%02dm' $((s/3600)) $(((s%3600)/60)); fi
}

# ------------------------------------------------------- step runner + spinner
# run_step <id> <human description> <soft|hard> -- <command...>
#   soft = a non-zero exit is recorded as a warning and the script continues
#   hard = a non-zero exit aborts the script
run_step() {
  local id="$1" desc="$2" mode="$3"; shift 3
  [[ "$1" == "--" ]] && shift
  STEP_NO=$((STEP_NO + 1))
  local marker="$STATE_DIR/$id.done"
  local log="$LOG_DIR/$id.log"
  local tag
  tag=$(printf '[%2d/%d]' "$STEP_NO" "$TOTAL_STEPS")

  if [[ -f "$marker" && ! " $REDO_STEPS " == *" $id "* ]]; then
    printf '%s %s %s%s%s %s(already done — %s; --redo %s to repeat)%s\n' \
      "$tag" "${GREEN}✔${RESET}" "$BOLD" "$desc" "$RESET" "$DIM" "$(cat "$marker")" "$id" "$RESET"
    return 0
  fi

  printf '%s %s%s%s\n' "$tag" "$BOLD" "$desc" "$RESET"
  printf '      %slog: %s%s\n' "$DIM" "$log" "$RESET"

  local t0; t0=$(date +%s)
  : > "$log"
  ( "$@" ) >>"$log" 2>&1 &
  local pid=$!

  # --- live status line: elapsed time, last log line, stall detection ---------
  local frames='⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏'
  # shellcheck disable=SC2206
  local farr=($frames)
  local i=0 last_size=0 last_change t_now elapsed idle size lastline cols avail
  last_change=$t0
  while kill -0 "$pid" 2>/dev/null; do
    t_now=$(date +%s); elapsed=$((t_now - t0))
    size=$(stat -c%s "$log" 2>/dev/null || echo 0)
    if [[ "$size" != "$last_size" ]]; then last_size=$size; last_change=$t_now; fi
    idle=$((t_now - last_change))

    if (( IS_TTY )); then
      # Budget the tail text to what actually fits on ONE terminal row. If the
      # status line ever wraps, the trailing \r\033[K only clears the last row
      # and the wrapped remainder scribbles over previously printed step lines.
      cols=$(tput cols 2>/dev/null || echo 80)
      avail=$(( cols - 34 )); (( avail < 12 )) && avail=12
      lastline=$(last_log_line "$log" "$avail")
      if (( idle > STALL_WARN_SECS )); then
        printf '\r\033[K      %s%s%s %sworking… %s elapsed  %s[quiet %s — still alive]%s' \
          "$YELLOW" "${farr[i]}" "$RESET" "$DIM" "$(fmt_dur "$elapsed")" "$YELLOW" "$(fmt_dur "$idle")" "$RESET"
      else
        printf '\r\033[K      %s%s%s %s%s elapsed%s  %s%s%s' \
          "$CYAN" "${farr[i]}" "$RESET" "$DIM" "$(fmt_dur "$elapsed")" "$RESET" "$DIM" "$lastline" "$RESET"
      fi
    else
      # non-interactive (CI / nohup): heartbeat every 30s instead of a spinner
      if (( elapsed > 0 && elapsed % 30 == 0 )); then
        printf '      … still running (%s elapsed)\n' "$(fmt_dur "$elapsed")"
      fi
    fi
    i=$(( (i + 1) % ${#farr[@]} ))
    sleep 1
  done
  wait "$pid"; local rc=$?
  local t1; t1=$(date +%s); local dur=$((t1 - t0))
  (( IS_TTY )) && printf '\r\033[K'

  if (( rc == 0 )); then
    printf '      %s✔ done%s in %s\n' "$GREEN" "$RESET" "$(fmt_dur "$dur")"
    printf '%s (%s)\n' "$(date '+%Y-%m-%d %H:%M')" "$(fmt_dur "$dur")" > "$marker"
    return 0
  fi

  if [[ "$mode" == "soft" ]]; then
    printf '      %s! finished with errors%s (exit %d, %s) — continuing, later steps repair this\n' \
      "$YELLOW" "$RESET" "$rc" "$(fmt_dur "$dur")"
    warn "step '$id' exited $rc (tolerated)"
    return 0
  fi

  printf '      %s✘ FAILED%s (exit %d after %s)\n' "$RED" "$RESET" "$rc" "$(fmt_dur "$dur")"
  printf '      %slast 25 lines of %s:%s\n' "$DIM" "$log" "$RESET"
  tail -25 "$log" | sed 's/^/        /'
  FAILURES+=("step '$id' failed with exit $rc")
  printf '\n  %sFix the issue above, then re-run this script — completed steps are skipped.%s\n' "$BOLD" "$RESET"
  exit "$rc"
}

confirm() {
  (( ASSUME_YES )) && return 0
  local reply
  printf '  %s?%s %s [y/N] ' "$YELLOW" "$RESET" "$1"
  read -r reply </dev/tty || return 1
  [[ "$reply" =~ ^[Yy] ]]
}

# ------------------------------------------------------------ conda bootstrap
conda_sh() { echo "$CONDA_DIR/etc/profile.d/conda.sh"; }

# IMPORTANT: conda's shell hook and the per-package activate.d/ scripts are NOT
# written to be safe under `set -u`. MKL's libblas_mkl_activate.sh, for example,
# reads $MKL_INTERFACE_LAYER before assigning it, which aborts the whole script
# with "unbound variable". So nounset is lifted around conda calls only.
load_conda() {
  set +u
  # shellcheck disable=SC1090
  [[ -f "$(conda_sh)" ]] && source "$(conda_sh)"
  set -u
}

activate_env() {
  local env="${1:-$CONDA_ENV_NAME}" rc
  set +u
  conda activate "$env"; rc=$?
  set -u
  return $rc
}

mkdir -p "$PROJECT_ROOT" "$LOG_DIR" "$STATE_DIR"
banner

# =============================================================================
#  STEP 1 — preflight
# =============================================================================
step_preflight() {
  echo "== preflight =="
  echo "--- os ---";        (cat /etc/os-release || true)
  echo "--- kernel ---";    uname -a
  echo "--- cpu/ram ---";   nproc; free -h
  echo "--- disk ---";      df -h "$HOME"
  echo "--- gpu ---";       (nvidia-smi || echo "NO nvidia-smi")
  echo "--- nvcc ---";      (nvcc --version || echo "no system nvcc (fine)")
  echo "--- gcc ---";       (gcc --version || echo "no gcc")
  echo "--- tools ---";     for t in git cmake make wget curl unzip python3; do
                              printf '%-8s %s\n' "$t" "$(command -v "$t" || echo MISSING)"; done
}

preflight_report() {
  info "checking the machine before touching anything…"
  step_preflight > "$LOG_DIR/preflight.log" 2>&1

  # OS
  local osname; osname=$(. /etc/os-release && echo "$PRETTY_NAME")
  ok "OS: $osname"
  case "$osname" in
    *"24.04"*|*"22.04"*) ;;
    *) warn "untested Ubuntu release — this script targets 22.04 / 24.04" ;;
  esac

  # GPU
  # The binary existing is NOT the same as the driver working. After a kernel
  # upgrade the nvidia kernel module is often missing or unbuilt for the running
  # kernel: nvidia-smi is still on PATH but every query prints an error to stdout.
  # Feeding that string into (( ... )) makes bash parse "NVIDIA" as an identifier
  # and, under 'set -u', abort with "NVIDIA: unbound variable" — an error that
  # points nowhere near the actual cause. So probe that it RUNS, not that it exists.
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    local gpu vram drv
    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    ok "GPU: $gpu — ${vram} MiB VRAM, driver $drv"
    # guard the arithmetic anyway, in case a future driver changes the output format
    [[ "$vram" =~ ^[0-9]+$ ]] || { warn "could not parse VRAM from nvidia-smi ('$vram')"; vram=0; }
    if (( vram > 0 )) && (( vram < 6000 )); then
      warn "under 6 GB VRAM: FiGS *simulation/inference* is fine, but training a new splat "
      warn "  (splatfacto) will likely OOM — use --downscale-factor or train elsewhere"
    fi
    # driver must support CUDA 11.8 -> >= 450
    local drvmaj=${drv%%.*}
    if [[ "$drvmaj" =~ ^[0-9]+$ ]] && (( drvmaj < 520 )); then
      warn "driver $drv is old; conda's CUDA 11.8 needs >= 450, PyTorch 2.1.2 prefers >= 520"
    fi
  elif command -v nvidia-smi >/dev/null 2>&1; then
    # binary present but not talking to the kernel module — a HOST problem that
    # no amount of reinstalling this environment will fix.
    fail "nvidia-smi is installed but cannot reach the NVIDIA driver."
    say  "        $(nvidia-smi 2>&1 | head -1)"
    say  ""
    say  "        This is almost always one of:"
    say  "          a) kernel was upgraded and the nvidia DKMS module was not rebuilt"
    say  "             for the running kernel   -> sudo dkms status; sudo apt install --reinstall nvidia-dkms-<ver>"
    say  "          b) driver packages were updated but the machine has not rebooted"
    say  "             since, so the loaded module and userspace disagree  -> reboot"
    say  "          c) the module is simply not loaded  -> sudo modprobe nvidia"
    say  ""
    say  "        Diagnose with:  uname -r ; dkms status ; lsmod | grep -i nvidia"
    say  "        Nothing about FiGS can work until this is resolved."
    confirm "Continue anyway (CPU-only, will not work for rendering)?" || exit 1
  else
    fail "nvidia-smi not found — no usable NVIDIA GPU driver detected."
    say  "        FiGS needs CUDA. Install the driver first:  sudo ubuntu-drivers install"
    confirm "Continue anyway (CPU-only, will not work for rendering)?" || exit 1
  fi

  # disk — measured on the *install prefix*, since conda/caches/tmp all live there
  local freegb mnt fstype opts
  freegb=$(df -BG --output=avail "$PROJECT_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
  mnt=$(df --output=target "$PROJECT_ROOT" 2>/dev/null | tail -1)
  fstype=$(findmnt -no FSTYPE --target "$PROJECT_ROOT" 2>/dev/null)
  opts=$(findmnt -no OPTIONS --target "$PROJECT_ROOT" 2>/dev/null)
  local need=40; (( SKIP_GSPLATS )) && need=35
  if (( freegb < need )); then
    fail "only ${freegb} GB free on ${mnt} — need ~${need} GB (conda ~8 GB, env ~15 GB, gsplats ~5 GB, build)"
    confirm "Continue anyway?" || exit 1
  else
    ok "disk: ${freegb} GB free on ${mnt} (${fstype:-unknown})"
  fi

  # the install volume must support execution and POSIX permissions
  case "$opts" in
    *noexec*) fail "${mnt} is mounted 'noexec' — conda and the acados build cannot run from here."
              say  "        Fix: sudo mount -o remount,exec ${mnt}"
              confirm "Continue anyway (will almost certainly fail)?" || exit 1 ;;
  esac
  case "$fstype" in
    ext2|ext3|ext4|xfs|btrfs|zfs|"") ;;
    fuseblk|ntfs|vfat|exfat)
      fail "${mnt} is ${fstype} — no POSIX permissions or symlinks; conda WILL break here."
      confirm "Continue anyway (not recommended)?" || exit 1 ;;
    *) warn "${mnt} is ${fstype} — unusual for a conda install; watch for permission errors" ;;
  esac

  # $HOME still needs a little room: conda init edits .bashrc, git config, etc.
  local homefree; homefree=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9')
  (( homefree < 2 )) && warn "\$HOME has only ${homefree} GB free — tight, but caches are redirected to the prefix"

  ok "caches redirected off \$HOME: pip, conda pkgs, tmp, torch/HF → $PROJECT_ROOT"

  # ram
  local ramgb; ramgb=$(free -g | awk '/^Mem:/{print $2}')
  (( ramgb < 12 )) && warn "only ${ramgb} GB RAM — COLMAP / splat training may struggle" \
                   || ok "RAM: ${ramgb} GB"

  # sudo
  if sudo -n true 2>/dev/null; then
    ok "sudo: available (cached)"
  elif sudo -v 2>/dev/null; then
    ok "sudo: available"
  else
    warn "no sudo — apt steps will be skipped; you may need IT to install gcc-11/cmake"
  fi

  # --- portable-drive sanity: is this env still at the path it was built at? ---
  # Conda bakes absolute paths into shebangs, .pth files and activation hooks.
  # If the drive is mounted somewhere other than at install time, nothing works
  # and the errors are deeply unhelpful. Catch it here instead.
  local envdir="$CONDA_DIR/envs/$CONDA_ENV_NAME"
  if [[ -f "$envdir/bin/pip" ]]; then
    local baked
    baked=$(head -1 "$envdir/bin/pip" | sed 's|^#!||; s|/bin/python.*||')
    if [[ -n "$baked" && "$baked" != "$envdir" ]]; then
      fail "this conda env was built for a DIFFERENT path than it now sits at."
      say  "        built at : $baked"
      say  "        now at   : $envdir"
      say  "        Conda envs are not relocatable. Either mount the drive at the"
      say  "        original path, or rebuild with --redo conda_env."
      confirm "Continue anyway (expect breakage)?" || exit 1
    else
      ok "conda env sits at the path it was built at (relocation-safe)"
    fi
  fi

  # --- does tiny-cuda-nn match this GPU's architecture? ---
  local archfile="$STATE_DIR/tcnn_arch"
  if [[ -f "$archfile" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    local built now
    built=$(cat "$archfile")
    now=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    if [[ -n "$now" && -n "$built" && "$built" != "$now" ]]; then
      warn "tiny-cuda-nn was compiled for compute capability $built; this GPU is $now"
      warn "  rebuild:  $0 --prefix $PROJECT_ROOT --redo tcnn    (or use --new-host)"
    elif [[ -n "$now" ]]; then
      ok "tiny-cuda-nn matches this GPU (compute capability $now)"
    fi
  fi

  # --- stray packages in ~/.local that would shadow the env ---
  local usersite
  for usersite in "$HOME"/.local/lib/python3.*/site-packages; do
    [[ -d "$usersite" ]] || continue
    local shadow
    shadow=$(ls -1 "$usersite" 2>/dev/null | grep -Ex 'torch|torchvision|nerfstudio|gsplat|tinycudann|setuptools' | tr '\n' ' ')
    if [[ -n "$shadow" ]]; then
      fail "packages installed in USER SITE would shadow the conda env:"
      say  "        $usersite"
      say  "        offending: $shadow"
      say  "        Python searches user-site BEFORE the env, so these win — builds"
      say  "        link against the wrong libtorch. This script sets"
      say  "        PYTHONNOUSERSITE=1, but other tools and your own shells will not."
      say  "        Recommended:  mv $usersite $usersite.bak"
      confirm "Continue (this script ignores user-site, but clean-up is advised)?" || exit 1
    else
      ok "no conflicting packages in $usersite"
    fi
  done

  # --- is the torch stack internally consistent? ---
  # torch and torchvision are released in lockstep pairs. A mismatch means pip
  # upgraded one of them, which invalidates every compiled extension in the env.
  if [[ -x "$CONDA_DIR/envs/$CONDA_ENV_NAME/bin/python" ]]; then
    local P="$CONDA_DIR/envs/$CONDA_ENV_NAME/bin/python" tvf tvvf tcu
    tvf=$("$P"  -c "import torch;print(torch.__version__)" 2>/dev/null || echo "")
    tvvf=$("$P" -c "import torchvision;print(torchvision.__version__)" 2>/dev/null || echo "")
    tcu=$("$P"  -c "import torch;print(torch.version.cuda)" 2>/dev/null || echo "")
    if [[ -n "$tvf" ]]; then
      # expected pairs: torch 2.1.x <-> torchvision 0.16.x
      local tmin vmin
      tmin=$(echo "$tvf"  | cut -d. -f1-2)
      vmin=$(echo "$tvvf" | cut -d. -f1-2)
      if [[ "$tmin" == "2.1" && "$vmin" == "0.16" ]]; then
        ok "torch stack consistent: torch $tvf / torchvision $tvvf / CUDA $tcu"
      else
        fail "torch stack is INCONSISTENT: torch $tvf, torchvision $tvvf, CUDA $tcu"
        say  "        torch and torchvision ship as matched pairs (2.1.x <-> 0.16.x)."
        say  "        A mismatch means pip upgraded torch, which invalidates every"
        say  "        compiled extension (tiny-cuda-nn, gsplat)."
        if [[ " $REDO_STEPS " == *" conda_env "* ]]; then
          # the fix is already queued — no point asking
          info "conda_env is queued for rebuild, which repairs this. Continuing."
        else
          say  "        Rebuild:  $0 --prefix $PROJECT_ROOT --redo conda_env --redo tcnn --redo pips"
          confirm "Continue anyway (expect ImportError: undefined symbol)?" || exit 1
        fi
      fi
    fi
  fi

  (( NEW_HOST )) && info "--new-host: reusing the drive; redoing apt deps, host compiler, tiny-cuda-nn, env file"

  ok "preflight complete (full detail: $LOG_DIR/preflight.log)"
}

# =============================================================================
#  step implementations
# =============================================================================
do_apt_deps() {
  if ! sudo -n true 2>/dev/null && ! sudo -v 2>/dev/null; then
    echo "no sudo available — skipping apt stage"; return 0
  fi
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    git cmake build-essential pkg-config wget curl unzip ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg
}

do_miniconda() {
  if [[ -x "$CONDA_DIR/bin/conda" ]]; then
    # Present — but if the tree was copied from another path its shebangs point
    # at a python that no longer exists. Test it rather than trusting existence.
    if "$CONDA_DIR/bin/conda" --version >/dev/null 2>&1; then
      echo "conda already present and working at $CONDA_DIR"
      "$CONDA_DIR/bin/conda" --version
      accept_conda_tos
      return 0
    fi
    echo "conda exists at $CONDA_DIR but does not run — almost certainly because"
    echo "the tree was copied from a different path (shebangs are absolute)."
    echo "Reinstalling Miniconda IN PLACE with -u: this rewrites the launchers"
    echo "while preserving pkgs/ (the package cache) and envs/."
    head -1 "$CONDA_DIR/bin/conda" 2>/dev/null || true
  fi
  mkdir -p "$CONDA_DIR"
  wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
       -O "$TMPDIR/miniconda.sh"
  bash "$TMPDIR/miniconda.sh" -b -u -p "$CONDA_DIR"
  rm -f "$TMPDIR/miniconda.sh"

  # Deliberately NOT running `conda init bash` when conda lives outside $HOME:
  # it would hard-code this path into ~/.bashrc, and if the volume is ever
  # unmounted every new shell throws errors. figs_env.sh sources conda.sh
  # directly instead, which is safe either way.
  if [[ "$CONDA_DIR" == "$HOME/"* ]]; then
    "$CONDA_DIR/bin/conda" init bash
  else
    echo "conda is outside \$HOME — skipping 'conda init' to keep ~/.bashrc clean."
    echo "Use 'source $PROJECT_ROOT/figs_env.sh' to activate."
  fi
  "$CONDA_DIR/bin/conda" config --set auto_activate_base false
  "$CONDA_DIR/bin/conda" config --add pkgs_dirs "$CONDA_DIR/pkgs"
  accept_conda_tos
  "$CONDA_DIR/bin/conda" --version
}

# Conda >= 25.x refuses to solve until the Anaconda 'defaults' channel Terms of
# Service are explicitly accepted, failing in ~5s with CondaToSNonInteractiveError.
# environment_x86.yml pulls from defaults, so this must happen before env create.
accept_conda_tos() {
  local conda_bin="$CONDA_DIR/bin/conda"
  [[ -x "$conda_bin" ]] || return 0
  if ! "$conda_bin" tos --help >/dev/null 2>&1; then
    echo "this conda has no 'tos' subcommand (older release) — nothing to accept"
    return 0
  fi
  local ch
  for ch in https://repo.anaconda.com/pkgs/main https://repo.anaconda.com/pkgs/r; do
    echo "accepting ToS for $ch"
    "$conda_bin" tos accept --override-channels --channel "$ch" || \
      echo "  (could not accept $ch — continuing; conda-forge packages are unaffected)"
  done
}

# Git refuses to touch a repository owned by a different UID ("detected dubious
# ownership"). That happens constantly with a portable drive: the files carry the
# UID of whoever created them. Packages using setuptools_scm (acados_template)
# derive their version from git, so the failure surfaces as a build error rather
# than anything git-shaped. Register the repos as safe, idempotently.
register_safe_dirs() {
  local d
  for d in "$REPO_DIR" \
           "$REPO_DIR/FiGS" \
           "$REPO_DIR/FiGS/acados" \
           "$REPO_DIR/FiGS/Hierarchical-Localization"; do
    [[ -d "$d/.git" || -f "$d/.git" ]] || continue
    if ! git config --global --get-all safe.directory 2>/dev/null | grep -qxF "$d"; then
      git config --global --add safe.directory "$d"
      echo "git safe.directory += $d"
    fi
  done
}

do_clone() {
  mkdir -p "$PROJECT_ROOT"
  if [[ -d "$REPO_DIR/.git" ]]; then
    echo "repo already cloned; syncing submodules"
    git -C "$REPO_DIR" submodule update --recursive --init
  else
    git clone "$REPO_URL" "$REPO_DIR"
    git -C "$REPO_DIR" submodule update --recursive --init
  fi
  echo "--- submodule status ---"
  git -C "$REPO_DIR" submodule status --recursive
  for d in FiGS FiGS/acados FiGS/Hierarchical-Localization; do
    [[ -d "$REPO_DIR/$d" ]] || { echo "MISSING submodule: $d"; exit 1; }
  done
}

do_acados() {
  local a="$REPO_DIR/FiGS/acados"
  if [[ -f "$a/lib/libacados.so" ]]; then
    echo "acados already built ($a/lib/libacados.so)"; return 0
  fi
  mkdir -p "$a/build"
  cd "$a/build" || exit 1

  # CMake 4.x removed compatibility with cmake_minimum_required(VERSION < 3.5).
  # acados' vendored blasfeo/hpipm still declare 2.8, so configure dies. The
  # escape hatch below restores the old floor. Harmless on CMake < 3.31 (it is
  # simply an unused cache variable), so it is passed unconditionally.
  local cmver; cmver=$(cmake --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
  echo "cmake version: ${cmver:-unknown}"

  # A stale CMakeCache.txt from the failed run pins the old generator/paths.
  rm -f CMakeCache.txt
  rm -rf CMakeFiles

  cmake -DACADOS_WITH_QPOASES=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
  make install -j"$MAKE_JOBS"
  [[ -f "$a/lib/libacados.so" ]] || { echo "libacados.so not produced"; exit 1; }
  echo "acados built OK"
}

# Detect whether we need a downgraded host compiler for CUDA 11.8's nvcc.
# CUDA 11.8 refuses host GCC > 11; Ubuntu 24.04 ships GCC 13.
do_hostgcc() {
  local gccmaj; gccmaj=$(gcc -dumpversion 2>/dev/null | cut -d. -f1)
  echo "system gcc major version: ${gccmaj:-none}"
  if [[ -n "$gccmaj" ]] && (( gccmaj <= 11 )); then
    echo "system gcc is already <= 11 — nothing to do"
    echo "/usr/bin/gcc" > "$STATE_DIR/host_cc"
    echo "/usr/bin/g++" > "$STATE_DIR/host_cxx"
    return 0
  fi
  if command -v gcc-11 >/dev/null 2>&1; then
    echo "gcc-11 already installed"
  elif sudo -n true 2>/dev/null || sudo -v 2>/dev/null; then
    sudo apt-get update
    sudo apt-get install -y gcc-11 g++-11 || {
      echo "apt could not provide gcc-11; falling back to conda-forge compilers"
      load_conda; conda install -y -n "$CONDA_ENV_NAME" -c conda-forge gcc_linux-64=11 gxx_linux-64=11 || true
    }
  else
    echo "no sudo and no gcc-11 — will try conda-forge compilers later"
  fi
  if command -v gcc-11 >/dev/null 2>&1; then
    echo "$(command -v gcc-11)" > "$STATE_DIR/host_cc"
    echo "$(command -v g++-11)" > "$STATE_DIR/host_cxx"
    gcc-11 --version | head -1
  else
    echo "/usr/bin/gcc" > "$STATE_DIR/host_cc"
    echo "/usr/bin/g++" > "$STATE_DIR/host_cxx"
    echo "WARNING: proceeding with system gcc $gccmaj — tiny-cuda-nn may reject it"
  fi
}

do_conda_env() {
  load_conda
  accept_conda_tos          # idempotent; also covers a pre-existing conda install
  local yml
  for cand in "$REPO_DIR/environment_x86.yml" "$REPO_DIR/FiGS/environment_x86.yml"; do
    [[ -f "$cand" ]] && { yml="$cand"; break; }
  done
  [[ -n "${yml:-}" ]] || { echo "environment_x86.yml not found under $REPO_DIR"; exit 1; }
  echo "using $yml"

  local envdir="$CONDA_DIR/envs/$CONDA_ENV_NAME"

  # A copied/moved env is unusable: conda writes absolute paths into shebangs,
  # .pth files and activate.d hooks. Detect and remove it so it gets rebuilt.
  local stale=0
  if [[ -f "$envdir/bin/pip" ]]; then
    local baked
    baked=$(head -1 "$envdir/bin/pip" | sed 's|^#!||; s|/bin/python.*||')
    if [[ -n "$baked" && "$baked" != "$envdir" ]]; then
      echo "existing env was built for '$baked' but now sits at '$envdir'"
      stale=1
    fi
  fi
  # An explicit '--redo conda_env' means REBUILD, not "skip because it exists".
  local forced=0
  [[ " $REDO_STEPS " == *" conda_env "* ]] && forced=1

  if (( RELOCATED )) || (( stale )) || (( forced )); then
    if [[ -d "$envdir" ]]; then
      echo "removing existing env (forced=$forced stale=$stale relocated=$RELOCATED)"
      echo "the package cache in $CONDA_DIR/pkgs is KEPT, so the rebuild links"
      echo "from local cache rather than re-downloading"
      conda env remove -n "$CONDA_ENV_NAME" -y 2>/dev/null || true
      rm -rf "$envdir"          # conda sometimes leaves the directory behind
      [[ -d "$envdir" ]] && { echo "FATAL: could not remove $envdir"; exit 1; }
    fi
  elif conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    echo "conda env '$CONDA_ENV_NAME' already exists and looks valid — skipping creation"
    return 0
  fi

  echo "--- package cache available for offline reuse ---"
  du -sh "$CONDA_DIR/pkgs" 2>/dev/null || echo "(no package cache — will download)"
  du -sh "$PIP_CACHE_DIR" 2>/dev/null || true

  cd "$REPO_DIR" || exit 1
  # NOTE: the pip section of this yml (tiny-cuda-nn) is EXPECTED to fail here.
  # The 'tcnn' and 'pips' steps below repair it. That is why this step is 'soft'.
  conda env create -f "$yml"
}

# Assert the env exists even if the previous step reported errors.
assert_env() {
  load_conda
  if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    fail "conda env '$CONDA_ENV_NAME' was not created — see $LOG_DIR/conda_env.log"
    # translate the most common failure modes into an actionable next command
    local l="$LOG_DIR/conda_env.log"
    if grep -q 'CondaToSNonInteractiveError\|Terms of Service' "$l" 2>/dev/null; then
      say "        Cause: Anaconda channel Terms of Service not accepted (conda >= 25.x)."
      say "        Fix:   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main"
      say "               conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r"
    elif grep -q 'PackagesNotFoundError\|ResolvePackageNotFound' "$l" 2>/dev/null; then
      say "        Cause: a pinned package is unavailable for this platform."
      say "        Check the package name printed in the log against the yml."
    elif grep -qi 'permission denied\|EACCES' "$l" 2>/dev/null; then
      say "        Cause: cannot write to the package cache or envs dir."
      say "        Check: ls -ld $CONDA_DIR/pkgs $CONDA_DIR/envs"
    elif grep -qi 'CondaHTTPError\|ConnectionError\|Timeout' "$l" 2>/dev/null; then
      say "        Cause: network/proxy problem reaching the conda channels."
      say "        On a university network you may need HTTP(S)_PROXY set."
    fi
    say "        Then re-run:  $0 --prefix $PROJECT_ROOT --redo conda_env"
    exit 1
  fi
}

# Write a constraints file pinning the CURRENTLY INSTALLED torch stack.
# Every compiled extension in this env (tiny-cuda-nn, gsplat) is built against a
# specific libtorch ABI, so torch must never move underneath them. tiny-cuda-nn's
# own metadata declares an UNBOUNDED 'torch' dependency, so without this pip
# cheerfully upgrades torch 2.1.2 -> latest+cu13x mid-install and every extension
# then dies with 'undefined symbol: at::_ops::...'.
write_pip_constraints() {
  local cons="$STATE_DIR/pip-constraints.txt"
  local tv tvv nv
  tv=$(python  -c "import torch;print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
  tvv=$(python -c "import torchvision;print(torchvision.__version__.split('+')[0])" 2>/dev/null || echo "")
  # numpy matters as much as torch here: numpy 2.x broke the C ABI, so letting a
  # dependency pull numpy>=2 into an env whose extensions were built against 1.x
  # produces "_ARRAY_API not found" / segfaults at import time.
  nv=$(python  -c "import numpy;print(numpy.__version__)" 2>/dev/null || echo "")
  : > "$cons"
  [[ -n "$tv"  ]] && echo "torch==$tv"        >> "$cons"
  [[ -n "$tvv" ]] && echo "torchvision==$tvv" >> "$cons"
  [[ -n "$nv"  ]] && echo "numpy==$nv"        >> "$cons"
  echo "$cons"
}

assert_torch_unchanged() {
  local before="$1" after
  after=$(python -c "import torch;print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
  if [[ -n "$before" && -n "$after" && "$before" != "$after" ]]; then
    echo "FATAL: torch changed $before -> $after."
    echo "       Every compiled extension (tiny-cuda-nn, gsplat) is now invalid."
    echo "       Rebuild the env:  $0 --prefix $PROJECT_ROOT --redo conda_env --redo tcnn --redo pips"
    return 1
  fi
  return 0
}

do_tcnn() {
  load_conda
  activate_env

  local TORCH0 CONS
  TORCH0=$(python -c "import torch;print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
  CONS=$(write_pip_constraints)
  echo "--- pinning the torch stack for this step ---"; cat "$CONS"

  # --- Fix 1: build isolation pulls an unpinned setuptools with no pkg_resources
  pip install -c "$CONS" "setuptools<81" wheel packaging ninja

  # --- Fix 2: conda's own CUDA 11.8 toolkit must precede system CUDA on PATH
  export PATH="$CONDA_PREFIX/bin:$PATH"
  export CUDA_HOME="$CONDA_PREFIX"
  echo "--- nvcc in use ---"; nvcc --version || echo "no nvcc in env!"

  # --- Fix 3: CUDA 11.8 nvcc rejects gcc > 11
  export CC="$(cat "$STATE_DIR/host_cc" 2>/dev/null || echo /usr/bin/gcc)"
  export CXX="$(cat "$STATE_DIR/host_cxx" 2>/dev/null || echo /usr/bin/g++)"
  echo "--- host compiler ---"; "$CC" --version | head -1

  # target the actual GPU so we don't compile every architecture
  if command -v nvidia-smi >/dev/null 2>&1; then
    local cc; cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    if [[ -n "$cc" ]]; then
      export TCNN_CUDA_ARCHITECTURES="$cc"
      echo "TCNN_CUDA_ARCHITECTURES=$cc"
      echo "$cc" > "$STATE_DIR/tcnn_arch"   # so a later run on another GPU notices
    fi
  fi

  echo "--- torch this build must match ---"
  python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)" || true

  if python -c "import tinycudann" 2>/dev/null; then
    echo "tinycudann already importable — skipping build"; return 0
  fi
  echo "tinycudann not importable — reason:"
  python -c "import tinycudann" 2>&1 | tail -3 || true

  local TCNN_SRC="git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

  echo ">>> building tiny-cuda-nn — this compiles CUDA kernels and can take 15-40 min"
  echo ">>> long silences are normal here; nvcc does not print progress"
  # --no-deps: tiny-cuda-nn declares an unbounded 'torch' requirement, and torch
  # is already present. Letting pip resolve it upgrades torch and breaks the env.
  # -c is belt-and-braces in case a transitive dep slips through.
  pip install --no-build-isolation --no-deps -c "$CONS" "$TCNN_SRC"
  assert_torch_unchanged "$TORCH0" || exit 1

  # pip's wheel cache is keyed on the SOURCE COMMIT only — not on the torch
  # version or the GPU architecture the extension was compiled against. So a
  # wheel cached on another machine (or before a torch upgrade) gets reused and
  # then fails to load with an 'undefined symbol: at::_ops::...' ImportError.
  # If that happens, discard the cache and genuinely recompile.
  if ! python -c "import tinycudann" 2>/dev/null; then
    echo
    echo "!!! tinycudann installed but will not import. Most likely a stale cached"
    echo "!!! wheel built against a different torch or GPU arch. Details:"
    python -c "import tinycudann" 2>&1 | tail -5 || true
    echo
    echo ">>> discarding cached wheel and recompiling from source (no cache)"
    pip cache remove 'tinycudann*' 2>/dev/null || true
    pip cache remove 'tiny_cuda_nn*' 2>/dev/null || true
    pip uninstall -y tinycudann 2>/dev/null || true
    pip install --no-build-isolation --no-deps --no-cache-dir --force-reinstall \
                -c "$CONS" "$TCNN_SRC"
    assert_torch_unchanged "$TORCH0" || exit 1
  fi

  python -c "import tinycudann; print('tinycudann OK')"
  # prove the CUDA kernels actually run, not just that the module imports
  python - <<'PY'
import torch, tinycudann as tcnn
enc = tcnn.Encoding(3, {"otype":"HashGrid","n_levels":4,"n_features_per_level":2,
                        "log2_hashmap_size":15,"base_resolution":16,
                        "per_level_scale":1.5}).cuda()
print("tinycudann forward pass OK:", tuple(enc(torch.rand(8,3,device="cuda")).shape))
PY
}

do_pips() {
  load_conda
  activate_env
  export PATH="$CONDA_PREFIX/bin:$PATH"
  export CUDA_HOME="$CONDA_PREFIX"
  export CC="$(cat "$STATE_DIR/host_cc" 2>/dev/null || echo /usr/bin/gcc)"
  export CXX="$(cat "$STATE_DIR/host_cxx" 2>/dev/null || echo /usr/bin/g++)"
  cd "$REPO_DIR" || exit 1

  # ---------------------------------------------------------------------------
  # Pin the torch stack for every pip call below. Without this, a transitive
  # dependency can quietly upgrade torch — which silently breaks tiny-cuda-nn
  # and gsplat, because both are C++ extensions compiled against a specific
  # libtorch ABI. The symptom is an unrelated-looking
  #   ImportError: ... undefined symbol: at::_ops::...
  # much later. A hard pin turns that into an immediate, legible resolver error.
  # ---------------------------------------------------------------------------
  local CONS tv
  tv=$(python -c "import torch;print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
  CONS=$(write_pip_constraints)
  echo "--- pip constraints (protecting the compiled-extension ABI) ---"
  cat "$CONS"

  local PIPC=(); [[ -s "$CONS" ]] && PIPC=(-c "$CONS")

  # setuptools_scm in acados_template reads git metadata; on a portable drive the
  # repo UID differs from ours and git bails out unless it is marked safe.
  register_safe_dirs

  # Every install below is checked. Previously a failure here was masked because
  # the function's exit status came from the LAST command, so a broken editable
  # install still reported the step as successful.
  local rc=0
  _pipdo() {
    local what="$1"; shift
    echo "=== pip: $what ==="
    if ! pip install "$@"; then
      echo "!!! FAILED: $what"
      rc=1
    fi
  }

  python -c "import nerfstudio" 2>/dev/null \
    || _pipdo "nerfstudio==1.1.4" "${PIPC[@]}" nerfstudio==1.1.4
  _pipdo "gdown"           "${PIPC[@]}" gdown

  # The editables DO need their dependencies resolved — acados_template imports
  # casadi at module load, so --no-deps here just trades a torch problem for a
  # ModuleNotFoundError. The constraints file is what protects the ABI-sensitive
  # packages (torch / torchvision / numpy); everything else may resolve freely.
  _pipdo "FiGS (editable)"            "${PIPC[@]}" -e ./FiGS/
  _pipdo "acados_template (editable)" "${PIPC[@]}" -e ./FiGS/acados/interfaces/acados_template/
  _pipdo "hloc (editable)"            "${PIPC[@]}" -e ./FiGS/Hierarchical-Localization/

  # confirm each one actually imports — installing is not the same as working
  # acados_template imports casadi at module load; make sure it is present rather
  # than discovering it via an import failure below.
  python -c "import casadi" 2>/dev/null || _pipdo "casadi" "${PIPC[@]}" casadi

  local m
  for m in nerfstudio figs acados_template hloc casadi; do
    if python -c "import $m" 2>/dev/null; then
      echo "import OK: $m"
    else
      echo "!!! import FAILED: $m"
      python -c "import $m" 2>&1 | tail -3
      rc=1
    fi
  done
  (( rc )) && { echo "one or more installs/imports failed (see above)"; return 1; }

  echo "--- final versions ---"
  pip list | grep -Ei 'nerfstudio|gsplat|torch|tinycudann|acados|hloc|figs|pycolmap' || true

  assert_torch_unchanged "$tv" || exit 1
}

do_gsplats() {
  load_conda
  activate_env
  cd "$REPO_DIR" || exit 1
  if [[ -d gsplats/capture || -d gsplats/workspace ]]; then
    echo "gsplats already present"; du -sh gsplats; return 0
  fi
  echo ">>> attempting gdown (~5 GB). Google Drive quota-blocks this link often."
  if gdown "https://drive.google.com/uc?id=$GSPLAT_GDRIVE_ID" -O gsplats.zip; then
    unzip -q gsplats.zip -d gsplats
    # the zip sometimes has an extra nesting level
    if [[ -d gsplats/gsplats ]]; then
      mv gsplats/gsplats/* gsplats/ 2>/dev/null || true
      rm -rf gsplats/gsplats
    fi
    rm -f gsplats.zip
    ls -la gsplats
  else
    cat <<EOF

  ============================================================================
   gdown FAILED — almost certainly Google Drive's per-file download quota on
   this shared academic link. Nothing is wrong with your setup.

   Workaround: download the zip in a normal browser, then scp it over:

     scp gsplats.zip $(whoami)@$(hostname -I 2>/dev/null | awk '{print $1}'):$REPO_DIR/gsplats.zip

   ...or copy it from a machine that already has it. Then on this machine:

     cd $REPO_DIR
     unzip -q gsplats.zip -d gsplats
     [ -d gsplats/gsplats ] && mv gsplats/gsplats/* gsplats/ && rm -rf gsplats/gsplats

   The rest of the install is unaffected — only the example notebook needs this.
  ============================================================================

EOF
    return 3   # soft step: recorded as a warning, install continues
  fi
}

do_envfile() {
  local f="$PROJECT_ROOT/figs_env.sh"
  cat > "$f" <<EOF
#!/usr/bin/env bash
# Source this before working with FiGS:   source $f
# conda's activate.d hooks (MKL etc.) are not 'set -u' safe, so nounset is
# lifted across activation and restored afterwards if the caller had it on.
case \$- in *u*) _figs_had_u=1;; *) _figs_had_u=0;; esac
set +u
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate $CONDA_ENV_NAME
[ "\$_figs_had_u" = 1 ] && set -u
unset _figs_had_u

export ACADOS_SOURCE_DIR="$REPO_DIR/FiGS/acados"
export LD_LIBRARY_PATH="\$ACADOS_SOURCE_DIR/lib:\${LD_LIBRARY_PATH:-}"

# Never let ~/.local/lib/pythonX.Y/site-packages shadow this env — it sits ahead
# of the env on sys.path and will silently override torch and friends.
export PYTHONNOUSERSITE=1
export PIP_USER=0

# keep caches off \$HOME (matches what the installer used)
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export CONDA_PKGS_DIRS="$CONDA_DIR/pkgs"
export TMPDIR="$PROJECT_ROOT/.tmp"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"

# Only needed when *compiling* CUDA extensions (e.g. rebuilding tiny-cuda-nn):
export CUDA_HOME="\$CONDA_PREFIX"
export PATH="\$CONDA_PREFIX/bin:\$PATH"
export CC="$(cat "$STATE_DIR/host_cc" 2>/dev/null || echo /usr/bin/gcc)"
export CXX="$(cat "$STATE_DIR/host_cxx" 2>/dev/null || echo /usr/bin/g++)"

cd "$REPO_DIR"
EOF
  chmod +x "$f"
  echo "wrote $f"
  cat "$f"
}

# =============================================================================
#  verification
# =============================================================================
CHECK_PASS=0; CHECK_FAIL=0; CHECK_WARN=0
check() {  # check "<label>" "<shell command>"
  local label="$1"; shift
  printf '  %-42s' "$label"
  local out
  if out=$( eval "$@" 2>&1 ); then
    printf '%s✔%s %s\n' "$GREEN" "$RESET" "$(echo "$out" | head -1 | cut -c1-40)"
    CHECK_PASS=$((CHECK_PASS+1))
  else
    printf '%s✘%s %s\n' "$RED" "$RESET" "$(echo "$out" | tail -1 | cut -c1-60)"
    CHECK_FAIL=$((CHECK_FAIL+1))
  fi
}
check_soft() {
  local label="$1"; shift
  printf '  %-42s' "$label"
  local out
  if out=$( eval "$@" 2>&1 ); then
    printf '%s✔%s %s\n' "$GREEN" "$RESET" "$(echo "$out" | head -1 | cut -c1-40)"
    CHECK_PASS=$((CHECK_PASS+1))
  else
    printf '%s!%s %s\n' "$YELLOW" "$RESET" "$(echo "$out" | tail -1 | cut -c1-60)"
    CHECK_WARN=$((CHECK_WARN+1))
  fi
}

run_verification() {
  hr
  printf '%s%s  Verification%s\n' "$BOLD" "$CYAN" "$RESET"
  hr
  load_conda
  activate_env 2>/dev/null || { fail "cannot activate env '$CONDA_ENV_NAME' — see above"; return 1; }
  export ACADOS_SOURCE_DIR="$REPO_DIR/FiGS/acados"
  export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"

  check      "python version"          "python -c 'import sys;print(sys.version.split()[0])'"
  check      "torch"                   "python -c 'import torch;print(torch.__version__)'"
  check      "torch.cuda.is_available" "python -c 'import torch;assert torch.cuda.is_available();print(torch.cuda.get_device_name(0))'"
  check      "torch CUDA build"        "python -c 'import torch;print(torch.version.cuda)'"
  check      "tinycudann"              "python -c 'import tinycudann as t;print(\"ok\")'"
  # nerfstudio has no module-level __version__; read it from package metadata
  check      "nerfstudio"              "python -c 'import nerfstudio,importlib.metadata as m;print(m.version(\"nerfstudio\"))'"
  check      "gsplat.rasterization"    "python -c 'from gsplat import rasterization;print(\"ok\")'"
  check      "hloc"                    "python -c 'import hloc;print(\"ok\")'"
  check      "figs"                    "python -c 'import figs;print(\"ok\")'"
  check      "acados_template"         "python -c 'import acados_template;print(\"ok\")'"
  check      "casadi (acados dep)"     "python -c 'import casadi;print(casadi.__version__)'"
  # numpy 2.x broke the C ABI; extensions built against 1.x fail at import
  check      "numpy 1.x ABI"           "python -c 'import numpy;v=numpy.__version__;assert int(v.split(\".\")[0])<2,f\"numpy {v} breaks extensions built for 1.x\";print(v)'"
  check      "libacados.so"            "test -f '$REPO_DIR/FiGS/acados/lib/libacados.so' && echo present"
  # 'colmap -h' prints its banner then exits non-zero; with pipefail that would
  # sink the whole pipeline, so swallow its status and judge on the output.
  check      "colmap"                  "{ colmap -h 2>&1 || true; } | grep -m1 -F 'COLMAP'"
  check      "ns-train splatfacto"     "ns-train --help 2>&1 | grep -q splatfacto && echo registered"
  check_soft "example gsplats present" "test -d '$REPO_DIR/gsplats/capture' && du -sh '$REPO_DIR/gsplats'"
  check_soft "GPU idle VRAM"           "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"

  # a real (tiny) CUDA op — catches driver/toolkit mismatches that imports miss
  check      "live CUDA matmul" \
    "python -c 'import torch;a=torch.randn(256,256,device=\"cuda\");print((a@a).sum().item())'"

  hr
  printf '  %s%d passed%s   %s%d warnings%s   %s%d failed%s\n' \
    "$GREEN" "$CHECK_PASS" "$RESET" "$YELLOW" "$CHECK_WARN" "$RESET" "$RED" "$CHECK_FAIL" "$RESET"
  hr
  return $(( CHECK_FAIL > 0 ? 1 : 0 ))
}

# =============================================================================
#  main
# =============================================================================
if (( VERIFY_ONLY )); then
  run_verification; exit $?
fi

preflight_report
echo
printf '%sPlan:%s 12 steps. Biggest time sinks: conda env (~15-25 min), tiny-cuda-nn build (~15-40 min), gsplats (~5 GB).\n' "$BOLD" "$RESET"
printf 'Total expected: %s45-90 minutes%s on a decent connection. Safe to Ctrl-C and re-run — progress is saved.\n\n' "$BOLD" "$RESET"
if ! confirm "Proceed with install into $PROJECT_ROOT?"; then echo "Aborted."; exit 0; fi
echo

run_step preflight "System preflight recorded"                 hard -- true
run_step apt_deps  "APT build dependencies"                    hard -- do_apt_deps
run_step miniconda "Miniconda"                                 hard -- do_miniconda
run_step clone     "Clone SousVide + submodules"               hard -- do_clone
run_step acados    "Build acados (MPC solver)"                 hard -- do_acados
run_step hostgcc   "CUDA-compatible host compiler"             hard -- do_hostgcc
run_step conda_env "Conda env '$CONDA_ENV_NAME' (pip stage expected to fail — repaired next)" soft -- do_conda_env
assert_env
run_step tcnn      "Build tiny-cuda-nn (longest step)"         hard -- do_tcnn
run_step pips      "nerfstudio + editable FiGS/acados/hloc"    hard -- do_pips
if (( SKIP_GSPLATS )); then
  STEP_NO=$((STEP_NO+1)); info "[$STEP_NO/$TOTAL_STEPS] Example GSplats — skipped (--skip-gsplats)"
else
  run_step gsplats "Example GSplat dataset (~5 GB)"            soft -- do_gsplats
fi
run_step envfile   "Write figs_env.sh"                         hard -- do_envfile

echo
run_verification; VERIFY_RC=$?

# ------------------------------------------------------------------- summary
TOTAL=$(( $(date +%s) - SCRIPT_START ))
hr
if (( VERIFY_RC == 0 )); then
  printf '%s%s  ✔ FiGS install complete%s  (total %s)\n' "$BOLD" "$GREEN" "$RESET" "$(fmt_dur "$TOTAL")"
else
  printf '%s%s  ✘ Install finished with FAILED checks%s  (total %s)\n' "$BOLD" "$RED" "$RESET" "$(fmt_dur "$TOTAL")"
fi
hr
if (( ${#WARNINGS[@]} )); then
  printf '%sWarnings:%s\n' "$YELLOW" "$RESET"
  printf '  - %s\n' "${WARNINGS[@]}"
  echo
fi
cat <<EOF
${BOLD}Next steps${RESET}
  source $PROJECT_ROOT/figs_env.sh          # activates 'kitchen' + acados env vars
  cd $REPO_DIR

  # headless smoke test (needs the example gsplats):
  jupyter nbconvert --to notebook --execute notebooks/figs_examples.ipynb \\
    --output executed_figs_examples.ipynb --ExecutePreprocessor.timeout=1800

  ${DIM}NOTE: notebooks/figs_examples.ipynb defaults to capture_name="button", which is
  NOT in the downloadable dataset. Change it to "backroom" and set
  scene_name, course_name = "backroom", "circuit" (both lines exist, commented out).${RESET}

  Re-run checks any time:   $0 --verify-only
  Logs:                     $LOG_DIR
EOF
exit "$VERIFY_RC"
