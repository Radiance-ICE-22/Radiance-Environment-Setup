#!/usr/bin/env bash
# run_ui.sh — start, stop and check Galley (the pipeline web UI) on this machine.
#
#   ./run_ui.sh              start in the background (tmux session "galley"), wait until it answers
#   ./run_ui.sh --pull       git pull --ff-only first, then start (restarts it if it was running)
#   ./run_ui.sh fg           run in this terminal instead (Ctrl-C stops it)
#   ./run_ui.sh status       is it up, which profile, is a job running
#   ./run_ui.sh logs         follow the server log
#   ./run_ui.sh restart      stop + start (refuses while a pipeline job is running)
#   ./run_ui.sh stop         stop it (refuses while a pipeline job is running; --force to override)
#
# What "start" does, every time, and only when needed:
#   1. finds the machine profile: $GALLEY_MACHINE, else ui/machine.toml, else
#      ui/machines/<hostname prefix>.toml (intellisense08.toml, dummy.toml, ...)
#   2. creates ui/backend/.venv with uv and installs the backend, or reinstalls it when
#      ui/backend/pyproject.toml changed. Never touches the kitchen conda env.
#   3. checks the FiGS install the profile points at (figs_env.sh, figs_pipeline.py) and the
#      committed frontend build (ui/frontend/dist) — no Node.js needed
#   4. starts the server, waits for /api/health, prints how to open it
#
# Galley runs pipeline jobs itself, one at a time, through figs_env.sh. Don't run GPU work
# from a shell while it is in use: the queue assumes it owns the GPU.
set -euo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
UI=$REPO/ui
BACKEND=$UI/backend
VENV=$BACKEND/.venv
PY=$VENV/bin/python
SESSION=${GALLEY_TMUX_SESSION:-galley}

c_ok=$'\e[32m'; c_warn=$'\e[33m'; c_err=$'\e[31m'; c_dim=$'\e[2m'; c_off=$'\e[0m'
[ -t 1 ] || { c_ok=; c_warn=; c_err=; c_dim=; c_off=; }
ok()   { echo "  ${c_ok}✔${c_off} $*"; }
warn() { echo "  ${c_warn}!${c_off} $*"; }
die()  { echo "  ${c_err}✗${c_off} $*" >&2; exit 1; }
step() { echo; echo "${c_dim}── $*${c_off}"; }

usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

CMD=start PULL=0 FORCE=0
for a in "$@"; do
    case "$a" in
        start|stop|restart|status|logs|fg) CMD=$a ;;
        --pull) PULL=1 ;;
        --force) FORCE=1 ;;
        -h|--help|help) usage 0 ;;
        *) echo "unknown argument: $a"; usage 1 ;;
    esac
done

# ── backend venv ─────────────────────────────────────────────────────────────
find_uv() {
    UV=$(command -v uv || true)
    [ -n "$UV" ] || { [ -x ~/.local/bin/uv ] && UV=~/.local/bin/uv; } || true
    [ -n "${UV:-}" ] || die "uv not found. Install it once (user-level, no sudo):
      curl -LsSf https://astral.sh/uv/install.sh | sh"
}

ensure_venv() {
    local stamp=$VENV/.galley_pyproject.sha256 want
    want=$(sha256sum "$BACKEND/pyproject.toml" | cut -d' ' -f1)
    if [ -x "$PY" ] && [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$want" ]; then
        ok "backend venv up to date ($VENV)"
        return
    fi
    find_uv
    if [ ! -x "$PY" ]; then
        echo "  creating $VENV"
        "$UV" venv -q "$VENV"
    else
        echo "  pyproject.toml changed: reinstalling the backend"
    fi
    (cd "$BACKEND" && "$UV" pip install -q -p "$VENV" -e '.[test]')
    echo "$want" > "$stamp"
    ok "backend installed"
}

# ── profile: ask Galley itself, so the script and the server can't disagree ──
read_profile() {
    local out
    out=$(cd "$BACKEND" && "$PY" - <<'EOF'
import shlex, sys
from galley.settings import load, NoMachineProfile
try:
    s = load()
except NoMachineProfile as e:
    print(f"ERR={shlex.quote(str(e))}"); sys.exit(0)
for k, v in {"PROFILE": s.source, "HOST": s.host, "PORT": s.port, "PROJECT_ROOT": s.project_root,
             "ENV_SCRIPT": s.env_script, "PIPELINE": s.pipeline, "DATA_DIR": s.data_dir,
             "TOKEN": s.token or ""}.items():
    print(f"{k}={shlex.quote(str(v))}")
EOF
)
    eval "$out"
    [ -z "${ERR:-}" ] || die "$ERR"
    LOG=$DATA_DIR/galley.log
    URL_HOST=$HOST; [ "$HOST" = "0.0.0.0" ] && URL_HOST=127.0.0.1
    BASE=http://$URL_HOST:$PORT
}

health() { curl -sf --max-time 2 "$BASE/api/health" 2>/dev/null; }
running_job() { health | "$PY" -c 'import json,sys; j=json.load(sys.stdin).get("current_job"); print(j if j is not None else "")' 2>/dev/null || true; }
in_tmux() { command -v tmux >/dev/null && tmux has-session -t "$SESSION" 2>/dev/null; }

checks() {
    [ -f "$ENV_SCRIPT" ] && ok "FiGS install: $PROJECT_ROOT" \
        || die "no figs_env.sh in $PROJECT_ROOT — install FiGS first (setup_scripts/install_figs.sh) or fix [paths] project_root in $PROFILE"
    [ -f "$PIPELINE" ] && ok "pipeline: $PIPELINE" || die "figs_pipeline.py not found at $PIPELINE (profile $PROFILE)"
    [ -f "$UI/frontend/dist/index.html" ] && ok "frontend build present (ui/frontend/dist)" \
        || die "ui/frontend/dist is missing: git pull (it is committed), or build it: cd ui/frontend && npm ci && npm run build"
    if [ -d "$UI/frontend/src" ] && [ -n "$(find "$UI/frontend/src" -newer "$UI/frontend/dist/index.html" -type f -print -quit 2>/dev/null)" ]; then
        warn "frontend sources are newer than dist/ — fine after a git pull; if you edited src/ here, rebuild (npm run build)"
    fi
    if command -v nvidia-smi >/dev/null; then
        local apps
        apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/      /' || true)
        if [ -n "$apps" ]; then
            warn "GPU already in use by other processes (Galley's jobs may run out of memory):"; echo "$apps"
        else
            ok "GPU idle: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
        fi
    fi
}

how_to_open() {
    echo
    echo "  Galley is up: ${c_ok}$BASE${c_off}   (API docs: $BASE/docs)"
    [ -n "$TOKEN" ] && echo "  token required: the browser asks for it (Authorization: Bearer …, or ?token=…)"
    if [ "$HOST" = "127.0.0.1" ] || [ "$HOST" = "localhost" ]; then
        echo "  It listens on this machine only. From your laptop, either:"
        echo "    • VS Code Remote-SSH: Ports tab → Forward a Port → $PORT, then open the address it shows"
        echo "    • a tunnel:  ssh -N -L 18800:localhost:$PORT $(whoami)@$(hostname)   →  http://localhost:18800"
        echo "      (18800 on the Windows side: Windows often reserves 8800)"
    else
        local ts; ts=$(command -v tailscale >/dev/null && tailscale ip -4 2>/dev/null | head -1 || true)
        echo "  It listens on all interfaces: http://$(hostname):$PORT${ts:+  or  http://$ts:$PORT (Tailscale)}"
    fi
    echo "  logs: ./run_ui.sh logs     stop: ./run_ui.sh stop"
}

do_stop() {
    local job; job=$(running_job)
    if [ -n "$job" ] && [ "$FORCE" != 1 ]; then
        die "job #$job is running. Stopping Galley now orphans it (it keeps the GPU, its log is lost, and it is marked
      'interrupted'). Cancel it on the Jobs page first, wait for it, or use: ./run_ui.sh stop --force"
    fi
    if in_tmux; then
        tmux send-keys -t "$SESSION" C-c 2>/dev/null || true
        for _ in $(seq 20); do health >/dev/null || break; sleep 0.5; done
        tmux kill-session -t "$SESSION" 2>/dev/null || true
        ok "stopped (tmux session $SESSION)"
    elif [ -f "$DATA_DIR/galley.pid" ] && kill -0 "$(cat "$DATA_DIR/galley.pid")" 2>/dev/null; then
        kill -INT "$(cat "$DATA_DIR/galley.pid")"
        for _ in $(seq 20); do health >/dev/null || break; sleep 0.5; done
        rm -f "$DATA_DIR/galley.pid"
        ok "stopped"
    elif health >/dev/null; then
        die "something answers on $BASE but was not started by run_ui.sh (a server in another terminal?). Stop it there."
    else
        ok "not running"
    fi
}

do_start() {
    step "1. profile and backend"
    ensure_venv
    read_profile
    ok "profile: $PROFILE  (serves on $HOST:$PORT)"

    if health >/dev/null; then
        ok "already running"
        how_to_open
        return
    fi
    if command -v ss >/dev/null && ss -ltn "( sport = :$PORT )" 2>/dev/null | grep -q LISTEN; then
        die "port $PORT is taken by another program: ss -ltnp '( sport = :$PORT )' shows which"
    fi

    step "2. checks"
    checks

    step "3. start"
    mkdir -p "$DATA_DIR"
    local run="cd '$BACKEND' && exec '$PY' -m galley"
    [ -n "${GALLEY_MACHINE:-}" ] && run="export GALLEY_MACHINE='$GALLEY_MACHINE'; $run"
    echo "=== $(date '+%F %T') start ($PROFILE)" >> "$LOG"
    if command -v tmux >/dev/null; then
        tmux new-session -d -s "$SESSION" "bash -c \"$run 2>&1 | tee -a '$LOG'\""
        ok "started in tmux session '$SESSION' (attach: tmux attach -t $SESSION, detach: Ctrl-b d)"
    else
        nohup setsid bash -c "$run" >> "$LOG" 2>&1 < /dev/null &
        echo $! > "$DATA_DIR/galley.pid"
        ok "started in the background (pid $!; no tmux found)"
    fi
    for _ in $(seq 60); do health >/dev/null && break; sleep 0.5; done
    if ! health >/dev/null; then
        echo; tail -n 25 "$LOG" | sed 's/^/    /'
        die "Galley did not answer on $BASE within 30 s (log above, full log: $LOG)"
    fi
    local h; h=$(health)
    echo "$h" | grep -q '"env_script":true' || warn "server says figs_env.sh is missing"
    ok "answering: $h"
    how_to_open
}

case "$CMD" in
    start|restart)
        if [ "$PULL" = 1 ]; then
            step "git pull"
            git -C "$REPO" pull --ff-only || die "pull failed: resolve it by hand (never merge on a run machine)"
            git -C "$REPO" log --oneline -1
            [ "$CMD" = start ] && CMD=restart     # new code needs a fresh server
        fi
        if [ "$CMD" = restart ]; then
            [ -x "$PY" ] || ensure_venv
            read_profile
            do_stop
        fi
        do_start ;;
    fg)
        ensure_venv; read_profile
        health >/dev/null && die "already running in the background: ./run_ui.sh stop first"
        checks
        echo; echo "  serving on $BASE — Ctrl-C to stop"
        cd "$BACKEND"; exec "$PY" -m galley ;;
    stop)
        [ -x "$PY" ] || { echo "  not installed yet"; exit 0; }
        read_profile; do_stop ;;
    status)
        [ -x "$PY" ] || { echo "  backend not installed yet: ./run_ui.sh"; exit 1; }
        read_profile
        echo "  profile: $PROFILE ($HOST:$PORT)"
        if h=$(health); then
            ok "up: $h"
            in_tmux && echo "  tmux session: $SESSION"
            job=$(running_job); [ -n "$job" ] && echo "  running job: #$job  ($BASE/#/jobs/$job)"
            how_to_open
        else
            warn "not running (start: ./run_ui.sh)"; exit 3
        fi ;;
    logs)
        [ -x "$PY" ] || die "backend not installed yet: ./run_ui.sh"
        read_profile
        [ -f "$LOG" ] || die "no log yet at $LOG"
        exec tail -n 50 -f "$LOG" ;;
esac
