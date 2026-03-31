#!/bin/bash
# mlx-speech-server launchd service manager
# Usage: ./scripts/service.sh [install|uninstall|start|stop|restart|status|logs]

set -euo pipefail

SERVICE_LABEL="com.local.mlx-speech-server"
PLIST_PATH="$HOME/Library/LaunchAgents/${SERVICE_LABEL}.plist"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.local/venvs/mlx-speech-server}"
PYTHON="${VENV_DIR}/bin/python"
MAIN="${PROJECT_DIR}/main.py"
LOG_DIR="${HOME}/.local/logs/mlx-speech-server"
STDOUT_LOG="${LOG_DIR}/server.log"
STDERR_LOG="${LOG_DIR}/server.err"

# --- helpers ---

ensure_logs_dir() {
    mkdir -p "$LOG_DIR"
}

check_installed() {
    if [ ! -f "$PLIST_PATH" ]; then
        echo "❌ Service not installed. Run: $0 install"
        exit 1
    fi
}

step() {
    echo ""
    echo "▶ $*"
}

cmd_setup_venv() {
    step "Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "   ✔ venv created ($(python3 --version))"

    step "Upgrading pip ..."
    "${VENV_DIR}/bin/pip" install --upgrade pip -q
    echo "   ✔ pip upgraded"

    step "Installing mlx-speech-server and dependencies ..."
    "${VENV_DIR}/bin/pip" install -e "${PROJECT_DIR}"
    echo "   ✔ Dependencies installed"
}

# --- commands ---

cmd_install() {
    echo "════════════════════════════════════════"
    echo "  mlx-speech-server — Install"
    echo "════════════════════════════════════════"

    # Create venv if it doesn't exist
    if [ ! -x "$PYTHON" ]; then
        cmd_setup_venv
    else
        echo ""
        echo "ℹ️  Virtual environment already exists at $VENV_DIR"
        echo "   Run '$0 upgrade' to reinstall dependencies."
    fi

    ensure_logs_dir

    # Build environment variables from .env file if it exists
    local env_vars=""
    if [ -f "${PROJECT_DIR}/.env" ]; then
        step "Loading environment from .env ..."
        while IFS='=' read -r key value; do
            [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
            value="${value%\"}"
            value="${value#\"}"
            env_vars+="        <key>${key}</key>\n        <string>${value}</string>\n"
            echo "   ✔ $key"
        done < "${PROJECT_DIR}/.env"
    else
        echo ""
        echo "ℹ️  No .env file found — using defaults"
    fi

    step "Writing launchd plist ..."
    cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${SERVICE_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON}</string>
        <string>${MAIN}</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>${VENV_DIR}/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
$(printf "%b" "$env_vars")    </dict>

    <key>StandardOutPath</key>
    <string>${STDOUT_LOG}</string>

    <key>StandardErrorPath</key>
    <string>${STDERR_LOG}</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>ThrottleInterval</key>
    <integer>5</integer>

    <key>ProcessType</key>
    <string>Standard</string>
</dict>
</plist>
PLIST
    echo "   ✔ $PLIST_PATH"

    echo ""
    echo "════════════════════════════════════════"
    echo "  ✅ Installation complete"
    echo "════════════════════════════════════════"
    echo "  Project:  $PROJECT_DIR"
    echo "  Venv:     $VENV_DIR"
    echo "  Logs:     $LOG_DIR"
    echo "  Plist:    $PLIST_PATH"
    echo ""
    echo "  Next steps:"
    echo "    $0 start    — start the service"
    echo "    $0 status   — check health"
    echo ""
    echo "  After starting, API docs available at:"
    echo "    Swagger UI → http://localhost:8000/docs"
    echo "    ReDoc      → http://localhost:8000/redoc"
    echo ""
    echo "  To reconfigure: edit .env then run '$0 install' again."
    echo "════════════════════════════════════════"
}

cmd_uninstall() {
    echo "════════════════════════════════════════"
    echo "  mlx-speech-server — Uninstall"
    echo "════════════════════════════════════════"
    if [ -f "$PLIST_PATH" ]; then
        step "Stopping service ..."
        launchctl bootout "gui/$(id -u)/${SERVICE_LABEL}" 2>/dev/null && echo "   ✔ Stopped" || echo "   ℹ️  Was not running"

        step "Removing plist ..."
        rm -f "$PLIST_PATH"
        echo "   ✔ $PLIST_PATH removed"

        echo ""
        echo "════════════════════════════════════════"
        echo "  ✅ Service uninstalled"
        echo "════════════════════════════════════════"
        echo "  Virtual environment kept at:"
        echo "    $VENV_DIR"
        echo "  To remove it: rm -rf $VENV_DIR"
        echo "════════════════════════════════════════"
    else
        echo "  ℹ️  Service not installed, nothing to do"
    fi
}

cmd_upgrade() {
    echo "════════════════════════════════════════"
    echo "  mlx-speech-server — Upgrade"
    echo "════════════════════════════════════════"
    step "Reinstalling dependencies from $PROJECT_DIR ..."
    "${VENV_DIR}/bin/pip" install -e "${PROJECT_DIR}"
    echo ""
    echo "  ✅ Dependencies updated."
    echo "  Run '$0 restart' to apply changes."
}

cmd_start() {
    check_installed
    ensure_logs_dir
    echo "▶ Starting service ..."
    launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH" 2>/dev/null || true
    launchctl kickstart -k "gui/$(id -u)/${SERVICE_LABEL}" 2>/dev/null || true
    sleep 1
    echo ""
    cmd_status
}

cmd_stop() {
    check_installed
    echo "▶ Stopping service ..."
    launchctl bootout "gui/$(id -u)/${SERVICE_LABEL}" 2>/dev/null || true
    echo "⏹  Service stopped"
}

cmd_restart() {
    check_installed
    echo "▶ Restarting service ..."
    launchctl bootout "gui/$(id -u)/${SERVICE_LABEL}" 2>/dev/null || true
    echo "   ✔ Stopped"
    sleep 1
    launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH" 2>/dev/null || true
    echo "   ✔ Started"
    sleep 1
    echo ""
    cmd_status
}

cmd_status() {
    echo "════════════════════════════════════════"
    echo "  mlx-speech-server — Status"
    echo "════════════════════════════════════════"

    if ! [ -f "$PLIST_PATH" ]; then
        echo "  ⚪ Not installed"
        echo "════════════════════════════════════════"
        return
    fi
    echo "  Plist:    $PLIST_PATH"
    echo "  Venv:     $VENV_DIR"
    echo "  Project:  $PROJECT_DIR"
    echo "  Logs:     $LOG_DIR"

    local info
    info=$(launchctl print "gui/$(id -u)/${SERVICE_LABEL}" 2>/dev/null) || {
        echo ""
        echo "  ⚪ Installed but not loaded"
        echo "════════════════════════════════════════"
        return
    }

    local pid
    pid=$(echo "$info" | grep "pid =" | awk '{print $3}')
    echo ""
    if [ -n "$pid" ] && [ "$pid" != "0" ]; then
        echo "  Process:  🟢 Running (PID $pid)"

        local port=8000
        if [ -f "${PROJECT_DIR}/.env" ]; then
            local env_port
            env_port=$(grep '^WHISPER_PORT=' "${PROJECT_DIR}/.env" 2>/dev/null | cut -d= -f2 | tr -d '"')
            [ -n "$env_port" ] && port="$env_port"
        fi
        echo "  Port:     $port"

        local health_resp
        health_resp=$(curl -s --max-time 2 "http://localhost:${port}/health" 2>/dev/null || true)
        if [ -n "$health_resp" ]; then
            echo "  Health:   🟢 OK — $health_resp"
        else
            echo "  Health:   🟡 Not responding yet (model may still be loading)"
        fi

        local queue_resp
        queue_resp=$(curl -s --max-time 2 "http://localhost:${port}/v1/queue/stats" 2>/dev/null || true)
        if [ -n "$queue_resp" ]; then
            echo "  Queue:    $queue_resp"
        fi
        echo ""
        echo "  API Docs:"
        echo "    Swagger UI → http://localhost:${port}/docs"
        echo "    ReDoc      → http://localhost:${port}/redoc"
    else
        echo "  Process:  🔴 Not running"
        echo "  Run 'launchctl print gui/$(id -u)/${SERVICE_LABEL}' for exit details"
    fi
    echo "════════════════════════════════════════"
}

cmd_logs() {
    ensure_logs_dir
    echo "════════════════════════════════════════"
    echo "  mlx-speech-server — Logs"
    echo "════════════════════════════════════════"
    echo "  stdout: $STDOUT_LOG"
    echo "  stderr: $STDERR_LOG"
    echo ""
    echo "─── stdout (last 30 lines) ─────────────"
    tail -30 "$STDOUT_LOG" 2>/dev/null || echo "(empty)"
    echo ""
    echo "─── stderr (last 30 lines) ─────────────"
    tail -30 "$STDERR_LOG" 2>/dev/null || echo "(empty)"
    echo ""
    echo "════════════════════════════════════════"
    echo "  💡 Live tail:"
    echo "     tail -f $STDOUT_LOG $STDERR_LOG"
    echo "════════════════════════════════════════"
}

# --- main ---

case "${1:-help}" in
    install)   cmd_install ;;
    uninstall) cmd_uninstall ;;
    upgrade)   cmd_upgrade ;;
    start)     cmd_start ;;
    stop)      cmd_stop ;;
    restart)   cmd_restart ;;
    status)    cmd_status ;;
    logs)      cmd_logs ;;
    *)
        echo "════════════════════════════════════════"
        echo "  mlx-speech-server service manager"
        echo "════════════════════════════════════════"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  install     Create venv, install deps, register launchd service"
        echo "  uninstall   Stop and remove launchd service (venv kept)"
        echo "  upgrade     Reinstall Python dependencies from source"
        echo "  start       Start the service"
        echo "  stop        Stop the service"
        echo "  restart     Restart the service"
        echo "  status      Show process status, health check, and queue stats"
        echo "  logs        Show recent stdout/stderr log output"
        echo ""
        echo "Paths:"
        echo "  Project:  $PROJECT_DIR"
        echo "  Venv:     $VENV_DIR"
        echo "  Logs:     $LOG_DIR"
        echo ""
        echo "Configuration:"
        echo "  Create a .env file in project root with WHISPER_* variables."
        echo "  Then run '$0 install' to apply."
        echo ""
        echo "  Override venv path:  VENV_DIR=/custom/path $0 install"
        echo "════════════════════════════════════════"
        ;;
esac
