#!/bin/bash
# mlx-whisper-server launchd service manager
# Usage: ./scripts/service.sh [install|uninstall|start|stop|restart|status|logs]

set -euo pipefail

SERVICE_LABEL="com.local.mlx-whisper-server"
PLIST_PATH="$HOME/Library/LaunchAgents/${SERVICE_LABEL}.plist"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PROJECT_DIR}/venv/bin/python"
MAIN="${PROJECT_DIR}/main.py"
LOG_DIR="${PROJECT_DIR}/logs"
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

# --- commands ---

cmd_install() {
    if [ ! -x "$PYTHON" ]; then
        echo "❌ Python not found at $PYTHON"
        echo "   Run: python3 -m venv venv && source venv/bin/activate && pip install -e '.[dev]'"
        exit 1
    fi

    ensure_logs_dir

    # Build environment variables from .env file if it exists
    local env_vars=""
    if [ -f "${PROJECT_DIR}/.env" ]; then
        echo "📄 Loading environment from .env"
        while IFS='=' read -r key value; do
            [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
            value="${value%\"}"
            value="${value#\"}"
            env_vars+="        <key>${key}</key>\n        <string>${value}</string>\n"
        done < "${PROJECT_DIR}/.env"
    fi

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
        <string>${PROJECT_DIR}/venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
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

    echo "✅ Service installed: $PLIST_PATH"
    echo "   Logs: $LOG_DIR/"
    echo ""
    echo "   To start:   $0 start"
    echo "   To configure, edit .env in project root then reinstall."
}

cmd_uninstall() {
    if [ -f "$PLIST_PATH" ]; then
        # Stop first if running
        launchctl bootout "gui/$(id -u)/${SERVICE_LABEL}" 2>/dev/null || true
        rm -f "$PLIST_PATH"
        echo "✅ Service uninstalled"
    else
        echo "ℹ️  Service not installed, nothing to do"
    fi
}

cmd_start() {
    check_installed
    ensure_logs_dir
    launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH" 2>/dev/null || true
    launchctl kickstart -k "gui/$(id -u)/${SERVICE_LABEL}" 2>/dev/null || true
    sleep 1
    cmd_status
}

cmd_stop() {
    check_installed
    launchctl bootout "gui/$(id -u)/${SERVICE_LABEL}" 2>/dev/null || true
    echo "⏹  Service stopped"
}

cmd_restart() {
    echo "🔄 Restarting service..."
    cmd_stop
    sleep 1
    # Re-bootstrap since bootout removes it
    launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH" 2>/dev/null || true
    sleep 1
    cmd_status
}

cmd_status() {
    if ! [ -f "$PLIST_PATH" ]; then
        echo "⚪ Not installed"
        return
    fi

    local info
    info=$(launchctl print "gui/$(id -u)/${SERVICE_LABEL}" 2>/dev/null) || {
        echo "⚪ Installed but not loaded"
        return
    }

    local pid
    pid=$(echo "$info" | grep "pid =" | awk '{print $3}')
    if [ -n "$pid" ] && [ "$pid" != "0" ]; then
        echo "🟢 Running (PID: $pid)"
        # Determine port from .env or default
        local port=8000
        if [ -f "${PROJECT_DIR}/.env" ]; then
            local env_port
            env_port=$(grep '^WHISPER_PORT=' "${PROJECT_DIR}/.env" 2>/dev/null | cut -d= -f2 | tr -d '"')
            [ -n "$env_port" ] && port="$env_port"
        fi
        if curl -s --max-time 2 "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "   Health: OK (port $port)"
        else
            echo "   Health: not responding yet (port $port)"
        fi
    else
        echo "🔴 Not running (last exit status in: launchctl print gui/$(id -u)/${SERVICE_LABEL})"
    fi
}

cmd_logs() {
    ensure_logs_dir
    echo "=== stdout (last 30 lines) ==="
    tail -30 "$STDOUT_LOG" 2>/dev/null || echo "(empty)"
    echo ""
    echo "=== stderr (last 30 lines) ==="
    tail -30 "$STDERR_LOG" 2>/dev/null || echo "(empty)"
    echo ""
    echo "💡 Live tail: tail -f $STDOUT_LOG $STDERR_LOG"
}

# --- main ---

case "${1:-help}" in
    install)   cmd_install ;;
    uninstall) cmd_uninstall ;;
    start)     cmd_start ;;
    stop)      cmd_stop ;;
    restart)   cmd_restart ;;
    status)    cmd_status ;;
    logs)      cmd_logs ;;
    *)
        echo "mlx-whisper-server service manager"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  install     Install launchd service (auto-start on login)"
        echo "  uninstall   Remove launchd service"
        echo "  start       Start the service"
        echo "  stop        Stop the service"
        echo "  restart     Restart the service"
        echo "  status      Show service status + health check"
        echo "  logs        Show recent log output"
        echo ""
        echo "Configuration:"
        echo "  Create a .env file in project root with WHISPER_* variables."
        echo "  Then run '$0 install' to apply."
        ;;
esac
