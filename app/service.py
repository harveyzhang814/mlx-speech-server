"""macOS launchd service management for mlx-speech-server."""
import json
import os
import subprocess
import sys
import urllib.request
import xml.sax.saxutils
from pathlib import Path

SERVICE_LABEL = "com.local.mlx-speech-server"
PROJECT_ROOT = Path(__file__).parent.parent
VENV_DIR = Path.home() / ".local/venvs/mlx-speech-server"
LOG_DIR = Path.home() / ".local/logs/mlx-speech-server"
PLIST_PATH = Path.home() / "Library/LaunchAgents" / f"{SERVICE_LABEL}.plist"
STDOUT_LOG = LOG_DIR / "server.log"
STDERR_LOG = LOG_DIR / "server.err"


def _require_darwin() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("mlx-speech-server service management requires macOS.")


def is_installed() -> bool:
    return PLIST_PATH.exists()


def _check_installed() -> None:
    if not is_installed():
        raise RuntimeError("Service not installed. Run: mlx-speech-server install")


def _read_env() -> dict[str, str]:
    """Parse .env file from project root into a dict."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return {}
    result: dict[str, str] = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _build_plist(env_vars: dict[str, str]) -> str:
    """Generate launchd plist XML content."""
    env_entries = "".join(
        f"        <key>{xml.sax.saxutils.escape(k)}</key>\n        <string>{xml.sax.saxutils.escape(v)}</string>\n"
        for k, v in env_vars.items()
    )
    python = VENV_DIR / "bin/python"
    main = PROJECT_ROOT / "main.py"
    path = f"{VENV_DIR}/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{SERVICE_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>{main}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{PROJECT_ROOT}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{path}</string>
{env_entries}    </dict>
    <key>StandardOutPath</key>
    <string>{STDOUT_LOG}</string>
    <key>StandardErrorPath</key>
    <string>{STDERR_LOG}</string>
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
</plist>"""


def install() -> None:
    """Create service venv, install package, register launchd plist."""
    _require_darwin()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not (VENV_DIR / "bin/python").exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)

    subprocess.run(
        [str(VENV_DIR / "bin/pip"), "install", "-e", str(PROJECT_ROOT)],
        check=True,
    )

    env_vars = _read_env()
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(_build_plist(env_vars))


def _launchctl_bootout() -> None:
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}/{SERVICE_LABEL}"],
        check=False,
    )


def _launchctl_bootstrap() -> None:
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{uid}", str(PLIST_PATH)],
        check=False,
    )


def _launchctl_kickstart() -> None:
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "kickstart", "-k", f"gui/{uid}/{SERVICE_LABEL}"],
        check=True,
    )


def stop() -> None:
    """Stop the running service."""
    _require_darwin()
    _check_installed()
    _launchctl_bootout()


def uninstall() -> None:
    """Stop service and remove plist. Venv is kept."""
    _require_darwin()
    _check_installed()
    _launchctl_bootout()
    PLIST_PATH.unlink(missing_ok=True)


def start() -> None:
    """Start service, auto-installing if not yet installed."""
    _require_darwin()
    if not is_installed():
        install()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _launchctl_bootstrap()
    _launchctl_kickstart()


def restart() -> None:
    """Stop then start the service."""
    _require_darwin()
    _check_installed()
    _launchctl_bootout()
    _launchctl_bootstrap()
    _launchctl_kickstart()


def get_status() -> dict:
    """Return a dict with service status information."""
    _require_darwin()
    if not is_installed():
        return {"installed": False}

    uid = os.getuid()
    result = subprocess.run(
        ["launchctl", "print", f"gui/{uid}/{SERVICE_LABEL}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"installed": True, "loaded": False, "running": False, "pid": None}

    pid: int | None = None
    for line in result.stdout.splitlines():
        if "pid =" in line:
            try:
                pid = int(line.split("=")[1].strip())
            except ValueError:
                pass
            break

    env_vars = _read_env()
    port = int(env_vars.get("WHISPER_PORT", 8000))

    health = None
    queue = None
    if pid:
        for url, key in [
            (f"http://localhost:{port}/health", "health"),
            (f"http://localhost:{port}/v1/queue/stats", "queue"),
        ]:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    value = json.loads(resp.read())
                if key == "health":
                    health = value
                else:
                    queue = value
            except Exception:
                pass

    return {
        "installed": True,
        "loaded": True,
        "running": bool(pid),
        "pid": pid,
        "port": port,
        "health": health,
        "queue": queue,
        "plist_path": PLIST_PATH,
        "venv_dir": VENV_DIR,
        "log_dir": LOG_DIR,
    }


def get_logs() -> tuple[str, str]:
    """Return (stdout_content, stderr_content), last 30 lines each."""
    def tail30(path: Path) -> str:
        if not path.exists():
            return "(empty)"
        lines = path.read_text().splitlines()
        return "\n".join(lines[-30:]) if lines else "(empty)"

    return tail30(STDOUT_LOG), tail30(STDERR_LOG)


def upgrade() -> dict[str, str]:
    """
    Pull latest code and reinstall if there are new commits.

    Returns {"status": "up_to_date"} or {"status": "upgraded"}.
    Raises subprocess.CalledProcessError on git failure.
    """
    _require_darwin()
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "pull", "origin", "main"],
        check=True,
        capture_output=True,
        text=True,
    )
    if "Already up to date." in result.stdout:
        return {"status": "up_to_date"}

    subprocess.run(
        [str(VENV_DIR / "bin/pip"), "install", "-e", str(PROJECT_ROOT)],
        check=True,
    )
    return {"status": "upgraded"}
