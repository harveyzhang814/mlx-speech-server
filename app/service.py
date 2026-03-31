"""macOS launchd service management for mlx-speech-server."""
import subprocess
import sys
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
