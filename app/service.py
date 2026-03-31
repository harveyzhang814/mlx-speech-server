"""macOS launchd service management for mlx-speech-server."""
import sys
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
