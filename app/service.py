"""macOS launchd service management for mlx-speech-server."""
import fcntl
import importlib.metadata
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.sax.saxutils
from pathlib import Path

from app.config import DEFAULT_PORT, ServerConfig

STARTUP_TIMEOUT = 30.0

SERVICE_LABEL = "com.local.mlx-speech-server"
CONFIG_DIR = Path.home() / ".config/mlx-speech-server"
CONFIG_ENV = CONFIG_DIR / "config.env"
VENV_DIR = Path.home() / ".local/venvs/mlx-speech-server"
LOG_DIR = Path.home() / ".local/logs/mlx-speech-server"
PLIST_PATH = Path.home() / "Library/LaunchAgents" / f"{SERVICE_LABEL}.plist"
STDOUT_LOG = LOG_DIR / "server.log"
STDERR_LOG = LOG_DIR / "server.err"


def error_message(error: Exception) -> str:
    """Bound subprocess diagnostics without disclosing its command or credentials."""
    def sanitize(value: str | bytes | None) -> str:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value or "")
        value = re.sub(r"https?://\S+", "[redacted URL]", value)
        value = re.sub(r"(?i)(authorization\s*[:=]\s*)[^\r\n]+", r"\1[redacted]", value)
        value = re.sub(r"(?i)([\w-]*(?:token|password|secret|api[_-]?key)[\w-]*\s*[:=]\s*)[^\s]+",
                       r"\1[redacted]", value)
        value = "".join(c for c in value if c in "\n\t" or c.isprintable())
        return value[:1000].strip()

    if isinstance(error, subprocess.CalledProcessError):
        parts = [f"Subprocess failed with exit status {error.returncode}."]
        for label, value in (("stderr", error.stderr), ("stdout", error.stdout)):
            detail = sanitize(value)
            if detail:
                parts.append(f"{label}: {detail}")
        return "\n".join(parts)
    return sanitize(str(error))


def _connect_host(env: dict[str, str]) -> str:
    host = env.get("WHISPER_HOST", "127.0.0.1").strip("[]")
    return {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)


def _require_darwin() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("mlx-speech-server service management requires macOS.")


def is_installed() -> bool:
    return PLIST_PATH.exists()


def _check_installed() -> None:
    if not is_installed():
        raise RuntimeError("Service not installed. Run: mlx-speech-server install")


def _is_local_install() -> bool:
    """Return True if the package was installed from a local directory (file:// URL).

    Covers both editable (pip install -e .) and non-editable (pipx install .) installs.
    """
    try:
        dist = importlib.metadata.distribution("mlx-speech-server")
        raw = dist.read_text("direct_url.json")
        if raw:
            data = json.loads(raw)
            return data.get("url", "").startswith("file://")
    except Exception:
        pass
    return False


def _get_project_dir() -> Path:
    """Return the project source directory for a local install.

    Raises RuntimeError if not a local install.
    """
    try:
        dist = importlib.metadata.distribution("mlx-speech-server")
        raw = dist.read_text("direct_url.json")
        if raw:
            data = json.loads(raw)
            url = data.get("url", "")
            if url.startswith("file://"):
                return Path(urllib.parse.unquote(urllib.parse.urlparse(url).path))
    except Exception:
        pass
    raise RuntimeError(
        "Not a local install. Cannot determine project directory."
    )


def _get_install_args() -> list[str]:
    """Return pip install arguments for installing into the service venv."""
    if _is_local_install():
        source = _get_project_dir()
        if source.suffix == ".whl":
            return [str(source)]
        return ["-e", str(source)]
    return ["mlx-speech-server"]


def _read_env() -> dict[str, str]:
    """Parse ~/.config/mlx-speech-server/config.env into a dict."""
    if not CONFIG_ENV.exists():
        return {}
    result: dict[str, str] = {}
    for line in CONFIG_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _build_plist(env_vars: dict[str, str], runner: Path | None = None) -> str:
    """Generate launchd plist XML content."""
    env_entries = "".join(
        f"        <key>{xml.sax.saxutils.escape(k)}</key>\n        <string>{xml.sax.saxutils.escape(v)}</string>\n"
        for k, v in env_vars.items()
    )
    runner = runner or VENV_DIR / "bin/mlx-run"
    path = f"{runner.parent}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{SERVICE_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{xml.sax.saxutils.escape(str(runner))}</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{xml.sax.saxutils.escape(path)}</string>
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


def install(*, model: str | None = None, port: int | None = None,
            use_current_runtime: bool = False) -> None:
    """Register a service; optional first-config values never replace existing config.

    A provisioned runtime can register its own mlx-run without pip or a second venv.
    This is explicit so existing CLI installation behavior remains available.
    """
    _require_darwin()
    runner = None
    if use_current_runtime and is_installed():
        return
    if use_current_runtime:
        runner = Path(sys.executable).absolute().parent / "mlx-run"
        if not runner.is_file() or not os.access(runner, os.X_OK):
            raise RuntimeError(f"Current runtime has no executable mlx-run: {runner}")
    if not CONFIG_ENV.exists():
        _validate_first_config(model, port)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not use_current_runtime:
        if not (VENV_DIR / "bin/python").exists():
            subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True,
                           capture_output=True, text=True)
        subprocess.run(
            [str(VENV_DIR / "bin/pip"), "install"] + _get_install_args(),
            check=True, capture_output=True, text=True,
        )

    # Exclusive create preserves even an empty pre-existing config byte for byte.
    try:
        with CONFIG_ENV.open("x") as config:
            config.write("WHISPER_HOST=127.0.0.1\n")
            config.write(f"WHISPER_PORT={port if port is not None else DEFAULT_PORT}\n")
            config.write(f"WHISPER_MODEL_PATH={model or ServerConfig.model_path}\n")
    except FileExistsError:
        pass
    env_vars = _read_env()
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(_build_plist(env_vars, runner))


def _validate_first_config(model: str | None, port: int | None) -> None:
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    if model is not None and (not model.strip() or any(c in model for c in "\n\r\x00")):
        raise ValueError("Model must be a non-empty single-line identifier")


def _port_in_use(port: int) -> bool:
    """Detect a TCP listener even if it does not implement HTTP or /health."""
    host = _connect_host(_read_env())
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except ConnectionRefusedError:
        return False


def ensure_running(*, model: str | None = None, port: int | None = None,
                   use_current_runtime: bool = False) -> dict:
    """Start a missing service under a user-wide lock, never restart a process.

    Health describes HTTP availability only; model preparation is a separate API.
    First-install options are ignored when a config or service already exists.
    """
    result = {"running": False, "pid": None, "port": DEFAULT_PORT,
              "health": None, "service_label": SERVICE_LABEL}
    def fail(code: str, message: str) -> dict:
        return {**result, "error": {"code": code, "message": message}}

    try:
        _require_darwin()
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        # Lock-file inode persists; unlinking it would permit two independent locks.
        with (CONFIG_DIR / "ensure-running.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            env = _read_env()
            configured_port = int(env.get("WHISPER_PORT", DEFAULT_PORT))
            current = get_status()
            if not current.get("installed") and not CONFIG_ENV.exists():
                _validate_first_config(model, port)
                configured_port = port if port is not None else DEFAULT_PORT
            result["port"] = configured_port
            for key in ("running", "pid", "port", "health"):
                if key in current:
                    result[key] = current[key]
            if result["running"]:
                if not isinstance(result["health"], dict) or result["health"].get("status") != "ok":
                    return fail("unhealthy_service", "Existing service process is not healthy; left running.")
                return result
            if _port_in_use(configured_port):
                return fail("port_in_use", f"Another listener owns port {configured_port}; left untouched.")
            if not current.get("installed"):
                kwargs = {"model": model, "port": port}
                if use_current_runtime:
                    kwargs["use_current_runtime"] = True
                try:
                    install(**kwargs)
                except Exception as exc:
                    return fail("install_failed", error_message(exc))
                if _port_in_use(configured_port):
                    return fail("port_in_use", f"Another listener owns port {configured_port}; left untouched.")
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            target = f"gui/{os.getuid()}"
            if current.get("loaded"):
                # No -k: if another actor just started it, leave that process alone.
                command = ["launchctl", "kickstart", f"{target}/{SERVICE_LABEL}"]
            else:
                command = ["launchctl", "bootstrap", target, str(PLIST_PATH)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            deadline = time.monotonic() + STARTUP_TIMEOUT
            while True:
                current = get_status()
                for key in ("running", "pid", "port", "health"):
                    if key in current:
                        result[key] = current[key]
                if (result["running"] and isinstance(result["health"], dict)
                        and result["health"].get("status") == "ok"):
                    return result
                if time.monotonic() >= deadline:
                    return fail("startup_timeout", "Service did not become healthy before the startup deadline.")
                time.sleep(0.1)
    except Exception as exc:
        return fail("startup_failed", error_message(exc))


def _launchctl_bootout() -> None:
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}/{SERVICE_LABEL}"],
        check=False,
        capture_output=True,
    )


def _launchctl_bootstrap() -> None:
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{uid}", str(PLIST_PATH)],
        check=False,
        capture_output=True,
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
    _launchctl_bootout()  # ensure clean state before bootstrap
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
    port = int(env_vars.get("WHISPER_PORT", DEFAULT_PORT))

    host = _connect_host(env_vars)
    url_host = f"[{host}]" if ":" in host else host
    health = None
    queue = None
    if pid:
        for url, key in [
            (f"http://{url_host}:{port}/health", "health"),
            (f"http://{url_host}:{port}/v1/queue/stats", "queue"),
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
    Update the service to the latest version.

    Editable install (local clone): git pull origin main, then pip install -e . if new commits.
    PyPI install: pip install --upgrade mlx-speech-server.

    Returns {"status": "up_to_date"} or {"status": "upgraded"}.
    Raises subprocess.CalledProcessError on failure.
    """
    _require_darwin()

    if _is_local_install():
        project_dir = _get_project_dir()
        result = subprocess.run(
            ["git", "-C", str(project_dir), "pull", "origin", "main"],
            check=True,
            capture_output=True,
            text=True,
        )
        if "Already up to date." in result.stdout:
            return {"status": "up_to_date"}
        subprocess.run(
            [str(VENV_DIR / "bin/pip"), "install", "-e", str(project_dir)],
            check=True,
        )
        return {"status": "upgraded"}

    result = subprocess.run(
        [str(VENV_DIR / "bin/pip"), "install", "--upgrade", "mlx-speech-server"],
        check=True,
        capture_output=True,
        text=True,
    )
    if "Requirement already satisfied" in result.stdout:
        return {"status": "up_to_date"}
    return {"status": "upgraded"}
