# CLI Subcommands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `mlx-speech-server install/uninstall/upgrade/start/stop/restart/status/logs` subcommands, replacing `scripts/service.sh` as the primary service management interface.

**Architecture:** New `cli.py` at project root is the Click group entry point; `app/service.py` owns all macOS launchd/subprocess/venv logic; `main.py` and launchd plist execution are completely unchanged. `scripts/service.sh` is kept as a backup.

**Tech Stack:** Click (existing dependency), Python stdlib (subprocess, pathlib, urllib.request, os)

---

## File Map

| File | Change | Responsibility |
| :--- | :--- | :--- |
| `cli.py` | Create | Click group + all subcommand definitions, terminal output, `click.confirm` interactions |
| `app/service.py` | Create | launchd plist management, venv lifecycle, subprocess wrappers, status/log queries |
| `tests/test_service.py` | Create | Unit tests for `app/service.py` |
| `tests/test_cli.py` | Create | CLI integration tests via `CliRunner` |
| `pyproject.toml` | Modify | Entry point: `main:cli` → `cli:cli` |
| `README.md` | Modify | Update quick start to use `pipx install .` + `mlx-speech-server` commands |
| `README_zh.md` | Modify | Same updates in Chinese |

---

### Task 1: Update pyproject.toml entry point

**Files:**
- Modify: `pyproject.toml:21`

- [ ] **Step 1: Update the entry point**

In `pyproject.toml`, change line 21:
```toml
[project.scripts]
mlx-speech-server = "cli:cli"
```

- [ ] **Step 2: Reinstall to register the new entry point**

```bash
pip install -e ".[dev]"
```

Expected: Completes without error. `mlx-speech-server --help` will error (`No module named cli`) — that's fine, `cli.py` doesn't exist yet.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: update CLI entry point to cli:cli"
```

---

### Task 2: app/service.py — constants, is_installed, _check_installed

**Files:**
- Create: `app/service.py`
- Create: `tests/test_service.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_service.py`:

```python
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

if sys.platform != "darwin":
    pytest.skip("Service management tests require macOS", allow_module_level=True)

from app import service


@pytest.fixture
def fake_paths(tmp_path):
    plist = tmp_path / "LaunchAgents" / "com.local.mlx-speech-server.plist"
    with patch.multiple(
        service,
        PLIST_PATH=plist,
        VENV_DIR=tmp_path / "venv",
        LOG_DIR=tmp_path / "logs",
        PROJECT_ROOT=tmp_path / "project",
        STDOUT_LOG=tmp_path / "logs" / "server.log",
        STDERR_LOG=tmp_path / "logs" / "server.err",
    ):
        yield tmp_path


@pytest.fixture
def installed(fake_paths):
    plist = fake_paths / "LaunchAgents" / "com.local.mlx-speech-server.plist"
    plist.parent.mkdir(parents=True)
    plist.write_text("fake plist")
    return fake_paths


def test_is_installed_true(installed):
    assert service.is_installed() is True


def test_is_installed_false(fake_paths):
    assert service.is_installed() is False


def test_check_installed_raises_when_not_installed(fake_paths):
    with pytest.raises(RuntimeError, match="not installed"):
        service._check_installed()


def test_check_installed_passes_when_installed(installed):
    service._check_installed()  # Should not raise
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_service.py -v
```

Expected: `ImportError` — `app/service.py` does not exist yet.

- [ ] **Step 3: Create app/service.py**

```python
"""macOS launchd service management for mlx-speech-server."""
import os
import subprocess
import sys
import urllib.error
import urllib.request
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
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_service.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app/service.py tests/test_service.py
git commit -m "feat: add app/service.py with constants and is_installed"
```

---

### Task 3: app/service.py — _read_env and _build_plist

**Files:**
- Modify: `app/service.py`
- Modify: `tests/test_service.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_service.py`:

```python
def test_read_env_parses_key_value_pairs(fake_paths):
    env_file = fake_paths / "project" / ".env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("WHISPER_PORT=9000\nWHISPER_MODEL_PATH=mlx-community/whisper-large\n")
    result = service._read_env()
    assert result == {
        "WHISPER_PORT": "9000",
        "WHISPER_MODEL_PATH": "mlx-community/whisper-large",
    }


def test_read_env_skips_comments_and_blank_lines(fake_paths):
    env_file = fake_paths / "project" / ".env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("# comment\n\nWHISPER_PORT=8000\n")
    result = service._read_env()
    assert result == {"WHISPER_PORT": "8000"}


def test_read_env_strips_quotes(fake_paths):
    env_file = fake_paths / "project" / ".env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text('WHISPER_MODEL_PATH="mlx-community/whisper-turbo"\n')
    result = service._read_env()
    assert result == {"WHISPER_MODEL_PATH": "mlx-community/whisper-turbo"}


def test_read_env_returns_empty_when_no_file(fake_paths):
    result = service._read_env()
    assert result == {}


def test_build_plist_contains_service_label():
    plist = service._build_plist({})
    assert service.SERVICE_LABEL in plist


def test_build_plist_contains_venv_python():
    plist = service._build_plist({})
    assert str(service.VENV_DIR / "bin/python") in plist


def test_build_plist_injects_env_vars():
    plist = service._build_plist({"WHISPER_PORT": "9000"})
    assert "<key>WHISPER_PORT</key>" in plist
    assert "<string>9000</string>" in plist


def test_build_plist_excludes_missing_env_vars():
    plist = service._build_plist({})
    assert "WHISPER_PORT" not in plist
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_service.py -k "read_env or build_plist" -v
```

Expected: `AttributeError: module 'app.service' has no attribute '_read_env'`

- [ ] **Step 3: Add _read_env to app/service.py**

```python
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
```

- [ ] **Step 4: Add _build_plist to app/service.py**

```python
def _build_plist(env_vars: dict[str, str]) -> str:
    """Generate launchd plist XML content."""
    env_entries = "".join(
        f"        <key>{k}</key>\n        <string>{v}</string>\n"
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
```

- [ ] **Step 5: Run tests to confirm pass**

```bash
pytest tests/test_service.py -k "read_env or build_plist" -v
```

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add app/service.py tests/test_service.py
git commit -m "feat: add _read_env and _build_plist helpers"
```

---

### Task 4: app/service.py — install()

**Files:**
- Modify: `app/service.py`
- Modify: `tests/test_service.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_service.py`:

```python
def test_install_creates_venv_when_not_exists(fake_paths, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(subprocess, "run", fake_run)

    service.install()

    venv_calls = [c for c in calls if "-m" in c and "venv" in c]
    assert len(venv_calls) == 1


def test_install_skips_venv_when_python_exists(fake_paths, monkeypatch):
    python = fake_paths / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()

    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(subprocess, "run", fake_run)

    service.install()

    venv_calls = [c for c in calls if "-m" in c and "venv" in c]
    assert len(venv_calls) == 0


def test_install_runs_pip_install(fake_paths, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(subprocess, "run", fake_run)

    service.install()

    pip_calls = [c for c in calls if "pip" in str(c[0]) and "install" in c]
    assert len(pip_calls) == 1
    assert "-e" in pip_calls[0]


def test_install_writes_plist(fake_paths, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))

    service.install()

    plist = fake_paths / "LaunchAgents" / "com.local.mlx-speech-server.plist"
    assert plist.exists()
    assert service.SERVICE_LABEL in plist.read_text()


def test_install_is_idempotent(fake_paths, monkeypatch):
    """Running install twice should not raise."""
    python = fake_paths / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))

    service.install()
    service.install()  # Should not raise
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_service.py -k "install" -v
```

Expected: `AttributeError: module 'app.service' has no attribute 'install'`

- [ ] **Step 3: Add install() to app/service.py**

```python
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
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_service.py -k "install" -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add app/service.py tests/test_service.py
git commit -m "feat: add service.install()"
```

---

### Task 5: app/service.py — stop() and uninstall()

**Files:**
- Modify: `app/service.py`
- Modify: `tests/test_service.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_service.py`:

```python
def test_stop_calls_bootout(installed, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(subprocess, "run", fake_run)

    service.stop()

    assert any("bootout" in c for c in calls)


def test_stop_raises_when_not_installed(fake_paths):
    with pytest.raises(RuntimeError, match="not installed"):
        service.stop()


def test_uninstall_removes_plist(installed, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))
    plist = installed / "LaunchAgents" / "com.local.mlx-speech-server.plist"
    assert plist.exists()

    service.uninstall()

    assert not plist.exists()


def test_uninstall_calls_bootout(installed, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(subprocess, "run", fake_run)

    service.uninstall()

    assert any("bootout" in c for c in calls)


def test_uninstall_raises_when_not_installed(fake_paths):
    with pytest.raises(RuntimeError, match="not installed"):
        service.uninstall()
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_service.py -k "stop or uninstall" -v
```

Expected: `AttributeError: module 'app.service' has no attribute 'stop'`

- [ ] **Step 3: Add launchctl helpers + stop() + uninstall() to app/service.py**

```python
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
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_service.py -k "stop or uninstall" -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add app/service.py tests/test_service.py
git commit -m "feat: add service.stop() and service.uninstall()"
```

---

### Task 6: app/service.py — start()

**Files:**
- Modify: `app/service.py`
- Modify: `tests/test_service.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_service.py`:

```python
def test_start_auto_installs_when_not_installed(fake_paths, monkeypatch):
    install_calls = []
    monkeypatch.setattr(service, "install", lambda: install_calls.append(True))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))

    service.start()

    assert len(install_calls) == 1


def test_start_skips_install_when_already_installed(installed, monkeypatch):
    install_calls = []
    monkeypatch.setattr(service, "install", lambda: install_calls.append(True))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))

    service.start()

    assert len(install_calls) == 0


def test_start_calls_bootstrap_and_kickstart(installed, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(subprocess, "run", fake_run)

    service.start()

    assert any("bootstrap" in c for c in calls)
    assert any("kickstart" in c for c in calls)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_service.py -k "test_start" -v
```

Expected: `AttributeError: module 'app.service' has no attribute 'start'`

- [ ] **Step 3: Add start() to app/service.py**

```python
def start() -> None:
    """Start service, auto-installing if not yet installed."""
    _require_darwin()
    if not is_installed():
        install()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _launchctl_bootstrap()
    _launchctl_kickstart()
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_service.py -k "test_start" -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app/service.py tests/test_service.py
git commit -m "feat: add service.start() with auto-install"
```

---

### Task 7: app/service.py — restart()

**Files:**
- Modify: `app/service.py`
- Modify: `tests/test_service.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_service.py`:

```python
def test_restart_calls_bootout_before_bootstrap(installed, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(subprocess, "run", fake_run)

    service.restart()

    bootout_idx = next(i for i, c in enumerate(calls) if "bootout" in c)
    bootstrap_idx = next(i for i, c in enumerate(calls) if "bootstrap" in c)
    assert bootout_idx < bootstrap_idx
    assert any("kickstart" in c for c in calls)


def test_restart_raises_when_not_installed(fake_paths):
    with pytest.raises(RuntimeError, match="not installed"):
        service.restart()
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_service.py -k "restart" -v
```

Expected: `AttributeError: module 'app.service' has no attribute 'restart'`

- [ ] **Step 3: Add restart() to app/service.py**

```python
def restart() -> None:
    """Stop then start the service."""
    _require_darwin()
    _check_installed()
    _launchctl_bootout()
    _launchctl_bootstrap()
    _launchctl_kickstart()
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_service.py -k "restart" -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app/service.py tests/test_service.py
git commit -m "feat: add service.restart()"
```

---

### Task 8: app/service.py — upgrade()

**Files:**
- Modify: `app/service.py`
- Modify: `tests/test_service.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_service.py`:

```python
def test_upgrade_returns_up_to_date(fake_paths, monkeypatch):
    def fake_run(args, **kwargs):
        m = MagicMock()
        m.returncode = 0
        m.stdout = "Already up to date.\n"
        return m
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = service.upgrade()

    assert result == {"status": "up_to_date"}


def test_upgrade_installs_when_new_commits(fake_paths, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        m = MagicMock()
        m.returncode = 0
        m.stdout = "Fast-forward\n 1 file changed\n"
        return m
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = service.upgrade()

    assert result == {"status": "upgraded"}
    pip_calls = [c for c in calls if "pip" in str(c[0]) and "install" in c]
    assert len(pip_calls) == 1


def test_upgrade_skips_pip_when_up_to_date(fake_paths, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        m = MagicMock()
        m.returncode = 0
        m.stdout = "Already up to date.\n"
        return m
    monkeypatch.setattr(subprocess, "run", fake_run)

    service.upgrade()

    pip_calls = [c for c in calls if "pip" in str(c[0]) and "install" in c]
    assert len(pip_calls) == 0


def test_upgrade_raises_on_git_failure(fake_paths, monkeypatch):
    def raise_error(args, **kwargs):
        raise subprocess.CalledProcessError(1, "git", stderr="network error")
    monkeypatch.setattr(subprocess, "run", raise_error)

    with pytest.raises(subprocess.CalledProcessError):
        service.upgrade()
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_service.py -k "upgrade" -v
```

Expected: `AttributeError: module 'app.service' has no attribute 'upgrade'`

- [ ] **Step 3: Add upgrade() to app/service.py**

```python
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
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/test_service.py -k "upgrade" -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app/service.py tests/test_service.py
git commit -m "feat: add service.upgrade()"
```

---

### Task 9: app/service.py — get_status() and get_logs()

**Files:**
- Modify: `app/service.py`
- Modify: `tests/test_service.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_service.py`:

```python
def test_get_status_not_installed(fake_paths):
    result = service.get_status()
    assert result == {"installed": False}


def test_get_status_installed_but_not_loaded(installed, monkeypatch):
    def fake_run(args, **kwargs):
        m = MagicMock()
        m.returncode = 1
        m.stdout = ""
        return m
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = service.get_status()

    assert result["installed"] is True
    assert result["loaded"] is False
    assert result["running"] is False


def test_get_status_parses_pid(installed, monkeypatch):
    def fake_run(args, **kwargs):
        m = MagicMock()
        m.returncode = 0
        m.stdout = "...\n    pid = 12345\n..."
        return m
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(service, "_read_env", lambda: {})

    with patch("urllib.request.urlopen", side_effect=OSError("no server")):
        result = service.get_status()

    assert result["running"] is True
    assert result["pid"] == 12345
    assert result["health"] is None


def test_get_status_includes_health_and_queue(installed, monkeypatch):
    import json as json_mod

    def fake_run(args, **kwargs):
        m = MagicMock()
        m.returncode = 0
        m.stdout = "    pid = 99\n"
        return m
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(service, "_read_env", lambda: {})

    health_cm = MagicMock()
    health_cm.__enter__ = lambda s: s
    health_cm.__exit__ = MagicMock(return_value=False)
    health_cm.read.return_value = json_mod.dumps({"status": "ok"}).encode()

    queue_cm = MagicMock()
    queue_cm.__enter__ = lambda s: s
    queue_cm.__exit__ = MagicMock(return_value=False)
    queue_cm.read.return_value = json_mod.dumps({"queue_size": 0}).encode()

    responses = iter([health_cm, queue_cm])
    with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: next(responses)):
        result = service.get_status()

    assert result["health"] == {"status": "ok"}
    assert result["queue"] == {"queue_size": 0}


def test_get_logs_returns_last_30_lines(fake_paths):
    logs_dir = fake_paths / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = logs_dir / "server.log"
    stderr_log = logs_dir / "server.err"
    stdout_log.write_text("\n".join(f"line {i}" for i in range(50)))
    stderr_log.write_text("err1\nerr2\n")

    with patch.multiple(service, STDOUT_LOG=stdout_log, STDERR_LOG=stderr_log):
        out, err = service.get_logs()

    assert len(out.splitlines()) == 30
    assert "line 49" in out
    assert "line 19" not in out
    assert "err1" in err


def test_get_logs_returns_empty_when_no_files(fake_paths):
    stdout_log = fake_paths / "logs" / "server.log"
    stderr_log = fake_paths / "logs" / "server.err"

    with patch.multiple(service, STDOUT_LOG=stdout_log, STDERR_LOG=stderr_log):
        out, err = service.get_logs()

    assert out == "(empty)"
    assert err == "(empty)"
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_service.py -k "get_status or get_logs" -v
```

Expected: `AttributeError: module 'app.service' has no attribute 'get_status'`

- [ ] **Step 3: Add get_status() to app/service.py**

```python
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

    import json
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
```

- [ ] **Step 4: Add get_logs() to app/service.py**

```python
def get_logs() -> tuple[str, str]:
    """Return (stdout_content, stderr_content), last 30 lines each."""
    def tail30(path: Path) -> str:
        if not path.exists():
            return "(empty)"
        lines = path.read_text().splitlines()
        return "\n".join(lines[-30:]) if lines else "(empty)"

    return tail30(STDOUT_LOG), tail30(STDERR_LOG)
```

- [ ] **Step 5: Run all service tests**

```bash
pytest tests/test_service.py -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/service.py tests/test_service.py
git commit -m "feat: add service.get_status() and service.get_logs()"
```

---

### Task 10: Create cli.py and tests/test_cli.py

**Files:**
- Create: `cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_cli.py`:

```python
import subprocess
import sys
import pytest
from click.testing import CliRunner
from unittest.mock import MagicMock, patch

if sys.platform != "darwin":
    pytest.skip("CLI service tests require macOS", allow_module_level=True)

from cli import cli
from app import service


def test_install_success():
    runner = CliRunner()
    with patch("cli.service.install"):
        result = runner.invoke(cli, ["install"])
    assert result.exit_code == 0
    assert "Installation complete" in result.output


def test_install_failure_shows_error():
    runner = CliRunner()
    with patch("cli.service.install", side_effect=Exception("pip failed")):
        result = runner.invoke(cli, ["install"])
    assert result.exit_code == 1
    assert "pip failed" in result.output


def test_uninstall_success():
    runner = CliRunner()
    with patch("cli.service.uninstall"):
        result = runner.invoke(cli, ["uninstall"])
    assert result.exit_code == 0
    assert "uninstalled" in result.output.lower()


def test_uninstall_not_installed_shows_error():
    runner = CliRunner()
    with patch("cli.service.uninstall", side_effect=RuntimeError("Service not installed. Run: mlx-speech-server install")):
        result = runner.invoke(cli, ["uninstall"])
    assert result.exit_code == 1
    assert "not installed" in result.output.lower()


def test_start_invokes_service_start():
    runner = CliRunner()
    with patch("cli.service.start"), \
         patch("cli.service.get_status", return_value={"installed": False}):
        result = runner.invoke(cli, ["start"])
    assert result.exit_code == 0


def test_stop_success():
    runner = CliRunner()
    with patch("cli.service.stop"):
        result = runner.invoke(cli, ["stop"])
    assert result.exit_code == 0
    assert "stopped" in result.output.lower()


def test_stop_not_installed_shows_error():
    runner = CliRunner()
    with patch("cli.service.stop", side_effect=RuntimeError("Service not installed. Run: mlx-speech-server install")):
        result = runner.invoke(cli, ["stop"])
    assert result.exit_code == 1
    assert "not installed" in result.output.lower()


def test_restart_success():
    runner = CliRunner()
    with patch("cli.service.restart"), \
         patch("cli.service.get_status", return_value={"installed": False}):
        result = runner.invoke(cli, ["restart"])
    assert result.exit_code == 0


def test_status_not_installed():
    runner = CliRunner()
    with patch("cli.service.get_status", return_value={"installed": False}):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "Not installed" in result.output


def test_status_running():
    runner = CliRunner()
    with patch("cli.service.get_status", return_value={
        "installed": True,
        "loaded": True,
        "running": True,
        "pid": 12345,
        "port": 8000,
        "health": {"status": "ok"},
        "queue": {"queue_size": 0},
        "plist_path": service.PLIST_PATH,
        "venv_dir": service.VENV_DIR,
        "log_dir": service.LOG_DIR,
    }):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "12345" in result.output
    assert "8000" in result.output


def test_upgrade_up_to_date():
    runner = CliRunner()
    with patch("cli.service.upgrade", return_value={"status": "up_to_date"}):
        result = runner.invoke(cli, ["upgrade"])
    assert result.exit_code == 0
    assert "up to date" in result.output.lower()


def test_upgrade_prompts_and_restarts_on_yes():
    runner = CliRunner()
    with patch("cli.service.upgrade", return_value={"status": "upgraded"}), \
         patch("cli.service.restart") as mock_restart:
        result = runner.invoke(cli, ["upgrade"], input="y\n")
    assert result.exit_code == 0
    mock_restart.assert_called_once()


def test_upgrade_prompts_and_skips_restart_on_no():
    runner = CliRunner()
    with patch("cli.service.upgrade", return_value={"status": "upgraded"}), \
         patch("cli.service.restart") as mock_restart:
        result = runner.invoke(cli, ["upgrade"], input="n\n")
    assert result.exit_code == 0
    mock_restart.assert_not_called()
    assert "restart" in result.output.lower()


def test_upgrade_shows_error_on_git_failure():
    runner = CliRunner()
    with patch("cli.service.upgrade", side_effect=subprocess.CalledProcessError(1, "git")):
        result = runner.invoke(cli, ["upgrade"])
    assert result.exit_code == 1


def test_logs_output():
    runner = CliRunner()
    with patch("cli.service.get_logs", return_value=("stdout content", "stderr content")):
        result = runner.invoke(cli, ["logs"])
    assert result.exit_code == 0
    assert "stdout content" in result.output
    assert "stderr content" in result.output
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_cli.py -v
```

Expected: `ModuleNotFoundError: No module named 'cli'`

- [ ] **Step 3: Create cli.py**

```python
"""CLI entry point for mlx-speech-server service management."""
import subprocess
import sys

import click

from app import service


@click.group()
def cli() -> None:
    """mlx-speech-server: OpenAI-compatible Whisper API on Apple Silicon."""


@cli.command()
def install() -> None:
    """Create service venv, install deps, register launchd agent."""
    try:
        service.install()
    except Exception as e:
        click.echo(f"Install failed: {e}", err=True)
        sys.exit(1)
    click.echo("Installation complete.")
    click.echo(f"  Project:  {service.PROJECT_ROOT}")
    click.echo(f"  Venv:     {service.VENV_DIR}")
    click.echo(f"  Logs:     {service.LOG_DIR}")
    click.echo(f"  Plist:    {service.PLIST_PATH}")
    click.echo("")
    click.echo("Run: mlx-speech-server start")


@cli.command()
def uninstall() -> None:
    """Stop and remove launchd agent. Venv is kept."""
    try:
        service.uninstall()
    except RuntimeError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    click.echo("Service uninstalled.")
    click.echo(f"  Venv kept at: {service.VENV_DIR}")
    click.echo(f"  To remove it: rm -rf {service.VENV_DIR}")


@cli.command()
def upgrade() -> None:
    """Pull latest code and reinstall if updated."""
    try:
        result = service.upgrade()
    except subprocess.CalledProcessError as e:
        click.echo(f"Upgrade failed: {e}", err=True)
        sys.exit(1)
    if result["status"] == "up_to_date":
        click.echo("Already up to date, nothing to do.")
        return
    click.echo("Dependencies updated.")
    if click.confirm("Restart now?", default=False):
        try:
            service.restart()
            click.echo("Service restarted.")
        except Exception as e:
            click.echo(f"Restart failed: {e}", err=True)
            sys.exit(1)
    else:
        click.echo("Run: mlx-speech-server restart")


@cli.command()
def start() -> None:
    """Start service (auto-installs if needed)."""
    try:
        service.start()
    except Exception as e:
        click.echo(f"Start failed: {e}", err=True)
        sys.exit(1)
    click.echo("Service started.")
    _print_status()


@cli.command()
def stop() -> None:
    """Stop the running service."""
    try:
        service.stop()
    except RuntimeError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    click.echo("Service stopped.")


@cli.command()
def restart() -> None:
    """Restart the service."""
    try:
        service.restart()
    except RuntimeError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    click.echo("Service restarted.")
    _print_status()


@cli.command()
def status() -> None:
    """Show process status and health."""
    _print_status()


@cli.command()
def logs() -> None:
    """Show recent stdout and stderr logs."""
    stdout, stderr = service.get_logs()
    click.echo("─── stdout (last 30 lines) ─────────────")
    click.echo(stdout)
    click.echo("")
    click.echo("─── stderr (last 30 lines) ─────────────")
    click.echo(stderr)
    click.echo("")
    click.echo(f"Live tail: tail -f {service.STDOUT_LOG} {service.STDERR_LOG}")


def _print_status() -> None:
    s = service.get_status()
    click.echo("════════════════════════════════════════")
    click.echo("  mlx-speech-server — Status")
    click.echo("════════════════════════════════════════")
    if not s["installed"]:
        click.echo("  ⚪ Not installed")
        click.echo("════════════════════════════════════════")
        return
    click.echo(f"  Plist:    {s['plist_path']}")
    click.echo(f"  Venv:     {s['venv_dir']}")
    click.echo(f"  Logs:     {s['log_dir']}")
    if not s.get("loaded"):
        click.echo("")
        click.echo("  ⚪ Installed but not loaded")
        click.echo("════════════════════════════════════════")
        return
    click.echo("")
    if s["running"]:
        click.echo(f"  Process:  🟢 Running (PID {s['pid']})")
        click.echo(f"  Port:     {s['port']}")
        if s["health"]:
            click.echo(f"  Health:   🟢 OK — {s['health']}")
        else:
            click.echo("  Health:   🟡 Not responding yet (model may still be loading)")
        if s["queue"]:
            click.echo(f"  Queue:    {s['queue']}")
        click.echo("")
        click.echo("  API Docs:")
        click.echo(f"    Swagger UI → http://localhost:{s['port']}/docs")
        click.echo(f"    ReDoc      → http://localhost:{s['port']}/redoc")
    else:
        click.echo("  Process:  🔴 Not running")
    click.echo("════════════════════════════════════════")
```

- [ ] **Step 4: Run CLI tests**

```bash
pytest tests/test_cli.py -v
```

Expected: 15 passed.

- [ ] **Step 5: Run full test suite**

```bash
pytest -v
```

Expected: All tests pass.

- [ ] **Step 6: Smoke test the CLI**

```bash
pip install -e ".[dev]"
mlx-speech-server --help
```

Expected output:
```
Usage: mlx-speech-server [OPTIONS] COMMAND [ARGS]...

  mlx-speech-server: OpenAI-compatible Whisper API on Apple Silicon.

Options:
  --help  Show this message and exit.

Commands:
  install    Create service venv, install deps, register launchd agent.
  logs       Show recent stdout and stderr logs.
  restart    Restart the service.
  start      Start service (auto-installs if needed).
  status     Show process status and health.
  stop       Stop the running service.
  uninstall  Stop and remove launchd agent. Venv is kept.
  upgrade    Pull latest code and reinstall if updated.
```

- [ ] **Step 7: Commit**

```bash
git add cli.py tests/test_cli.py
git commit -m "feat: add cli.py with all service management subcommands"
```

---

### Task 11: Update README files

**Files:**
- Modify: `README.md`
- Modify: `README_zh.md`

- [ ] **Step 1: Update README.md quick start section**

Replace the managed service section. The new install flow is:

```markdown
### Managed service (recommended)

Install the CLI via [pipx](https://pipx.pypa.io/stable/) (isolated Python app installer):

\```bash
pipx install .
\```

Then manage the service:

\```bash
mlx-speech-server install
mlx-speech-server start
mlx-speech-server status
\```
```

Update the commands table — replace `./scripts/service.sh <cmd>` with `mlx-speech-server <cmd>` for all rows.

- [ ] **Step 2: Update README_zh.md quick start section**

Apply the same changes to `README_zh.md`. The pipx explanation in Chinese:

```markdown
通过 [pipx](https://pipx.pypa.io/stable/) 安装 CLI（隔离的 Python 应用安装工具）：

\```bash
pipx install .
\```
```

Update the commands table in the same way.

- [ ] **Step 3: Add note that python main.py still works for development**

In the Manual start / 手动启动 section of both READMEs, add a note that `python main.py` still works for foreground/dev usage (the existing launchd service runs `main.py` directly).

- [ ] **Step 4: Commit**

```bash
git add README.md README_zh.md
git commit -m "docs: update READMEs with pipx install and mlx-speech-server CLI commands"
```
