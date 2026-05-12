# Service Dual-Install Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `app/service.py` work correctly whether the package was installed from PyPI (`pip install mlx-speech-server`) or from a local git clone (`pip install -e .`), with config stored at `~/.config/mlx-speech-server/config.env` instead of the project's `.env` file.

**Architecture:** Add `CONFIG_DIR`/`CONFIG_ENV` constants replacing `PROJECT_ROOT / ".env"`. Add `_is_editable_install()` / `_get_project_dir()` / `_get_install_args()` helpers that read pip's `direct_url.json` to detect install source. `install()` uses `_get_install_args()` for pip. `upgrade()` branches on `_is_editable_install()`: editable → `git pull` + `pip install -e .`; PyPI → `pip install --upgrade mlx-speech-server`. `PROJECT_ROOT` is removed as a module-level constant.

**Tech Stack:** Python stdlib (`importlib.metadata`, `urllib.parse`, `json`, `pathlib`), pytest, Click

---

## File Map

| File | Change | Responsibility |
| :--- | :--- | :--- |
| `app/service.py` | Modify | Add config constants, install-source helpers, update `install()` / `upgrade()` / `_read_env()` |
| `tests/test_service.py` | Modify | Update `fake_paths` fixture, replace `PROJECT_ROOT`-based tests, add new helper + upgrade tests |
| `app/cli.py` | Modify | Update `install` success message to show config file path |
| `README.md` | Modify | Update config section to reference `~/.config/mlx-speech-server/config.env` |
| `README_zh.md` | Modify | Same in Chinese |

---

### Task 1: Add CONFIG constants and refactor `_read_env()`

Config should live at `~/.config/mlx-speech-server/config.env`, independent of any project directory.

**Files:**
- Modify: `app/service.py`
- Modify: `tests/test_service.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_service.py`, replace the four `test_read_env_*` tests and update the `fake_paths` fixture. The new fixture drops `PROJECT_ROOT` and adds `CONFIG_DIR` / `CONFIG_ENV`:

```python
@pytest.fixture
def fake_paths(tmp_path):
    plist = tmp_path / "LaunchAgents" / "com.local.mlx-speech-server.plist"
    with patch.multiple(
        service,
        PLIST_PATH=plist,
        VENV_DIR=tmp_path / "venv",
        LOG_DIR=tmp_path / "logs",
        CONFIG_DIR=tmp_path / "config",
        CONFIG_ENV=tmp_path / "config" / "config.env",
        STDOUT_LOG=tmp_path / "logs" / "server.log",
        STDERR_LOG=tmp_path / "logs" / "server.err",
    ):
        yield tmp_path
```

Replace the four `test_read_env_*` tests with:

```python
def test_read_env_parses_key_value_pairs(fake_paths):
    config_env = fake_paths / "config" / "config.env"
    config_env.parent.mkdir(parents=True, exist_ok=True)
    config_env.write_text("WHISPER_PORT=9000\nWHISPER_MODEL_PATH=mlx-community/whisper-large\n")
    result = service._read_env()
    assert result == {
        "WHISPER_PORT": "9000",
        "WHISPER_MODEL_PATH": "mlx-community/whisper-large",
    }


def test_read_env_skips_comments_and_blank_lines(fake_paths):
    config_env = fake_paths / "config" / "config.env"
    config_env.parent.mkdir(parents=True, exist_ok=True)
    config_env.write_text("# comment\n\nWHISPER_PORT=8000\n")
    result = service._read_env()
    assert result == {"WHISPER_PORT": "8000"}


def test_read_env_strips_quotes(fake_paths):
    config_env = fake_paths / "config" / "config.env"
    config_env.parent.mkdir(parents=True, exist_ok=True)
    config_env.write_text('WHISPER_MODEL_PATH="mlx-community/whisper-turbo"\n')
    result = service._read_env()
    assert result == {"WHISPER_MODEL_PATH": "mlx-community/whisper-turbo"}


def test_read_env_returns_empty_when_no_file(fake_paths):
    result = service._read_env()
    assert result == {}
```

- [ ] **Step 2: Run to confirm failure**

```bash
python3 -m pytest tests/test_service.py::test_read_env_parses_key_value_pairs -v
```

Expected: FAIL — either `PROJECT_ROOT` attribute error or file not found.

- [ ] **Step 3: Update `app/service.py`**

Replace the `_find_project_root()` function and `PROJECT_ROOT` constant with config constants, and update `_read_env()`:

```python
# Remove _find_project_root() and PROJECT_ROOT entirely.
# Add after SERVICE_LABEL:

CONFIG_DIR = Path.home() / ".config/mlx-speech-server"
CONFIG_ENV = CONFIG_DIR / "config.env"
VENV_DIR = Path.home() / ".local/venvs/mlx-speech-server"
LOG_DIR = Path.home() / ".local/logs/mlx-speech-server"
PLIST_PATH = Path.home() / "Library/LaunchAgents" / f"{SERVICE_LABEL}.plist"
STDOUT_LOG = LOG_DIR / "server.log"
STDERR_LOG = LOG_DIR / "server.err"
```

Update `_read_env()`:

```python
def _read_env() -> dict[str, str]:
    """Parse config.env from ~/.config/mlx-speech-server/ into a dict."""
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
```

Also remove `urllib.parse` from imports (no longer needed yet — it will be re-added in Task 2).

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_service.py -v 2>&1 | tail -20
```

Expected: All pass except tests that still reference `PROJECT_ROOT` (install/upgrade tests). Those will be fixed in Tasks 2–3.

- [ ] **Step 5: Commit**

```bash
git add app/service.py tests/test_service.py
git commit -m "refactor: move service config from project .env to ~/.config/mlx-speech-server/config.env"
```

---

### Task 2: Add install-source detection helpers and update `install()`

**Files:**
- Modify: `app/service.py`
- Modify: `tests/test_service.py`

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/test_service.py` (after the `test_build_plist_*` group):

```python
def test_is_editable_install_true(monkeypatch):
    fake_dist = MagicMock()
    fake_dist.read_text.return_value = '{"url": "file:///home/user/project", "dir_info": {"editable": true}}'
    monkeypatch.setattr(service.importlib.metadata, "distribution", lambda _: fake_dist)
    assert service._is_editable_install() is True


def test_is_editable_install_false_for_pypi(monkeypatch):
    fake_dist = MagicMock()
    fake_dist.read_text.return_value = '{"url": "https://files.pythonhosted.org/...", "dir_info": {}}'
    monkeypatch.setattr(service.importlib.metadata, "distribution", lambda _: fake_dist)
    assert service._is_editable_install() is False


def test_is_editable_install_false_when_no_metadata(monkeypatch):
    monkeypatch.setattr(
        service.importlib.metadata, "distribution",
        lambda _: (_ for _ in ()).throw(service.importlib.metadata.PackageNotFoundError())
    )
    assert service._is_editable_install() is False


def test_get_project_dir_returns_path_for_editable(monkeypatch):
    fake_dist = MagicMock()
    fake_dist.read_text.return_value = '{"url": "file:///home/user/myproject", "dir_info": {"editable": true}}'
    monkeypatch.setattr(service.importlib.metadata, "distribution", lambda _: fake_dist)
    assert service._get_project_dir() == Path("/home/user/myproject")


def test_get_project_dir_raises_for_non_editable(monkeypatch):
    fake_dist = MagicMock()
    fake_dist.read_text.return_value = '{"url": "https://files.pythonhosted.org/...", "dir_info": {}}'
    monkeypatch.setattr(service.importlib.metadata, "distribution", lambda _: fake_dist)
    with pytest.raises(RuntimeError, match="editable"):
        service._get_project_dir()


def test_get_install_args_editable(monkeypatch):
    monkeypatch.setattr(service, "_is_editable_install", lambda: True)
    monkeypatch.setattr(service, "_get_project_dir", lambda: Path("/home/user/project"))
    assert service._get_install_args() == ["-e", "/home/user/project"]


def test_get_install_args_pypi(monkeypatch):
    monkeypatch.setattr(service, "_is_editable_install", lambda: False)
    assert service._get_install_args() == ["mlx-speech-server"]
```

Replace the three `test_install_*` tests that check pip args / plist content:

```python
def test_install_creates_venv_when_not_exists(fake_paths, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(service.subprocess, "run", fake_run)
    monkeypatch.setattr(service, "_get_install_args", lambda: ["mlx-speech-server"])

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
    monkeypatch.setattr(service.subprocess, "run", fake_run)
    monkeypatch.setattr(service, "_get_install_args", lambda: ["mlx-speech-server"])

    service.install()

    venv_calls = [c for c in calls if "-m" in c and "venv" in c]
    assert len(venv_calls) == 0


def test_install_runs_pip_install_with_get_install_args(fake_paths, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(service.subprocess, "run", fake_run)
    monkeypatch.setattr(service, "_get_install_args", lambda: ["-e", "/home/user/project"])

    service.install()

    pip_calls = [c for c in calls if "pip" in str(c[0]) and "install" in c]
    assert len(pip_calls) == 1
    assert "-e" in pip_calls[0]
    assert "/home/user/project" in pip_calls[0]


def test_install_pip_uses_package_name_for_pypi(fake_paths, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(service.subprocess, "run", fake_run)
    monkeypatch.setattr(service, "_get_install_args", lambda: ["mlx-speech-server"])

    service.install()

    pip_calls = [c for c in calls if "pip" in str(c[0]) and "install" in c]
    assert len(pip_calls) == 1
    assert "mlx-speech-server" in pip_calls[0]
    assert "-e" not in pip_calls[0]


def test_install_writes_plist(fake_paths, monkeypatch):
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))
    monkeypatch.setattr(service, "_get_install_args", lambda: ["mlx-speech-server"])

    service.install()

    plist = fake_paths / "LaunchAgents" / "com.local.mlx-speech-server.plist"
    assert plist.exists()
    assert service.SERVICE_LABEL in plist.read_text()


def test_install_creates_config_dir(fake_paths, monkeypatch):
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))
    monkeypatch.setattr(service, "_get_install_args", lambda: ["mlx-speech-server"])

    service.install()

    assert (fake_paths / "config").exists()


def test_install_is_idempotent(fake_paths, monkeypatch):
    python = fake_paths / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))
    monkeypatch.setattr(service, "_get_install_args", lambda: ["mlx-speech-server"])

    service.install()
    service.install()
```

- [ ] **Step 2: Run to confirm failure**

```bash
python3 -m pytest tests/test_service.py::test_is_editable_install_true -v
```

Expected: FAIL — `service._is_editable_install` does not exist.

- [ ] **Step 3: Add helpers to `app/service.py`**

Add `urllib.parse` back to imports. Add after the constants block:

```python
def _is_editable_install() -> bool:
    """Return True if the package was installed in editable mode (local clone)."""
    try:
        dist = importlib.metadata.distribution("mlx-speech-server")
        raw = dist.read_text("direct_url.json")
        if raw:
            data = json.loads(raw)
            return bool(data.get("dir_info", {}).get("editable"))
    except Exception:
        pass
    return False


def _get_project_dir() -> Path:
    """Return the project source directory for an editable install.

    Raises RuntimeError if not an editable install.
    """
    try:
        dist = importlib.metadata.distribution("mlx-speech-server")
        raw = dist.read_text("direct_url.json")
        if raw:
            data = json.loads(raw)
            if data.get("dir_info", {}).get("editable"):
                url = data.get("url", "")
                if url.startswith("file://"):
                    return Path(urllib.parse.urlparse(url).path)
    except Exception:
        pass
    raise RuntimeError(
        "Not an editable install. Use 'pip install -e .' for local clone upgrades."
    )


def _get_install_args() -> list[str]:
    """Return pip install arguments for installing into the service venv."""
    if _is_editable_install():
        return ["-e", str(_get_project_dir())]
    return ["mlx-speech-server"]
```

Update `install()` to use `_get_install_args()` and create the config dir:

```python
def install() -> None:
    """Create service venv, install package, register launchd plist."""
    _require_darwin()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not (VENV_DIR / "bin/python").exists():
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)

    subprocess.run(
        [str(VENV_DIR / "bin/pip"), "install"] + _get_install_args(),
        check=True,
    )

    env_vars = _read_env()
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(_build_plist(env_vars))
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_service.py -v 2>&1 | tail -25
```

Expected: All pass. (upgrade tests may still fail — fixed in Task 3.)

- [ ] **Step 5: Commit**

```bash
git add app/service.py tests/test_service.py
git commit -m "feat: add install-source detection, update install() to support PyPI and editable installs"
```

---

### Task 3: Update `upgrade()` for dual-path support

**Files:**
- Modify: `app/service.py`
- Modify: `tests/test_service.py`

- [ ] **Step 1: Write failing tests**

Replace all four `test_upgrade_*` tests in `tests/test_service.py`:

```python
def test_upgrade_editable_returns_up_to_date(fake_paths, monkeypatch):
    monkeypatch.setattr(service, "_is_editable_install", lambda: True)
    monkeypatch.setattr(service, "_get_project_dir", lambda: Path("/home/user/project"))

    def fake_run(args, **kwargs):
        m = MagicMock()
        m.returncode = 0
        m.stdout = "Already up to date.\n"
        return m
    monkeypatch.setattr(service.subprocess, "run", fake_run)

    result = service.upgrade()
    assert result == {"status": "up_to_date"}


def test_upgrade_editable_installs_when_new_commits(fake_paths, monkeypatch):
    monkeypatch.setattr(service, "_is_editable_install", lambda: True)
    monkeypatch.setattr(service, "_get_project_dir", lambda: Path("/home/user/project"))

    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        m = MagicMock()
        m.returncode = 0
        m.stdout = "Fast-forward\n 1 file changed\n"
        return m
    monkeypatch.setattr(service.subprocess, "run", fake_run)

    result = service.upgrade()

    assert result == {"status": "upgraded"}
    pip_calls = [c for c in calls if "pip" in str(c[0]) and "install" in c]
    assert len(pip_calls) == 1


def test_upgrade_editable_skips_pip_when_up_to_date(fake_paths, monkeypatch):
    monkeypatch.setattr(service, "_is_editable_install", lambda: True)
    monkeypatch.setattr(service, "_get_project_dir", lambda: Path("/home/user/project"))

    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        m = MagicMock()
        m.returncode = 0
        m.stdout = "Already up to date.\n"
        return m
    monkeypatch.setattr(service.subprocess, "run", fake_run)

    service.upgrade()

    pip_calls = [c for c in calls if "pip" in str(c[0]) and "install" in c]
    assert len(pip_calls) == 0


def test_upgrade_editable_raises_on_git_failure(fake_paths, monkeypatch):
    monkeypatch.setattr(service, "_is_editable_install", lambda: True)
    monkeypatch.setattr(service, "_get_project_dir", lambda: Path("/home/user/project"))

    def raise_error(args, **kwargs):
        raise subprocess.CalledProcessError(1, "git", stderr="network error")
    monkeypatch.setattr(service.subprocess, "run", raise_error)

    with pytest.raises(subprocess.CalledProcessError):
        service.upgrade()


def test_upgrade_pypi_runs_pip_upgrade(fake_paths, monkeypatch):
    monkeypatch.setattr(service, "_is_editable_install", lambda: False)

    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        m = MagicMock()
        m.returncode = 0
        m.stdout = "Successfully installed mlx-speech-server-1.1.0\n"
        return m
    monkeypatch.setattr(service.subprocess, "run", fake_run)

    result = service.upgrade()

    assert result == {"status": "upgraded"}
    assert any("--upgrade" in c and "mlx-speech-server" in c for c in calls)


def test_upgrade_pypi_returns_up_to_date_when_already_satisfied(fake_paths, monkeypatch):
    monkeypatch.setattr(service, "_is_editable_install", lambda: False)

    def fake_run(args, **kwargs):
        m = MagicMock()
        m.returncode = 0
        m.stdout = "Requirement already satisfied: mlx-speech-server\n"
        return m
    monkeypatch.setattr(service.subprocess, "run", fake_run)

    result = service.upgrade()
    assert result == {"status": "up_to_date"}
```

- [ ] **Step 2: Run to confirm failure**

```bash
python3 -m pytest tests/test_service.py::test_upgrade_pypi_runs_pip_upgrade -v
```

Expected: FAIL — `upgrade()` still calls `git` unconditionally.

- [ ] **Step 3: Update `upgrade()` in `app/service.py`**

```python
def upgrade() -> dict[str, str]:
    """
    Update the service to the latest version.

    Editable install (local clone): git pull origin main, then pip install -e . if new commits.
    PyPI install: pip install --upgrade mlx-speech-server.

    Returns {"status": "up_to_date"} or {"status": "upgraded"}.
    Raises subprocess.CalledProcessError on failure.
    """
    _require_darwin()

    if _is_editable_install():
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
```

- [ ] **Step 4: Run full test suite**

```bash
python3 -m pytest -v 2>&1 | tail -15
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add app/service.py tests/test_service.py
git commit -m "feat: upgrade() supports both PyPI (pip --upgrade) and editable (git pull) installs"
```

---

### Task 4: Update CLI message and README

**Files:**
- Modify: `app/cli.py`
- Modify: `README.md`
- Modify: `README_zh.md`

- [ ] **Step 1: Update `app/cli.py` install success message**

In `app/cli.py`, find the `install` command's success output. Update to show config file path instead of project root:

```python
@cli.command()
def install() -> None:
    """Create service venv, install deps, register launchd agent."""
    try:
        service.install()
    except Exception as e:
        click.echo(f"Install failed: {e}")
        sys.exit(1)
    click.echo("Installation complete.")
    click.echo(f"  Venv:     {service.VENV_DIR}")
    click.echo(f"  Config:   {service.CONFIG_ENV}")
    click.echo(f"  Logs:     {service.LOG_DIR}")
    click.echo(f"  Plist:    {service.PLIST_PATH}")
    click.echo("")
    click.echo(f"Edit {service.CONFIG_ENV} to set WHISPER_* variables.")
    click.echo("Run: mlx-speech-server start")
```

- [ ] **Step 2: Update test for install success message**

In `tests/test_cli.py`, update `test_install_success` to check for "Config:" instead of "Project:":

```python
def test_install_success():
    runner = CliRunner()
    with patch("app.cli.service.install"):
        result = runner.invoke(cli, ["install"])
    assert result.exit_code == 0
    assert "Installation complete" in result.output
```

(The test only checks "Installation complete" so it may already pass — verify before changing.)

- [ ] **Step 3: Run CLI tests**

```bash
python3 -m pytest tests/test_cli.py -v 2>&1 | tail -10
```

Expected: All 17 pass.

- [ ] **Step 4: Update README.md config section**

Find the Configuration section (around line 270–290). Replace the `.env` file reference:

**Before:**
```markdown
Create a `.env` file in the project root:
```

**After:**
```markdown
Create `~/.config/mlx-speech-server/config.env` (auto-created by `mlx-speech-server install`):
```

Also update the reinstall example at line ~286 if it references `.env` path.

- [ ] **Step 5: Update README_zh.md config section**

Same change in Chinese. Find the 配置 section. Replace:
```markdown
在项目根目录创建 `.env` 文件：
```
with:
```markdown
创建 `~/.config/mlx-speech-server/config.env`（`mlx-speech-server install` 时自动创建）：
```

- [ ] **Step 6: Run full suite and lint**

```bash
python3 -m pytest -q 2>&1 | tail -5
ruff check app/service.py app/cli.py tests/test_service.py tests/test_cli.py
```

Expected: All pass, no lint errors.

- [ ] **Step 7: Commit**

```bash
git add app/cli.py tests/test_cli.py README.md README_zh.md
git commit -m "docs: update install message and README config section to use ~/.config path"
```
