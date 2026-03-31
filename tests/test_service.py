import subprocess  # noqa: F401  # Used in later tasks (CalledProcessError)
import sys
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


def test_build_plist_escapes_xml_special_chars():
    plist = service._build_plist({"WHISPER_EXTRA": "a&b<c>d"})
    assert "<string>a&amp;b&lt;c&gt;d</string>" in plist
    assert "a&b<c>d" not in plist


def test_install_creates_venv_when_not_exists(fake_paths, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(service.subprocess, "run", fake_run)

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

    service.install()

    venv_calls = [c for c in calls if "-m" in c and "venv" in c]
    assert len(venv_calls) == 0


def test_install_runs_pip_install(fake_paths, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(service.subprocess, "run", fake_run)

    service.install()

    pip_calls = [c for c in calls if "pip" in str(c[0]) and "install" in c]
    assert len(pip_calls) == 1
    assert "-e" in pip_calls[0]


def test_install_writes_plist(fake_paths, monkeypatch):
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))

    service.install()

    plist = fake_paths / "LaunchAgents" / "com.local.mlx-speech-server.plist"
    assert plist.exists()
    assert service.SERVICE_LABEL in plist.read_text()


def test_install_is_idempotent(fake_paths, monkeypatch):
    """Running install twice should not raise."""
    python = fake_paths / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))

    service.install()
    service.install()  # Should not raise


def test_stop_calls_bootout(installed, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(service.subprocess, "run", fake_run)

    service.stop()

    assert any("bootout" in c for c in calls)


def test_stop_raises_when_not_installed(fake_paths):
    with pytest.raises(RuntimeError, match="not installed"):
        service.stop()


def test_uninstall_removes_plist(installed, monkeypatch):
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))
    plist = installed / "LaunchAgents" / "com.local.mlx-speech-server.plist"
    assert plist.exists()

    service.uninstall()

    assert not plist.exists()


def test_uninstall_calls_bootout(installed, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(service.subprocess, "run", fake_run)

    service.uninstall()

    assert any("bootout" in c for c in calls)


def test_uninstall_raises_when_not_installed(fake_paths):
    with pytest.raises(RuntimeError, match="not installed"):
        service.uninstall()


def test_start_auto_installs_when_not_installed(fake_paths, monkeypatch):
    install_calls = []
    monkeypatch.setattr(service, "install", lambda: install_calls.append(True))
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))

    service.start()

    assert len(install_calls) == 1


def test_start_skips_install_when_already_installed(installed, monkeypatch):
    install_calls = []
    monkeypatch.setattr(service, "install", lambda: install_calls.append(True))
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))

    service.start()

    assert len(install_calls) == 0


def test_start_calls_bootstrap_and_kickstart(installed, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(service.subprocess, "run", fake_run)

    service.start()

    assert any("bootstrap" in c for c in calls)
    assert any("kickstart" in c for c in calls)


def test_restart_calls_bootout_before_bootstrap(installed, monkeypatch):
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return MagicMock(returncode=0)
    monkeypatch.setattr(service.subprocess, "run", fake_run)

    service.restart()

    bootout_idx = next(i for i, c in enumerate(calls) if "bootout" in c)
    bootstrap_idx = next(i for i, c in enumerate(calls) if "bootstrap" in c)
    assert bootout_idx < bootstrap_idx
    assert any("kickstart" in c for c in calls)


def test_restart_raises_when_not_installed(fake_paths):
    with pytest.raises(RuntimeError, match="not installed"):
        service.restart()
