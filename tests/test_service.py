import sys
from unittest.mock import patch

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
