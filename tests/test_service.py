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
