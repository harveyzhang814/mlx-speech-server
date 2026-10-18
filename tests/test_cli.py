import subprocess
import sys
import pytest
from click.testing import CliRunner
from unittest.mock import patch

if sys.platform != "darwin":
    pytest.skip("CLI service tests require macOS", allow_module_level=True)

from app.cli import cli
from app import service


def test_install_success():
    runner = CliRunner()
    with patch("app.cli.service.install"):
        result = runner.invoke(cli, ["install"])
    assert result.exit_code == 0
    assert "Installation complete" in result.output


def test_install_failure_shows_error():
    runner = CliRunner()
    with patch("app.cli.service.install", side_effect=Exception("pip failed")):
        result = runner.invoke(cli, ["install"])
    assert result.exit_code == 1
    assert "pip failed" in result.output


def test_uninstall_success():
    runner = CliRunner()
    with patch("app.cli.service.uninstall"):
        result = runner.invoke(cli, ["uninstall"])
    assert result.exit_code == 0
    assert "uninstalled" in result.output.lower()


def test_uninstall_not_installed_shows_error():
    runner = CliRunner()
    with patch("app.cli.service.uninstall", side_effect=RuntimeError("Service not installed. Run: mlx-speech-server install")):
        result = runner.invoke(cli, ["uninstall"])
    assert result.exit_code == 1
    assert "not installed" in result.output.lower()


def test_start_invokes_service_start():
    runner = CliRunner()
    with patch("app.cli.service.start"), \
         patch("app.cli.service.get_status", return_value={"installed": False}):
        result = runner.invoke(cli, ["start"])
    assert result.exit_code == 0


def test_stop_success():
    runner = CliRunner()
    with patch("app.cli.service.stop"):
        result = runner.invoke(cli, ["stop"])
    assert result.exit_code == 0
    assert "stopped" in result.output.lower()


def test_stop_not_installed_shows_error():
    runner = CliRunner()
    with patch("app.cli.service.stop", side_effect=RuntimeError("Service not installed. Run: mlx-speech-server install")):
        result = runner.invoke(cli, ["stop"])
    assert result.exit_code == 1
    assert "not installed" in result.output.lower()


def test_restart_success():
    runner = CliRunner()
    with patch("app.cli.service.restart"), \
         patch("app.cli.service.get_status", return_value={"installed": False}):
        result = runner.invoke(cli, ["restart"])
    assert result.exit_code == 0


def test_status_not_installed():
    runner = CliRunner()
    with patch("app.cli.service.get_status", return_value={"installed": False}):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "Not installed" in result.output


def test_status_running():
    runner = CliRunner()
    with patch("app.cli.service.get_status", return_value={
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
    with patch("app.cli.service.upgrade", return_value={"status": "up_to_date"}):
        result = runner.invoke(cli, ["upgrade"])
    assert result.exit_code == 0
    assert "up to date" in result.output.lower()


def test_upgrade_prompts_and_restarts_on_yes():
    runner = CliRunner()
    with patch("app.cli.service.upgrade", return_value={"status": "upgraded"}), \
         patch("app.cli.service.restart") as mock_restart:
        result = runner.invoke(cli, ["upgrade"], input="y\n")
    assert result.exit_code == 0
    mock_restart.assert_called_once()


def test_upgrade_prompts_and_skips_restart_on_no():
    runner = CliRunner()
    with patch("app.cli.service.upgrade", return_value={"status": "upgraded"}), \
         patch("app.cli.service.restart") as mock_restart:
        result = runner.invoke(cli, ["upgrade"], input="n\n")
    assert result.exit_code == 0
    mock_restart.assert_not_called()
    assert "restart" in result.output.lower()


def test_upgrade_shows_error_on_git_failure():
    runner = CliRunner()
    with patch("app.cli.service.upgrade", side_effect=subprocess.CalledProcessError(1, "git")):
        result = runner.invoke(cli, ["upgrade"])
    assert result.exit_code == 1


def test_stop_launchctl_failure_shows_error():
    runner = CliRunner()
    with patch("app.cli.service.stop", side_effect=subprocess.CalledProcessError(1, "launchctl")):
        result = runner.invoke(cli, ["stop"])
    assert result.exit_code == 1


def test_uninstall_launchctl_failure_shows_error():
    runner = CliRunner()
    with patch("app.cli.service.uninstall", side_effect=subprocess.CalledProcessError(1, "launchctl")):
        result = runner.invoke(cli, ["uninstall"])
    assert result.exit_code == 1


def test_logs_output():
    runner = CliRunner()
    with patch("app.cli.service.get_logs", return_value=("stdout content", "stderr content")):
        result = runner.invoke(cli, ["logs"])
    assert result.exit_code == 0
    assert "stdout content" in result.output
    assert "stderr content" in result.output


def test_ensure_running_json_and_first_install_options():
    import json
    payload = {"running": True, "pid": 1, "port": 47301, "health": {"status": "ok"},
               "service_label": service.SERVICE_LABEL}
    with patch("app.cli.service.ensure_running", return_value=payload, create=True) as ensure:
        result = CliRunner().invoke(cli, ["ensure-running", "--json", "--model", "mlx-community/whisper-small-mlx", "--port", "47301"])
    assert result.exit_code == 0
    assert json.loads(result.output) == payload
    ensure.assert_called_once_with(model="mlx-community/whisper-small-mlx", port=47301)


def test_ensure_running_error_json_exits_nonzero():
    import json
    payload = {"running": False, "pid": None, "port": 47300, "health": None,
               "service_label": service.SERVICE_LABEL,
               "error": {"code": "port_in_use", "message": "Another listener"}}
    with patch("app.cli.service.ensure_running", return_value=payload, create=True):
        result = CliRunner().invoke(cli, ["ensure-running", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.output) == payload


def test_install_json_is_noninteractive():
    import json
    with patch("app.cli.service.install") as install:
        result = CliRunner().invoke(cli, ["install", "--json", "--port", "47301"])
    assert result.exit_code == 0
    assert json.loads(result.output)["installed"] is True
    install.assert_called_once_with(model=None, port=47301)


def test_install_failure_json_exits_nonzero():
    import json
    with patch("app.cli.service.install", side_effect=RuntimeError("pip failed")):
        result = CliRunner().invoke(cli, ["install", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.output)["error"]["code"] == "install_failed"


def test_ensure_running_can_register_locked_current_runtime():
    payload = {"running": True, "pid": 1, "port": 47300, "health": {"status": "ok"},
               "service_label": service.SERVICE_LABEL}
    with patch("app.cli.service.ensure_running", return_value=payload, create=True) as ensure:
        result = CliRunner().invoke(cli, ["ensure-running", "--json", "--use-current-runtime"])
    assert result.exit_code == 0
    ensure.assert_called_once_with(model=None, port=None, use_current_runtime=True)
