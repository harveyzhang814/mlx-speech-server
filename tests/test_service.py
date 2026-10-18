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
        CONFIG_DIR=tmp_path / "config",
        CONFIG_ENV=tmp_path / "config" / "config.env",
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


def test_build_plist_contains_service_label():
    plist = service._build_plist({})
    assert service.SERVICE_LABEL in plist


def test_build_plist_contains_runner_script():
    plist = service._build_plist({})
    assert str(service.VENV_DIR / "bin/mlx-run") in plist


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


def test_is_local_install_true_for_editable(monkeypatch):
    import importlib.metadata as _meta
    fake_dist = MagicMock()
    fake_dist.read_text.return_value = '{"url": "file:///home/user/project", "dir_info": {"editable": true}}'
    monkeypatch.setattr(_meta, "distribution", lambda _: fake_dist)
    assert service._is_local_install() is True


def test_is_local_install_true_for_pipx(monkeypatch):
    import importlib.metadata as _meta
    fake_dist = MagicMock()
    fake_dist.read_text.return_value = '{"url": "file:///home/user/project", "dir_info": {"editable": false}}'
    monkeypatch.setattr(_meta, "distribution", lambda _: fake_dist)
    assert service._is_local_install() is True


def test_is_local_install_false_for_pypi(monkeypatch):
    import importlib.metadata as _meta
    fake_dist = MagicMock()
    fake_dist.read_text.return_value = '{"url": "https://files.pythonhosted.org/...", "dir_info": {}}'
    monkeypatch.setattr(_meta, "distribution", lambda _: fake_dist)
    assert service._is_local_install() is False


def test_is_local_install_false_when_no_metadata(monkeypatch):
    import importlib.metadata as _meta

    def raise_not_found(_):
        raise _meta.PackageNotFoundError()

    monkeypatch.setattr(_meta, "distribution", raise_not_found)
    assert service._is_local_install() is False


def test_get_project_dir_returns_path_for_editable(monkeypatch):
    import importlib.metadata as _meta
    fake_dist = MagicMock()
    fake_dist.read_text.return_value = '{"url": "file:///home/user/myproject", "dir_info": {"editable": true}}'
    monkeypatch.setattr(_meta, "distribution", lambda _: fake_dist)
    assert service._get_project_dir() == Path("/home/user/myproject")


def test_get_project_dir_raises_for_non_local(monkeypatch):
    import importlib.metadata as _meta
    fake_dist = MagicMock()
    fake_dist.read_text.return_value = '{"url": "https://files.pythonhosted.org/...", "dir_info": {}}'
    monkeypatch.setattr(_meta, "distribution", lambda _: fake_dist)
    with pytest.raises(RuntimeError, match="local install"):
        service._get_project_dir()


def test_get_install_args_local(monkeypatch):
    monkeypatch.setattr(service, "_is_local_install", lambda: True)
    monkeypatch.setattr(service, "_get_project_dir", lambda: Path("/home/user/project"))
    assert service._get_install_args() == ["-e", "/home/user/project"]


def test_get_install_args_pypi(monkeypatch):
    monkeypatch.setattr(service, "_is_local_install", lambda: False)
    assert service._get_install_args() == ["mlx-speech-server"]


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


def test_install_creates_config_dir(fake_paths, monkeypatch):
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))
    monkeypatch.setattr(service, "_get_install_args", lambda: ["mlx-speech-server"])

    service.install()

    assert (fake_paths / "config").exists()


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


def test_install_writes_plist(fake_paths, monkeypatch):
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))
    monkeypatch.setattr(service, "_get_install_args", lambda: ["mlx-speech-server"])

    service.install()

    plist = fake_paths / "LaunchAgents" / "com.local.mlx-speech-server.plist"
    assert plist.exists()
    assert service.SERVICE_LABEL in plist.read_text()


def test_install_is_idempotent(fake_paths, monkeypatch):
    python = fake_paths / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))
    monkeypatch.setattr(service, "_get_install_args", lambda: ["mlx-speech-server"])

    service.install()
    service.install()


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


def test_upgrade_editable_returns_up_to_date(fake_paths, monkeypatch):
    monkeypatch.setattr(service, "_is_local_install", lambda: True)
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
    monkeypatch.setattr(service, "_is_local_install", lambda: True)
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
    monkeypatch.setattr(service, "_is_local_install", lambda: True)
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
    monkeypatch.setattr(service, "_is_local_install", lambda: True)
    monkeypatch.setattr(service, "_get_project_dir", lambda: Path("/home/user/project"))

    def raise_error(args, **kwargs):
        raise subprocess.CalledProcessError(1, "git", stderr="network error")
    monkeypatch.setattr(service.subprocess, "run", raise_error)

    with pytest.raises(subprocess.CalledProcessError):
        service.upgrade()


def test_upgrade_pypi_runs_pip_upgrade(fake_paths, monkeypatch):
    monkeypatch.setattr(service, "_is_local_install", lambda: False)

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
    monkeypatch.setattr(service, "_is_local_install", lambda: False)

    def fake_run(args, **kwargs):
        m = MagicMock()
        m.returncode = 0
        m.stdout = "Requirement already satisfied: mlx-speech-server\n"
        return m
    monkeypatch.setattr(service.subprocess, "run", fake_run)

    result = service.upgrade()
    assert result == {"status": "up_to_date"}


def test_get_status_not_installed(fake_paths):
    result = service.get_status()
    assert result == {"installed": False}


def test_get_status_installed_but_not_loaded(installed, monkeypatch):
    def fake_run(args, **kwargs):
        m = MagicMock()
        m.returncode = 1
        m.stdout = ""
        return m
    monkeypatch.setattr(service.subprocess, "run", fake_run)

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
    monkeypatch.setattr(service.subprocess, "run", fake_run)
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
    monkeypatch.setattr(service.subprocess, "run", fake_run)
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


def test_install_creates_loopback_config_with_first_model_and_port(fake_paths, monkeypatch):
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))
    monkeypatch.setattr(service, "_get_install_args", lambda: ["mlx-speech-server"])
    service.install(model="mlx-community/whisper-small-mlx", port=47301)
    assert service._read_env() == {
        "WHISPER_HOST": "127.0.0.1", "WHISPER_PORT": "47301",
        "WHISPER_MODEL_PATH": "mlx-community/whisper-small-mlx",
    }
    assert "127.0.0.1" in service.PLIST_PATH.read_text()


def test_install_preserves_existing_config_bytes(fake_paths, monkeypatch):
    service.CONFIG_DIR.mkdir()
    original = "# custom config\nWHISPER_HOST=0.0.0.0\nWHISPER_PORT=9000\n"
    service.CONFIG_ENV.write_text(original)
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))
    monkeypatch.setattr(service, "_get_install_args", lambda: ["mlx-speech-server"])
    service.install(model="ignored", port=47301)
    assert service.CONFIG_ENV.read_text() == original


def test_get_install_args_preserves_wheel_source(monkeypatch):
    fake_dist = MagicMock()
    fake_dist.read_text.return_value = '{"url":"file:///tmp/mlx_speech_server-0.3.0-py3-none-any.whl","archive_info":{}}'
    monkeypatch.setattr(service.importlib.metadata, "distribution", lambda _: fake_dist)
    assert service._get_install_args() == ["/tmp/mlx_speech_server-0.3.0-py3-none-any.whl"]


def test_install_current_runtime_registers_existing_runner_without_pip(fake_paths, monkeypatch):
    runner = fake_paths / "locked-runtime/bin/mlx-run"
    runner.parent.mkdir(parents=True)
    runner.write_text("#!/bin/sh\n")
    runner.chmod(0o755)
    monkeypatch.setattr(service.sys, "executable", str(runner.parent / "python"))
    run = MagicMock(side_effect=AssertionError("No subprocess needed"))
    monkeypatch.setattr(service.subprocess, "run", run)
    service.install(use_current_runtime=True)
    import plistlib
    plist = plistlib.loads(service.PLIST_PATH.read_bytes())
    assert plist["ProgramArguments"] == [str(runner)]
    assert plist["EnvironmentVariables"]["PATH"].startswith(str(runner.parent) + ":")
    assert not service.VENV_DIR.exists()
    run.assert_not_called()


def test_install_current_runtime_rejects_missing_runner_before_writes(fake_paths, monkeypatch):
    monkeypatch.setattr(service.sys, "executable", str(fake_paths / "missing/bin/python"))
    with pytest.raises(RuntimeError, match="mlx-run"):
        service.install(use_current_runtime=True)
    assert not service.PLIST_PATH.exists()
    assert not service.CONFIG_ENV.exists()


def test_current_runtime_install_preserves_existing_plist(installed, monkeypatch):
    original = service.PLIST_PATH.read_bytes()
    monkeypatch.setattr(service.sys, "executable", str(installed / "missing/bin/python"))
    monkeypatch.setattr(service.subprocess, "run", MagicMock(side_effect=AssertionError("no install")))
    service.install(use_current_runtime=True)
    assert service.PLIST_PATH.read_bytes() == original
    assert not service.CONFIG_ENV.exists()


@pytest.mark.parametrize("model,port", [("bad\nWHISPER_PORT=1", None), ("", None), (None, 0), (None, 65536)])
def test_install_rejects_invalid_first_config_without_writes(fake_paths, monkeypatch, model, port):
    run = MagicMock(side_effect=AssertionError("no install"))
    monkeypatch.setattr(service.subprocess, "run", run)
    with pytest.raises(ValueError):
        service.install(model=model, port=port)
    assert not service.CONFIG_ENV.exists()
    run.assert_not_called()


def test_get_install_args_decodes_wheel_file_url(monkeypatch):
    fake_dist = MagicMock()
    fake_dist.read_text.return_value = '{"url":"file:///tmp/App%20Support/mlx_speech_server-0.3.0-py3-none-any.whl","archive_info":{}}'
    monkeypatch.setattr(service.importlib.metadata, "distribution", lambda _: fake_dist)
    assert service._get_install_args() == ["/tmp/App Support/mlx_speech_server-0.3.0-py3-none-any.whl"]


def test_current_runtime_plist_supports_xml_path_characters(fake_paths, monkeypatch):
    import plistlib
    runner = fake_paths / "runtime & tools/bin/mlx-run"
    runner.parent.mkdir(parents=True)
    runner.write_text("#!/bin/sh\n")
    runner.chmod(0o755)
    monkeypatch.setattr(service.sys, "executable", str(runner.parent / "python"))
    service.install(use_current_runtime=True)
    assert plistlib.loads(service.PLIST_PATH.read_bytes())["ProgramArguments"] == [str(runner)]


@pytest.mark.parametrize("host,connect,url_host", [
    ("127.0.0.2", "127.0.0.2", "127.0.0.2"),
    ("0.0.0.0", "127.0.0.1", "127.0.0.1"),
    ("::", "::1", "[::1]"),
    ("::1", "::1", "[::1]"),
    ("[::1]", "::1", "[::1]"),
])
def test_probes_use_configured_connect_host(installed, host, connect, url_host):
    service.CONFIG_ENV.parent.mkdir(parents=True)
    content = f"WHISPER_HOST={host}\nWHISPER_PORT=49000\n"
    service.CONFIG_ENV.write_text(content)
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b'{"status":"ok"}'
    with patch.object(service.socket, "create_connection") as tcp, \
         patch.object(service.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "pid = 321")), \
         patch.object(service.urllib.request, "urlopen", return_value=response) as http:
        assert service._port_in_use(49000)
        assert service.get_status()["health"] == {"status": "ok"}
    tcp.assert_called_once_with((connect, 49000), timeout=0.5)
    assert [call.args[0] for call in http.call_args_list] == [
        f"http://{url_host}:49000/health", f"http://{url_host}:49000/v1/queue/stats"]
    assert service.CONFIG_ENV.read_text() == content
