"""Idempotent bootstrap tests: never operate on the user's service paths."""
import json
import multiprocessing
import socket
import subprocess
import sys
import time
from unittest.mock import MagicMock
from pathlib import Path

import pytest

from app import service

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS")


@pytest.fixture(autouse=True)
def isolated_service(tmp_path, monkeypatch):
    for name, path in {
        "CONFIG_DIR": tmp_path / "config", "CONFIG_ENV": tmp_path / "config/config.env",
        "PLIST_PATH": tmp_path / "agents/server.plist", "VENV_DIR": tmp_path / "venv",
        "LOG_DIR": tmp_path / "logs", "STDOUT_LOG": tmp_path / "logs/out",
        "STDERR_LOG": tmp_path / "logs/err",
    }.items():
        monkeypatch.setattr(service, name, path)
    # Any unconfigured attempt to invoke a real service command fails the test.
    monkeypatch.setattr(service.subprocess, "run", MagicMock(side_effect=AssertionError("real command")))
    monkeypatch.setattr(service, "_port_in_use", lambda port: False, raising=False)


def status(running=True, health=None, loaded=True):
    return {"installed": True, "loaded": loaded, "running": running,
            "pid": 321 if running else None, "port": 47300,
            "health": {"status": "ok"} if health is None else health}


def test_running_service_is_not_restarted(monkeypatch):
    monkeypatch.setattr(service, "get_status", lambda: status())
    result = service.ensure_running()
    assert result == {"running": True, "pid": 321, "port": 47300,
                      "health": {"status": "ok"}, "service_label": service.SERVICE_LABEL}
    service.subprocess.run.assert_not_called()


def test_nonhealthy_running_service_is_reported(monkeypatch):
    monkeypatch.setattr(service, "get_status", lambda: status(health={"status": "failed"}))
    result = service.ensure_running()
    assert result["running"] is True
    assert result["error"]["code"] == "unhealthy_service"
    service.subprocess.run.assert_not_called()


def test_other_listener_not_replaced(monkeypatch):
    monkeypatch.setattr(service, "get_status", lambda: {"installed": False})
    monkeypatch.setattr(service, "_port_in_use", lambda port: True)
    result = service.ensure_running()
    assert result["running"] is False
    assert result["error"]["code"] == "port_in_use"
    assert not service.CONFIG_ENV.exists()
    service.subprocess.run.assert_not_called()


def test_port_probe_detects_non_http_listener(monkeypatch):
    # A TCP listener that never speaks HTTP must still block startup.
    monkeypatch.undo()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        assert service._port_in_use(listener.getsockname()[1]) is True


def test_installed_unloaded_service_bootstraps_without_restart(monkeypatch):
    states = iter([status(False, loaded=False), status()])
    monkeypatch.setattr(service, "get_status", lambda: next(states))
    calls = []
    monkeypatch.setattr(service.subprocess, "run", lambda args, **kw: calls.append((args, kw)))
    result = service.ensure_running()
    assert result["running"] is True
    assert [args[1] for args, _ in calls] == ["bootstrap"]
    assert calls[0][1]["check"] is True


def test_loaded_stopped_job_kickstarts_without_force(monkeypatch):
    states = iter([status(False), status()])
    monkeypatch.setattr(service, "get_status", lambda: next(states))
    calls = []
    monkeypatch.setattr(service.subprocess, "run", lambda args, **kw: calls.append(args))
    assert service.ensure_running()["running"] is True
    assert calls[0][1] == "kickstart"
    assert "-k" not in calls[0]


def test_start_failure_is_structured(monkeypatch):
    monkeypatch.setattr(service, "get_status", lambda: status(False, loaded=False))
    monkeypatch.setattr(service.subprocess, "run", MagicMock(
        side_effect=subprocess.CalledProcessError(5, ["launchctl", "bootstrap"])))
    assert service.ensure_running()["error"]["code"] == "startup_failed"


def test_start_timeout_does_not_retry_restart(monkeypatch):
    monkeypatch.setattr(service, "get_status", lambda: status(False, loaded=False))
    monkeypatch.setattr(service, "STARTUP_TIMEOUT", 0, raising=False)
    calls = []
    monkeypatch.setattr(service.subprocess, "run", lambda args, **kw: calls.append(args))
    result = service.ensure_running()
    assert result["error"]["code"] == "startup_timeout"
    assert len(calls) == 1


def test_first_install_receives_model_and_port(monkeypatch):
    states = iter([{"installed": False}, status()])
    monkeypatch.setattr(service, "get_status", lambda: next(states))
    installs = []
    monkeypatch.setattr(service, "install", lambda **kw: installs.append(kw))
    monkeypatch.setattr(service.subprocess, "run", lambda *args, **kw: None)
    service.ensure_running(model="mlx-community/whisper-small-mlx", port=47301)
    assert installs == [{"model": "mlx-community/whisper-small-mlx", "port": 47301}]


def _parallel_ensure(root, gate, queue):
    """Fresh process simulates launchctl using only files in its test directory."""
    root = Path(root)
    service.CONFIG_DIR = root / "config"
    service.CONFIG_ENV = root / "config/config.env"
    service.PLIST_PATH = root / "agents/server.plist"
    service.LOG_DIR = root / "logs"
    service._port_in_use = lambda port: False
    started = root / "started"
    calls = root / "calls"
    def get_status():
        return status() if started.exists() else status(False, loaded=False)
    def run(args, **kw):
        with calls.open("a") as output:
            output.write(json.dumps(args) + "\n")
        time.sleep(0.05)
        started.touch()
    service.get_status = get_status
    service.subprocess.run = run
    gate.wait(timeout=5)
    queue.put(service.ensure_running())


def test_parallel_ensure_starts_once(tmp_path):
    # Fresh interpreters share only the filesystem; a Python/thread lock cannot pass.
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    gate = ctx.Barrier(4)
    clients = [ctx.Process(target=_parallel_ensure, args=(str(tmp_path), gate, queue))
               for _ in range(4)]
    try:
        for client in clients:
            client.start()
        for client in clients:
            client.join(10)
            assert not client.is_alive(), "ensure-running hung"
            assert client.exitcode == 0
        assert all(queue.get(timeout=1)["running"] for _ in clients)
        assert len((tmp_path / "calls").read_text().splitlines()) == 1
    finally:
        for client in clients:
            if client.is_alive():
                client.terminate()
                client.join()
        queue.close()
        queue.join_thread()


def test_first_install_can_use_locked_current_runtime(monkeypatch):
    states = iter([{"installed": False}, status()])
    monkeypatch.setattr(service, "get_status", lambda: next(states))
    installs = []
    monkeypatch.setattr(service, "install", lambda **kw: installs.append(kw))
    monkeypatch.setattr(service.subprocess, "run", lambda *args, **kw: None)
    service.ensure_running(use_current_runtime=True)
    assert installs == [{"model": None, "port": None, "use_current_runtime": True}]


def test_existing_service_ignores_runtime_and_config_overrides(monkeypatch):
    service.CONFIG_DIR.mkdir()
    service.CONFIG_ENV.write_text("WHISPER_PORT=47300\n")
    monkeypatch.setattr(service, "get_status", lambda: status())
    assert service.ensure_running(model="ignored", port=47301,
                                  use_current_runtime=True)["port"] == 47300
    assert service.CONFIG_ENV.read_text() == "WHISPER_PORT=47300\n"


@pytest.mark.parametrize("health", [{"status": "ok"}, None])
def test_running_job_http_probe_never_mutates_launchd(monkeypatch, health):
    service.PLIST_PATH.parent.mkdir(parents=True)
    service.PLIST_PATH.write_text("existing plist")
    commands = []
    def run(args, **kw):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="  pid = 321\n")
    def http(url, **kw):
        if health is None:
            raise OSError("HTTP not available")
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps(health).encode()
        return response
    monkeypatch.setattr(service.subprocess, "run", run)
    monkeypatch.setattr(service.urllib.request, "urlopen", http)
    result = service.ensure_running()
    assert result["pid"] == 321
    assert bool(result.get("error")) is (health is None)
    assert commands == [["launchctl", "print", f"gui/{service.os.getuid()}/{service.SERVICE_LABEL}"]]


def test_listener_appearing_during_install_blocks_launch(monkeypatch):
    monkeypatch.setattr(service, "get_status", lambda: {"installed": False})
    probes = iter([False, True])
    monkeypatch.setattr(service, "_port_in_use", lambda port: next(probes))
    monkeypatch.setattr(service, "install", lambda **kw: None)
    result = service.ensure_running()
    assert result["error"]["code"] == "port_in_use"
    service.subprocess.run.assert_not_called()
