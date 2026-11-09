"""Opt-in, real Metal/model/HTTP acceptance; never contacts a shared service.

Run: python -m pytest tests/e2e/test_explicit_preparation.py -m e2e -v
Uses an existing local Whisper snapshot only (no downloads). Override it with
MLX_E2E_MODEL_PATH; missing model/Metal/speech tools means SKIPPED, not passed.
The child adds an observation-only route; all loading/inference stays real.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import httpx
import pytest

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[2]

# This observer runs only in a disposable subprocess, never the product server.
# Wrappers call the original implementations without sleeping or substituting data.
_SERVER = r'''
import json, os, time
from importlib import import_module
from pathlib import Path
import uvicorn
from app.config import ServerConfig
from app.registry import ModelRegistry
from app.server import create_app
from app.worker import InferenceWorker, QueueTimeoutError
from app.preparation import WhisperModelAdapter

config = ServerConfig(**json.loads(Path(os.environ["MLX_E2E_CONFIG"]).read_text()))
worker = InferenceWorker(config.queue_max_size, config.queue_timeout)
registry = ModelRegistry()
module = import_module("mlx_whisper.transcribe")
observations = {"weight_loads": 0, "native_started": None, "native_finished": None,
                "timeouts": []}
real_load = module.load_model
def load_model(*args, **kwargs):
    observations["weight_loads"] += 1
    return real_load(*args, **kwargs)
module.load_model = load_model
real_prepare = WhisperModelAdapter.load
def prepare(self, *args, **kwargs):
    observations["native_started"] = time.monotonic()
    try:
        return real_prepare(self, *args, **kwargs)
    finally:
        observations["native_finished"] = time.monotonic()
WhisperModelAdapter.load = prepare
real_submit = worker.submit
async def submit(*args, **kwargs):
    try:
        return await real_submit(*args, **kwargs)
    except QueueTimeoutError:
        observations["timeouts"].append({"active": worker.active,
            "active_status": worker.active_status, "at": time.monotonic(),
            "native_finished": observations["native_finished"]})
        raise
worker.submit = submit
app = create_app(config, registry, worker)
@app.get("/__e2e/observations")
async def observe():
    model = module.ModelHolder.model
    return {**observations, "model_object": id(model) if model is not None else None,
            "holder_path": module.ModelHolder.model_path, "pid": os.getpid()}
uvicorn.run(app, host=config.host, port=config.port, log_level="warning")
'''


def _shared_state():
    """Fingerprint shared files and known service PIDs without reading config values."""
    files = [Path.home() / ".config/mlx-speech-server/config.env",
             Path.home() / "Library/LaunchAgents/com.local.mlx-speech-server.plist"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
              for p in files}
    processes = subprocess.check_output(["ps", "-axo", "pid=,comm="], text=True)
    pids = {}
    for line in processes.splitlines():
        pid, command = line.strip().split(None, 1)
        if any(name in command.lower() for name in ("mlx-speech", "mlx-run", "uvicorn", "ollama")):
            pids[int(pid)] = command
    # Also cover a default-port server whose executable is named simply Python.
    listener = subprocess.run(["/usr/sbin/lsof", "-tiTCP:47300", "-sTCP:LISTEN"],
                              text=True, capture_output=True, check=False)
    for pid in listener.stdout.split():
        pids[int(pid)] = "default-port listener"
    return hashes, pids


@pytest.fixture(scope="module")
def local_model():
    if sys.platform != "darwin":
        pytest.skip("real MLX E2E requires Apple Silicon macOS")
    import mlx.core as mx
    if not mx.metal.is_available():
        pytest.skip("Metal unavailable; real model loading was not verified")
    configured = os.environ.get("MLX_E2E_MODEL_PATH")
    if configured:
        model = Path(configured).expanduser().absolute()
    else:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        try:
            model = Path(snapshot_download("mlx-community/whisper-large-v3-turbo",
                                           local_files_only=True))
        except LocalEntryNotFoundError:
            pytest.skip("no cached Whisper snapshot; set MLX_E2E_MODEL_PATH")
    if not (model / "config.json").is_file() or not any(model.glob("*.safetensors")):
        pytest.skip(f"incomplete local model snapshot: {model}")
    return str(model)


@pytest.fixture(scope="module")
def speech_wav(tmp_path_factory):
    """Generate non-private speech locally, without fetching an audio recording."""
    if not shutil.which("say") or not shutil.which("ffmpeg"):
        pytest.skip("generated speech fixture requires macOS say and ffmpeg")
    directory = tmp_path_factory.mktemp("public-generated-speech")
    aiff, wav = directory / "speech.aiff", directory / "speech.wav"
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff),
                    "The quick brown fox jumps over the lazy dog."], check=True,
                   capture_output=True, timeout=30)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(aiff), "-ar", "16000",
                    "-ac", "1", str(wav)], check=True, capture_output=True, timeout=30)
    return wav


@contextmanager
def _server(directory, model, timeout):
    before_hashes, original_pids = _shared_state()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    config = directory / "config.json"
    config.write_text(json.dumps({"host": "127.0.0.1", "port": port,
                                  "model_path": model, "queue_timeout": timeout}))
    env = {**os.environ, "MLX_E2E_CONFIG": str(config), "HF_HUB_OFFLINE": "1",
           "HF_HOME": str(directory / "hf-cache")}
    with (directory / "server.log").open("w+") as log:
        process = subprocess.Popen([sys.executable, "-u", "-c", _SERVER], cwd=ROOT,
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120,
                              trust_env=False) as client:
                deadline = time.monotonic() + 30
                while True:
                    assert process.poll() is None, "owned server exited during startup"
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    assert time.monotonic() < deadline, "owned server startup timed out"
                    time.sleep(0.05)
                assert client.get("/__e2e/observations").json()["pid"] == process.pid
                print(f"\nowned PID={process.pid} port={port} model={model} timeout={timeout}")
                yield client
        finally:
            # Only this Popen child is signalled. Shared launchd jobs are untouched.
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            after_hashes, after_pids = _shared_state()
            assert after_hashes == before_hashes, "shared config/plist changed"
            assert all(after_pids.get(pid) == command for pid, command in original_pids.items()), (
                "an original shared service PID changed")
            assert process.poll() is not None
            print(f"owned PID={process.pid} stopped; shared hashes/PIDs unchanged")
            log.seek(0)
            if process.returncode not in (0, -15):
                print(log.read()[-6000:])


def _ready(client, path, operation_id):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        response = client.get(path)
        response.raise_for_status()
        status = response.json()
        assert status["operation_id"] == operation_id
        assert status["phase"] != "failed", status
        if status["phase"] == "ready":
            assert status["loaded"] is True and status["error"] is None
            return status
        assert status["loaded"] is False
        time.sleep(0.01)
    pytest.fail("real model preparation did not reach ready in 120 seconds")


def test_prepare_ready_repeated_prepare_and_original_audio_reuse(tmp_path, local_model, speech_wav):
    with _server(tmp_path, local_model, timeout=120) as client:
        capabilities = client.get("/v1/preparation/capabilities").json()
        assert capabilities["protocol_version"] == 1
        assert capabilities["explicit_prepare"] is True
        assert capabilities["configured_model_path"] == local_model
        model_id = capabilities["configured_model_id"]
        path = f"/v1/models/{model_id}"
        initial = client.get(f"{path}/preparation").json()
        assert initial["phase"] == "idle" and initial["loaded"] is False
        assert client.get("/__e2e/observations").json()["model_object"] is None
        wrong = client.post("/v1/models/not-the-configured-model/prepare")
        assert wrong.status_code == 409 and wrong.json()["error"]["code"] == "model_mismatch"
        accepted = client.post(f"{path}/prepare")
        assert accepted.status_code == 202
        operation = accepted.json()["operation_id"]
        assert operation
        duplicate = client.post(f"{path}/prepare")
        assert duplicate.status_code in (200, 202)
        assert duplicate.json()["operation_id"] == operation
        ready = _ready(client, f"{path}/preparation", operation)
        assert ready["instance_id"] == capabilities["instance_id"]
        before = client.get("/__e2e/observations").json()
        assert before["weight_loads"] == 1 and before["model_object"] is not None
        again = client.post(f"{path}/prepare")
        assert again.status_code == 200 and again.json() == ready
        # Exercise both the original canonical ID and exact configured path alias.
        for audio_model in (model_id, local_model):
            with speech_wav.open("rb") as audio:
                transcription = client.post("/v1/audio/transcriptions", data={
                    "model": audio_model, "language": "en", "response_format": "json"},
                    files={"file": ("generated-speech.wav", audio, "audio/wav")})
            assert transcription.status_code == 200, transcription.text
            assert "quick brown fox" in transcription.json()["text"].lower()
        after = client.get("/__e2e/observations").json()
        assert after["weight_loads"] == 1
        assert after["model_object"] == before["model_object"]
        assert after["holder_path"] == local_model
        assert after["timeouts"] == []
        assert client.get(f"{path}/preparation").json() == ready
        print(f"real preparation/audio reused model object {after['model_object']}; weight_loads=1")


def test_real_loading_timeout_keeps_native_worker_active(tmp_path, local_model):
    with _server(tmp_path, local_model, timeout=0.001) as client:
        capabilities = client.get("/v1/preparation/capabilities").json()
        path = f"/v1/models/{capabilities['configured_model_id']}"
        accepted = client.post(f"{path}/prepare")
        assert accepted.status_code == 202
        operation = accepted.json()["operation_id"]
        deadline = time.monotonic() + 120
        observed_active_after_timeout = False
        while time.monotonic() < deadline:
            observed = client.get("/__e2e/observations").json()
            if observed["timeouts"]:
                # This record is captured inside the timeout handler in one event-loop turn.
                # A second HTTP request may legitimately arrive after native completion.
                timeout = observed["timeouts"][0]
                assert timeout["active"] is True and timeout["active_status"] == "cancelling"
                assert timeout["native_finished"] is None
                observed_active_after_timeout = True
                repeat = client.post(f"{path}/prepare")
                assert repeat.json()["operation_id"] == operation
                break
            if observed["native_finished"] is not None:
                break
            time.sleep(0.005)
        assert observed_active_after_timeout, "did not observe real native work after deadline"
        _ready(client, f"{path}/preparation", operation)
        observed = client.get("/__e2e/observations").json()
        timeout = observed["timeouts"][0]
        assert timeout["active"] is True and timeout["native_finished"] is None
        assert timeout["active_status"] == "cancelling"
        assert observed["native_finished"] > timeout["at"]
        assert observed["weight_loads"] == 1
        deadline = time.monotonic() + 5
        while client.get("/v1/queue/stats").json()["active"]:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert client.post(f"{path}/prepare").status_code == 200
        duration = observed["native_finished"] - observed["native_started"]
        elapsed = timeout["at"] - observed["native_started"]
        print(f"real native load={duration:.4f}s, timeout at {elapsed:.4f}s; "
              "worker stayed active until native completion")
