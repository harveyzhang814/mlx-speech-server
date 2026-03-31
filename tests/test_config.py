import os
import pytest
from app.config import ServerConfig


def test_default_values():
    config = ServerConfig()
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.model_path == "mlx-community/whisper-large-v3-turbo"
    assert config.quantize is None
    assert config.memory_cleanup_interval == 20
    assert config.queue_max_size == 10
    assert config.queue_timeout == 300.0
    assert config.log_level == "info"


def test_from_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("WHISPER_PORT", "9000")
    monkeypatch.setenv("WHISPER_MODEL_PATH", "/local/whisper")
    monkeypatch.setenv("WHISPER_QUANTIZE", "4")
    monkeypatch.setenv("WHISPER_QUEUE_MAX_SIZE", "5")
    monkeypatch.setenv("WHISPER_QUEUE_TIMEOUT", "60.0")
    monkeypatch.setenv("WHISPER_MEMORY_CLEANUP_INTERVAL", "10")
    monkeypatch.setenv("WHISPER_LOG_LEVEL", "debug")

    config = ServerConfig.from_env()
    assert config.port == 9000
    assert config.model_path == "/local/whisper"
    assert config.quantize == 4
    assert config.queue_max_size == 5
    assert config.queue_timeout == 60.0
    assert config.memory_cleanup_interval == 10
    assert config.log_level == "debug"


def test_from_env_uses_defaults_when_vars_absent(monkeypatch):
    for key in ["WHISPER_PORT", "WHISPER_MODEL_PATH", "WHISPER_QUANTIZE"]:
        monkeypatch.delenv(key, raising=False)
    config = ServerConfig.from_env()
    assert config.port == 8000
    assert config.model_path == "mlx-community/whisper-large-v3-turbo"
    assert config.quantize is None
