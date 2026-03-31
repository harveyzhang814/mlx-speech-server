import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.config import ServerConfig
from app.registry import ModelRegistry
from app.worker import InferenceWorker
from app.server import create_app


def _make_test_app():
    config = ServerConfig(model_path="mlx-community/whisper-large-v3-turbo", memory_cleanup_interval=2)
    registry = ModelRegistry()
    worker = InferenceWorker(max_size=5, timeout=10.0)
    return create_app(config, registry, worker), worker


def test_health_endpoint():
    app, worker = _make_test_app()
    try:
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model" in data
    finally:
        worker.stop()


def test_unknown_route_returns_404():
    app, worker = _make_test_app()
    try:
        client = TestClient(app)
        resp = client.get("/nonexistent")
        assert resp.status_code == 404
    finally:
        worker.stop()


def test_metal_cleanup_triggered_at_interval():
    with patch("app.server.mx") as mock_mx, patch("app.server.gc") as mock_gc:
        config = ServerConfig(memory_cleanup_interval=2)
        registry = ModelRegistry()
        worker = InferenceWorker(max_size=5, timeout=10.0)
        app = create_app(config, registry, worker)
        try:
            client = TestClient(app)
            client.get("/health")
            client.get("/health")
            assert mock_mx.clear_cache.call_count >= 1
        finally:
            worker.stop()
