import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from app.api.models import create_models_router
from app.schemas.common import ModelCard


def _make_client() -> TestClient:
    registry = MagicMock()
    registry.list_models.return_value = [ModelCard(id="whisper-large-v3-turbo")]
    app = FastAPI()
    app.include_router(create_models_router(registry))
    return TestClient(app)


def test_list_models_returns_openai_format():
    client = _make_client()
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "whisper-large-v3-turbo"
    assert data["data"][0]["object"] == "model"


def test_list_models_empty_registry():
    registry = MagicMock()
    registry.list_models.return_value = []
    app = FastAPI()
    app.include_router(create_models_router(registry))
    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"] == []
