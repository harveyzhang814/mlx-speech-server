import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, PropertyMock
from app.api.queue import create_queue_router


def _make_client(queue_size: int = 2, active: bool = True, max_size: int = 10) -> TestClient:
    worker = MagicMock()
    type(worker).queue_size = PropertyMock(return_value=queue_size)
    type(worker).active = PropertyMock(return_value=active)
    type(worker).max_size = PropertyMock(return_value=max_size)
    app = FastAPI()
    app.include_router(create_queue_router(worker))
    return TestClient(app)


def test_queue_stats_returns_correct_fields():
    client = _make_client(queue_size=2, active=True, max_size=10)
    resp = client.get("/v1/queue/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["queue_size"] == 2
    assert data["queue_max_size"] == 10
    assert data["active"] is True


def test_queue_stats_idle():
    client = _make_client(queue_size=0, active=False, max_size=10)
    resp = client.get("/v1/queue/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["queue_size"] == 0
    assert data["active"] is False
