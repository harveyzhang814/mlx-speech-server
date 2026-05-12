import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from app.api.audio import create_audio_router
from app.registry import ModelRegistry
from app.handlers.base import BaseHandler, AudioCapable
from app.schemas.common import ModelCard
from app.schemas.audio import TranscriptionParams, TranscriptionResult, SegmentResult
from app.worker import TaskNotFoundError
from typing import AsyncGenerator


def _delete_batch(client: TestClient, task_ids: list):
    return client.request("DELETE", "/v1/audio/transcriptions", json={"task_ids": task_ids})

FAKE_RESULT = TranscriptionResult(
    text=" Hello",
    language="en",
    duration=1.0,
    segments=[SegmentResult(id=0, start=0.0, end=1.0, text=" Hello", no_speech_prob=0.0)],
)


class FakeAudioHandler(BaseHandler, AudioCapable):
    async def initialize(self) -> None: pass
    async def cleanup(self) -> None: pass
    def model_info(self) -> ModelCard:
        return ModelCard(id="whisper-large-v3-turbo")
    async def transcribe(self, audio_path: Path, params: TranscriptionParams, task_id: str) -> TranscriptionResult:
        return FAKE_RESULT
    async def transcribe_stream(self, audio_path: Path, params: TranscriptionParams, task_id: str) -> AsyncGenerator[str, None]:
        yield 'data: {"text": " Hello"}\n\n'
        yield "data: [DONE]\n\n"


def _make_client(cancel_side_effect=None) -> TestClient:
    registry = ModelRegistry()
    registry.register("whisper-large-v3-turbo", FakeAudioHandler())
    worker = MagicMock()
    worker.cancel = MagicMock(side_effect=cancel_side_effect)
    app = FastAPI()
    app.include_router(create_audio_router(registry, worker))
    return TestClient(app)


def _make_client_with_per_id_cancel(results: dict) -> TestClient:
    """results: {task_id: Exception | None} — None means success."""
    registry = ModelRegistry()
    registry.register("whisper-large-v3-turbo", FakeAudioHandler())
    worker = MagicMock()

    def _cancel(task_id: str):
        exc = results.get(task_id)
        if exc is not None:
            raise exc

    worker.cancel = MagicMock(side_effect=_cancel)
    app = FastAPI()
    app.include_router(create_audio_router(registry, worker))
    return TestClient(app)


def test_cancel_known_task_returns_200():
    client = _make_client()
    resp = client.delete("/v1/audio/transcriptions/some-task-id")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "cancelled"
    assert data["task_id"] == "some-task-id"


def test_cancel_calls_worker_cancel_with_task_id():
    registry = ModelRegistry()
    registry.register("whisper-large-v3-turbo", FakeAudioHandler())
    worker = MagicMock()
    app = FastAPI()
    app.include_router(create_audio_router(registry, worker))
    client = TestClient(app)

    client.delete("/v1/audio/transcriptions/my-task-id")
    worker.cancel.assert_called_once_with("my-task-id")


def test_cancel_unknown_task_id_returns_404():
    client = _make_client(cancel_side_effect=TaskNotFoundError("not found"))
    resp = client.delete("/v1/audio/transcriptions/bad-id")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "task_not_found"
    assert body["error"]["type"] == "invalid_request_error"


# ── batch cancel ──────────────────────────────────────────────────────────────

def test_batch_cancel_all_known_returns_200():
    client = _make_client_with_per_id_cancel({"id1": None, "id2": None, "id3": None})
    resp = _delete_batch(client, ["id1", "id2", "id3"])
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["cancelled"]) == {"id1", "id2", "id3"}
    assert data["not_found"] == []


def test_batch_cancel_partial_not_found_returns_200():
    client = _make_client_with_per_id_cancel({
        "id1": None,
        "id2": TaskNotFoundError("gone"),
        "id3": None,
    })
    resp = _delete_batch(client, ["id1", "id2", "id3"])
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["cancelled"]) == {"id1", "id3"}
    assert data["not_found"] == ["id2"]


def test_batch_cancel_all_not_found_returns_200():
    client = _make_client_with_per_id_cancel({
        "id1": TaskNotFoundError("gone"),
        "id2": TaskNotFoundError("gone"),
    })
    resp = _delete_batch(client, ["id1", "id2"])
    assert resp.status_code == 200
    data = resp.json()
    assert data["cancelled"] == []
    assert set(data["not_found"]) == {"id1", "id2"}


def test_batch_cancel_calls_worker_for_each_task_id():
    registry = ModelRegistry()
    registry.register("whisper-large-v3-turbo", FakeAudioHandler())
    worker = MagicMock()
    app = FastAPI()
    app.include_router(create_audio_router(registry, worker))
    client = TestClient(app)

    _delete_batch(client, ["a", "b", "c"])
    assert worker.cancel.call_count == 3
    called_ids = {call.args[0] for call in worker.cancel.call_args_list}
    assert called_ids == {"a", "b", "c"}


def test_batch_cancel_empty_list_returns_200():
    client = _make_client()
    resp = _delete_batch(client, [])
    assert resp.status_code == 200
    data = resp.json()
    assert data["cancelled"] == []
    assert data["not_found"] == []
