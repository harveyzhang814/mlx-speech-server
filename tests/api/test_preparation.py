"""Protocol tests use real handler/worker state, replacing only native weight I/O."""

import asyncio
import importlib
from contextlib import asynccontextmanager
from threading import Event
from unittest.mock import patch

import httpx
import mlx.core as mx
import pytest

from app.config import ServerConfig
from app.handlers.whisper import WhisperHandler
from app.registry import ModelRegistry
from app.server import create_app
from app.worker import InferenceWorker


class Model:
    def parameters(self):
        return {"weight": mx.array([1.0])}


@asynccontextmanager
async def preparation_client(*, fail=False, blocked=False, download=None):
    holder = importlib.import_module("mlx_whisper.transcribe").ModelHolder
    worker = InferenceWorker(max_size=5, timeout=5)
    handler = WhisperHandler("configured/model", worker)
    registry = ModelRegistry()
    registry.register("model", handler)
    config = ServerConfig(model_path="configured/model", memory_cleanup_interval=1000)
    app = create_app(config, registry, worker)
    entered, release, calls = Event(), Event(), []
    if not blocked:
        release.set()

    def load(path, dtype):
        calls.append((path, dtype))
        entered.set()
        assert release.wait(3)
        if fail:
            raise RuntimeError("missing weights")
        holder.model = Model()
        holder.model_path = path
        return holder.model

    with (
        patch.object(holder, "model", None),
        patch.object(holder, "model_path", None),
        patch("app.preparation.snapshot_download", side_effect=download,
              return_value="configured/model"),
        patch.object(holder, "get_model", side_effect=load),
        patch(
            "app.handlers.whisper.mlx_whisper.transcribe", side_effect=AssertionError("inference")
        ),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            try:
                yield client, handler, holder, entered, release, calls
            finally:
                release.set()
                if handler._preparation_task is not None:
                    await handler._preparation_task
                worker.stop()


async def poll_terminal(client):
    async with asyncio.timeout(2):
        while True:
            response = await client.get("/v1/models/model/preparation")
            assert response.status_code == 200
            data = response.json()
            if data["phase"] in {"ready", "failed"}:
                return data
            await asyncio.sleep(0.005)


async def test_prepare_polling_reaches_ready():
    async with preparation_client(blocked=True) as (
        client,
        handler,
        holder,
        entered,
        release,
        calls,
    ):
        caps = (await client.get("/v1/preparation/capabilities")).json()
        assert caps["protocol_version"] == 1
        assert caps["configured_model_id"] == "model"
        assert caps["explicit_prepare"] is True
        assert caps["server_version"]
        assert caps["instance_id"]
        response = await asyncio.wait_for(client.post("/v1/models/model/prepare"), timeout=1)
        assert response.status_code == 202
        loading = response.json()
        assert set(loading) == {
            "instance_id",
            "operation_id",
            "model_id",
            "phase",
            "loaded",
            "progress",
            "downloaded_bytes",
            "error",
        }
        assert loading["instance_id"] == caps["instance_id"]
        assert loading["operation_id"]
        assert loading["phase"] in {"idle", "loading"}
        assert loading["loaded"] is False
        assert loading["progress"] is None
        assert loading["error"] is None
        assert await asyncio.to_thread(entered.wait, 1)
        release.set()
        ready = await poll_terminal(client)
        assert ready["phase"] == "ready"
        assert ready["loaded"] is True
        assert ready["model_id"] == "model"
        assert ready["operation_id"] == loading["operation_id"]
        assert ready["instance_id"] == caps["instance_id"]
        assert holder.model is not None
        again = await client.post("/v1/models/model/prepare")
        assert again.status_code == 200
        assert again.json() == ready
        assert calls == [("configured/model", mx.float16)]


@pytest.mark.parametrize("model_id", ["unconfigured", "another/remote-model", "configured/model"])
async def test_unknown_model_cannot_load_arbitrary_path(model_id):
    async with preparation_client() as (client, handler, holder, entered, release, calls):
        for method, suffix in [("POST", "prepare"), ("GET", "preparation")]:
            response = await client.request(method, f"/v1/models/{model_id}/{suffix}")
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "model_mismatch"
        assert calls == []
        assert holder.model is None
        assert handler.preparation_status().phase == "idle"


async def test_concurrent_posts_share_operation():
    async with preparation_client(blocked=True) as (
        client,
        handler,
        holder,
        entered,
        release,
        calls,
    ):
        responses = await asyncio.gather(
            *(client.post("/v1/models/model/prepare") for _ in range(8))
        )
        assert all(response.status_code == 202 for response in responses)
        operation_ids = {response.json()["operation_id"] for response in responses}
        assert len(operation_ids) == 1
        assert None not in operation_ids
        assert await asyncio.to_thread(entered.wait, 1)
        release.set()
        assert (await poll_terminal(client))["loaded"] is True
        assert len(calls) == 1


async def test_idle_release_invalidates_loaded():
    async with preparation_client() as (client, handler, holder, entered, release, calls):
        assert (await client.post("/v1/models/model/prepare")).status_code == 202
        ready = await poll_terminal(client)
        # Run the existing idle-unload action under its actual worker lock.
        handler._last_used = 0
        await handler._worker.submit("idle-unload", handler._unload_if_idle)
        idle = (await client.get("/v1/models/model/preparation")).json()
        assert idle["loaded"] is False
        assert idle["phase"] == "idle"
        assert idle["instance_id"] == ready["instance_id"]
        assert holder.model is None
        again = await client.post("/v1/models/model/prepare")
        assert again.status_code == 202
        assert again.json()["operation_id"] != ready["operation_id"]
        assert (await poll_terminal(client))["loaded"] is True
        assert len(calls) == 2


async def test_failed_load_is_pollable_and_retryable():
    async with preparation_client(fail=True) as (client, handler, holder, entered, release, calls):
        assert (await client.post("/v1/models/model/prepare")).status_code == 202
        failed = await poll_terminal(client)
        assert failed["loaded"] is False
        assert failed["progress"] is None
        assert failed["error"] == {"code": "model_load_failed", "message": "missing weights"}
        assert holder.model is None
        response = await client.post("/v1/models/model/prepare")
        assert response.status_code == 202
        assert response.json()["operation_id"] != failed["operation_id"]
        await poll_terminal(client)


async def test_unexpected_background_error_is_captured():
    async with preparation_client() as (client, handler, holder, entered, release, calls):

        async def broken_prepare():
            raise RuntimeError("unexpected preparation error")

        with patch.object(handler, "prepare", broken_prepare):
            assert (await client.post("/v1/models/model/prepare")).status_code == 202
            failed = await poll_terminal(client)
            assert failed["loaded"] is False
            assert failed["error"] == {
                "code": "model_load_failed",
                "message": "unexpected preparation error",
            }


@pytest.mark.parametrize("code", ["queue_full", "queue_timeout"])
async def test_admission_errors_remain_distinct_from_load_errors(code):
    async with preparation_client() as (client, handler, holder, entered, release, calls):
        busy_entered, busy_release = Event(), Event()

        def occupy_worker():
            busy_entered.set()
            assert busy_release.wait(3)

        busy = asyncio.create_task(handler._worker.submit("busy", occupy_worker))
        try:
            assert await asyncio.to_thread(busy_entered.wait, 1)
            handler._worker._max_size = 1 if code == "queue_full" else 2
            handler._worker._timeout = 0.03
            response = await client.post("/v1/models/model/prepare")
            assert response.status_code == 202
            failed = await poll_terminal(client)
            assert failed["loaded"] is False
            assert failed["error"]["code"] == code
            assert calls == []
        finally:
            busy_release.set()
            await busy


async def test_background_failure_recovers_after_audio_inference(tmp_wav_file):
    async with preparation_client() as (client, handler, holder, entered, release, calls):

        async def broken_prepare():
            raise RuntimeError("unexpected preparation error")

        with patch.object(handler, "prepare", broken_prepare):
            assert (await client.post("/v1/models/model/prepare")).status_code == 202
            assert (await poll_terminal(client))["phase"] == "failed"

        def transcribe(*args, **kwargs):
            holder.get_model("configured/model", mx.float16)
            return {"text": "recovered", "language": "en", "segments": []}

        with patch("app.handlers.whisper.mlx_whisper.transcribe", side_effect=transcribe):
            response = await client.post(
                "/v1/audio/transcriptions",
                data={"model": "model"},
                files={"file": ("audio.wav", tmp_wav_file.read_bytes(), "audio/wav")},
            )
        assert response.status_code == 200
        assert response.json() == {"text": "recovered"}
        assert holder.model is not None
        status = (await client.get("/v1/models/model/preparation")).json()
        assert status["phase"] == "ready"
        assert status["loaded"] is True
        assert status["error"] is None
        assert (await client.post("/v1/models/model/prepare")).status_code == 200


async def test_background_failure_does_not_mask_new_handler_operation():
    async with preparation_client(blocked=True) as (
        client,
        handler,
        holder,
        entered,
        release,
        calls,
    ):

        async def broken_prepare():
            raise RuntimeError("unexpected preparation error")

        with patch.object(handler, "prepare", broken_prepare):
            assert (await client.post("/v1/models/model/prepare")).status_code == 202
            failed = await poll_terminal(client)
            assert failed["phase"] == "failed"

        # An independent handler consumer starts a new operation without another
        # API POST clearing the router's background failure.
        preparing = asyncio.create_task(handler.prepare())
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            status = (await client.get("/v1/models/model/preparation")).json()
            assert status["phase"] == "loading"
            assert status["loaded"] is False
            assert status["error"] is None
            assert status["operation_id"] != failed["operation_id"]
        finally:
            release.set()
            await preparing
        assert (await poll_terminal(client))["phase"] == "ready"


async def test_capabilities_publish_actual_configured_path():
    async with preparation_client() as (client, *_):
        caps = (await client.get("/v1/preparation/capabilities")).json()
        assert caps["configured_model_id"] == "model"
        assert caps["configured_model_path"] == "configured/model"


async def test_api_publishes_actual_download_bytes_before_separate_loading_phase():
    downloading, acquired = Event(), Event()

    def download(*, repo_id, tqdm_class):
        assert repo_id == "configured/model"
        with tqdm_class(unit="B", total=0, disable=True) as bar:
            bar.update(100)
            downloading.set()
            assert acquired.wait(3)
        return "configured/model"

    async with preparation_client(blocked=True, download=download) as (
        client, handler, holder, loading, release, calls
    ):
        task = asyncio.create_task(client.post("/v1/models/model/prepare"))
        try:
            assert await asyncio.to_thread(downloading.wait, 1)
            response = await task
            assert response.status_code == 202
            first = (await client.get("/v1/models/model/preparation")).json()
            assert first["phase"] == "downloading"
            assert first["downloaded_bytes"] == 100
            assert first["progress"] is None
            assert first["loaded"] is False
            assert (await client.get("/v1/models/model/preparation")).json() == first
            acquired.set()
            assert await asyncio.to_thread(loading.wait, 1)
            state = (await client.get("/v1/models/model/preparation")).json()
            assert state["phase"] == "loading"
            assert state["downloaded_bytes"] == 100
            assert state["operation_id"] == first["operation_id"]
        finally:
            acquired.set()
            release.set()
        ready = await poll_terminal(client)
        assert ready["loaded"] is True
        assert ready["downloaded_bytes"] == 100
