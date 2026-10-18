import asyncio
import importlib
from dataclasses import asdict
from threading import Event
from unittest.mock import patch

import mlx.core as mx
import pytest

from app.handlers.whisper import WhisperHandler
from app.worker import InferenceWorker


@pytest.fixture
def preparation():
    holder = importlib.import_module("mlx_whisper.transcribe").ModelHolder
    worker = InferenceWorker(max_size=5, timeout=5)
    handler = WhisperHandler("configured/model", worker)
    with patch.object(holder, "model", None), patch.object(holder, "model_path", None):
        yield handler, holder
    worker.stop()


class Model:
    def parameters(self):
        return {"weight": mx.array([1.0])}


def install_loader(holder, calls, entered=None, release=None):
    def load(path, dtype):
        calls.append((path, dtype))
        if entered:
            entered.set()
        if release:
            assert release.wait(3)
        holder.model = Model()
        holder.model_path = path
        return holder.model
    return patch.object(holder, "get_model", side_effect=load)


async def test_prepare_deduplicates(preparation):
    handler, holder = preparation
    calls, entered, release = [], Event(), Event()
    with install_loader(holder, calls, entered, release):
        first = asyncio.create_task(handler.prepare())
        second = asyncio.create_task(handler.prepare())
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            assert handler.preparation_status().phase == "loading"
        finally:
            release.set()
        a, b = await asyncio.gather(first, second)
        assert a == b
        assert a.operation_id
        assert a.phase == "ready"
        assert len(calls) == 1
        assert (await handler.prepare()).operation_id == a.operation_id


async def test_prepare_loads_without_transcription(preparation):
    handler, holder = preparation
    calls = []
    with install_loader(holder, calls), patch(
        "app.handlers.whisper.mlx_whisper.transcribe", side_effect=AssertionError("inference")
    ), patch("mlx.core.eval", wraps=mx.eval) as evaluate:
        ready = await handler.prepare()
    assert ready.loaded is True
    assert ready.phase == "ready"
    assert ready.model_id == "model"
    assert calls == [("configured/model", mx.float16)]
    assert evaluate.call_count == 1
    assert asdict(ready)["error"] is None


async def test_failed_load_not_ready(preparation):
    handler, holder = preparation
    with patch.object(holder, "get_model", side_effect=RuntimeError("missing weights")):
        failed = await handler.prepare()
    assert failed.loaded is False
    assert failed.phase == "failed"
    assert failed.error.code == "model_load_failed"
    assert failed.error.message == "missing weights"
    assert handler.preparation_status() == failed
    with install_loader(holder, []):
        assert (await handler.prepare()).loaded is True


async def test_cancelling_prepare_waiter_does_not_cancel_other_waiter(preparation):
    handler, holder = preparation
    calls, entered, release = [], Event(), Event()
    with install_loader(holder, calls, entered, release):
        first = asyncio.create_task(handler.prepare())
        second = asyncio.create_task(handler.prepare())
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert handler._worker.active
        finally:
            release.set()
        assert (await second).loaded
        assert len(calls) == 1


async def test_idle_unload_invalidates_ready_snapshot(preparation):
    handler, holder = preparation
    with install_loader(holder, []):
        ready = await handler.prepare()
    await handler._worker.submit("idle-unload", handler._unload)
    status = handler.preparation_status()
    assert status.loaded is False
    assert status.phase == "idle"
    assert status.operation_id == ready.operation_id
    assert holder.model is None


async def test_failed_weight_evaluation_does_not_leave_loaded_holder(preparation):
    handler, holder = preparation
    with install_loader(holder, []), patch(
        "mlx.core.eval", side_effect=RuntimeError("invalid weights")
    ):
        failed = await handler.prepare()
    assert failed.loaded is False
    assert failed.error.message == "invalid weights"
    assert holder.model is None


@pytest.mark.parametrize("fails", [False, True])
async def test_native_load_outlives_worker_timeout_without_duplicate(preparation, fails):
    handler, holder = preparation
    handler._worker._timeout = 0.02
    calls, entered, release = [], Event(), Event()
    first = second = None
    with install_loader(holder, calls, entered, release), patch(
        "mlx.core.eval", side_effect=RuntimeError("late failure") if fails else None
    ):
        try:
            first = asyncio.create_task(handler.prepare())
            assert await asyncio.to_thread(entered.wait, 1)
            await asyncio.sleep(0.06)
            assert handler.preparation_status().phase == "loading"
            assert not first.done()
            second = asyncio.create_task(handler.prepare())
            await asyncio.sleep(0.01)
        finally:
            release.set()
        snapshots = await asyncio.gather(first, second)
        assert snapshots[0] == snapshots[1]
        assert len(calls) == 1
        assert snapshots[0].phase == ("failed" if fails else "ready")
        assert snapshots[0].loaded is not fails
        if fails:
            assert snapshots[0].error.message == "late failure"


async def test_prepare_rechecks_actual_holder_identity(preparation):
    handler, holder = preparation
    calls = []
    with install_loader(holder, calls):
        await handler.prepare()
        # Simulate upstream cache replacement on the same serialized resource.
        await handler._worker.submit("replace-holder", setattr, holder, "model", Model())
        await handler.prepare()
    assert len(calls) == 2


async def test_repeated_ready_prepare_reuses_evaluated_weights(preparation):
    handler, holder = preparation
    calls = []
    with install_loader(holder, calls):
        ready = await handler.prepare()
        again = await handler.prepare()
    assert again == ready
    assert len(calls) == 1


@pytest.mark.parametrize("admission", ["queue_full", "queue_timeout"])
@pytest.mark.parametrize("already_ready", [False, True])
async def test_prepare_admission_failure_preserves_model_state(
    preparation, admission, already_ready
):
    from pathlib import Path
    from app.schemas.audio import TranscriptionParams

    handler, holder = preparation
    if already_ready:
        with install_loader(holder, []):
            await handler.prepare()
    before = handler.preparation_status()
    loaded_model = holder.model
    entered, release = Event(), Event()

    def transcribe(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return {"text": "done", "language": "en", "segments": []}

    with patch("app.handlers.whisper.mlx_whisper.transcribe", side_effect=transcribe):
        busy = asyncio.create_task(
            handler.transcribe(Path("not-read.wav"), TranscriptionParams(), "busy")
        )
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            handler._worker._max_size = 1 if admission == "queue_full" else 2
            handler._worker._timeout = 0.02
            result = await handler.prepare()
            assert holder.model is loaded_model
            if already_ready:
                assert result == before
                assert handler.preparation_status() == before
            else:
                assert result.phase == "failed"
                assert result.loaded is False
                assert result.error.code == admission
                assert result.error.message
        finally:
            release.set()
            await busy
