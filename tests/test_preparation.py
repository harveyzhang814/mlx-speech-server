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
    with patch.object(holder, "model", None), patch.object(holder, "model_path", None), patch(
        "app.preparation.snapshot_download", return_value="configured/model"
    ):
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
    # The replacement already resides under the exact configured key; evaluate
    # that object again rather than acquire/load it needlessly.
    assert len(calls) == 1
    assert handler._loaded_model is holder.model


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


def test_prefetch_bytes_then_local_load_preserves_configured_holder_key(tmp_path):
    from app.preparation import WhisperModelAdapter

    holder = importlib.import_module("mlx_whisper.transcribe").ModelHolder
    transcribe_module = importlib.import_module("mlx_whisper.transcribe")
    events, fetches, loads = [], [], []
    model = Model()

    def snapshot(*, repo_id, tqdm_class):
        fetches.append(repo_id)
        with tqdm_class(unit="B", total=0, disable=True) as bar:
            bar.update(100)
            bar.total = 10000  # Metadata totals are incomplete, not a percent/activity update.
            bar.refresh()
            bar.update(50)
        return str(tmp_path)

    def load(path, dtype):
        loads.append((path, dtype))
        return model

    with patch.object(holder, "model", None), patch.object(holder, "model_path", None), patch(
        "app.preparation.snapshot_download", side_effect=snapshot
    ), patch.object(transcribe_module, "load_model", side_effect=load):
        adapter = WhisperModelAdapter()
        assert adapter.load("configured/model", report=lambda *event: events.append(event)) is model
        assert holder.model_path == "configured/model"
        assert holder.get_model("configured/model", mx.float16) is model
        assert adapter.load("configured/model", report=lambda *event: events.append(event)) is model
    assert fetches == ["configured/model"]
    assert loads == [(str(tmp_path), mx.float16)]
    assert events == [("downloading", 0), ("downloading", 100), ("downloading", 150),
                      ("loading", None), ("loading", None)]


def test_existing_local_model_directory_never_fetches(tmp_path):
    from app.preparation import WhisperModelAdapter

    holder = importlib.import_module("mlx_whisper.transcribe").ModelHolder
    with patch.object(holder, "model", None), patch.object(holder, "model_path", None), install_loader(
        holder, []
    ), patch("app.preparation.snapshot_download") as fetch:
        assert WhisperModelAdapter().load(str(tmp_path), report=lambda *_: None) is holder.model
        fetch.assert_not_called()


def test_pinned_hf_public_snapshot_reports_aggregate_bytes_not_file_count(tmp_path):
    from functools import partial
    from types import SimpleNamespace
    from huggingface_hub import snapshot_download
    from app.preparation import WhisperModelAdapter

    module = importlib.import_module("huggingface_hub._snapshot_download")
    holder = importlib.import_module("mlx_whisper.transcribe").ModelHolder
    info = SimpleNamespace(sha="a" * 40, siblings=[SimpleNamespace(rfilename="config.json"),
                                                SimpleNamespace(rfilename="weights.safetensors")])
    events = []

    def download(repo_id, *, filename, tqdm_class, **kwargs):
        with tqdm_class(total=32, initial=0) as bar:
            bar.update(32)
        return str(tmp_path / filename)

    with patch.object(holder, "model", None), patch.object(holder, "model_path", None), install_loader(
        holder, []
    ), patch.object(module, "HfApi") as api, patch.object(
        module, "hf_hub_download", side_effect=download
    ), patch("app.preparation.snapshot_download", side_effect=partial(snapshot_download, cache_dir=tmp_path)):
        api.return_value.repo_info.return_value = info
        WhisperModelAdapter().load("configured/model", report=lambda *event: events.append(event))
    assert events[0] == ("downloading", 0)
    assert [value for phase, value in events if phase == "downloading"] == [0, 32, 64]
    assert events[-1] == ("loading", None)


async def test_download_and_loading_publish_bytes_under_worker_lock_ignore_late_updates(preparation):
    handler, holder = preparation
    downloading, acquire, loading, evaluate = Event(), Event(), Event(), Event()
    captured = []
    calls = []

    def snapshot(*, repo_id, tqdm_class):
        bar = tqdm_class(unit="B", total=0, disable=True)
        captured.append(bar)
        bar.update(100)
        downloading.set()
        assert acquire.wait(3)
        return "configured/model"

    with patch("app.preparation.snapshot_download", side_effect=snapshot), install_loader(
        holder, calls, loading, evaluate
    ):
        task = asyncio.create_task(handler.prepare())
        queued = None
        try:
            assert await asyncio.to_thread(downloading.wait, 1)
            status = handler.preparation_status()
            assert status.phase == "downloading"
            assert status.downloaded_bytes == 100
            assert status.progress is None
            assert handler._worker.active
            queued = asyncio.create_task(handler._worker.submit("after-download", lambda: "done"))
            await asyncio.sleep(0)
            assert not queued.done()
            acquire.set()
            assert await asyncio.to_thread(loading.wait, 1)
            status = handler.preparation_status()
            assert status.phase == "loading"
            captured[0].update(1000)
            assert handler.preparation_status() == status
            assert not queued.done()
        finally:
            acquire.set()
            evaluate.set()
        ready = await task
        assert ready.phase == "ready"
        assert ready.downloaded_bytes == 100
        captured[0].update(1000)
        assert handler.preparation_status() == ready
        await handler._worker.submit("idle-unload", handler._unload)
        idle = handler.preparation_status()
        captured[0].update(1000)
        assert handler.preparation_status() == idle
        if queued is not None:
            assert await queued == "done"


async def test_transcription_reuses_prefetched_holder_without_second_download_or_load(preparation, tmp_path):
    from pathlib import Path
    from app.schemas.audio import TranscriptionParams

    handler, holder = preparation
    module = importlib.import_module("mlx_whisper.transcribe")
    model = Model()

    def transcribe(*args, path_or_hf_repo, **kwargs):
        assert path_or_hf_repo == "configured/model"
        assert holder.get_model(path_or_hf_repo, mx.float16) is model
        return {"text": "reused", "language": "en", "segments": []}

    with patch("app.preparation.snapshot_download", return_value=str(tmp_path)) as fetch, patch.object(
        module, "load_model", return_value=model
    ) as load, patch("app.handlers.whisper.mlx_whisper.transcribe", side_effect=transcribe):
        assert (await handler.prepare()).loaded
        assert holder.model_path == "configured/model"
        assert (await handler.transcribe(Path("not-read.wav"), TranscriptionParams(), "next")).text == "reused"
        assert handler._loaded_model is model
        assert fetch.call_count == 1
        load.assert_called_once_with(str(tmp_path), dtype=mx.float16)


async def test_download_outlives_worker_deadline_without_releasing_lock_or_duplicate(preparation):
    handler, holder = preparation
    handler._worker._timeout = 0.02
    entered, release = Event(), Event()
    fetches = []

    def download(*, repo_id, tqdm_class):
        fetches.append(repo_id)
        with tqdm_class(unit="B", total=0, disable=True) as bar:
            bar.update(64)
            entered.set()
            assert release.wait(3)
        return "configured/model"

    with patch("app.preparation.snapshot_download", side_effect=download), install_loader(holder, []):
        first = asyncio.create_task(handler.prepare())
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            await asyncio.sleep(0.06)
            second = asyncio.create_task(handler.prepare())
            await asyncio.sleep(0)
            assert handler._worker.active
            assert not first.done()
            assert not second.done()
            assert handler.preparation_status().phase == "downloading"
            assert handler.preparation_status().downloaded_bytes == 64
        finally:
            release.set()
        a, b = await asyncio.gather(first, second)
        assert a == b
        assert a.loaded
        assert fetches == ["configured/model"]
