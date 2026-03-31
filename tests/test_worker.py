# tests/test_worker.py
import asyncio
import time
import pytest
from app.worker import InferenceWorker, QueueFullError, QueueTimeoutError


@pytest.mark.asyncio
async def test_submit_runs_function_and_returns_result():
    worker = InferenceWorker(max_size=5, timeout=5.0)
    try:
        result = await worker.submit(lambda: 42)
        assert result == 42
    finally:
        worker.stop()


@pytest.mark.asyncio
async def test_submit_propagates_exception():
    worker = InferenceWorker(max_size=5, timeout=5.0)
    try:
        def raise_err():
            raise ValueError("oops")
        with pytest.raises(ValueError, match="oops"):
            await worker.submit(raise_err)
    finally:
        worker.stop()


@pytest.mark.asyncio
async def test_queue_full_raises_immediately():
    worker = InferenceWorker(max_size=1, timeout=30.0)

    def slow():
        time.sleep(5)
        return "done"

    task = asyncio.create_task(worker.submit(slow))
    await asyncio.sleep(0.1)

    with pytest.raises(QueueFullError):
        await worker.submit(lambda: "second")

    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    finally:
        worker.stop()


@pytest.mark.asyncio
async def test_queue_size_reflects_pending_count():
    worker = InferenceWorker(max_size=10, timeout=10.0)
    assert worker.queue_size == 0
    assert worker.active is False
    worker.stop()


@pytest.mark.asyncio
async def test_timeout_raises_queue_timeout_error():
    worker = InferenceWorker(max_size=10, timeout=0.05)

    def slow():
        time.sleep(2)
        return "done"

    with pytest.raises(QueueTimeoutError):
        await worker.submit(slow)
    worker.stop()
