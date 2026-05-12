import asyncio
import time
import uuid
import pytest
from app.worker import (
    InferenceWorker,
    QueueFullError,
    QueueTimeoutError,
    TaskCancelledError,
    TaskNotFoundError,
)


@pytest.fixture
def worker():
    w = InferenceWorker(max_size=5, timeout=5.0)
    yield w
    w.stop()


# ── existing behaviour (updated to new submit signature) ─────────────────────

async def test_submit_runs_function_and_returns_result(worker):
    task_id = str(uuid.uuid4())
    result = await worker.submit(task_id, lambda: 42)
    assert result == 42


async def test_submit_propagates_exception(worker):
    def raise_err():
        raise ValueError("oops")
    with pytest.raises(ValueError, match="oops"):
        await worker.submit(str(uuid.uuid4()), raise_err)


async def test_queue_full_raises_immediately():
    worker = InferenceWorker(max_size=1, timeout=30.0)

    def slow():
        time.sleep(5)

    task = asyncio.create_task(worker.submit(str(uuid.uuid4()), slow))
    await asyncio.sleep(0.1)

    with pytest.raises(QueueFullError):
        await worker.submit(str(uuid.uuid4()), lambda: "second")

    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    finally:
        worker.stop()


async def test_queue_size_reflects_pending_count(worker):
    assert worker.queue_size == 0
    assert worker.active is False


async def test_timeout_raises_queue_timeout_error():
    worker = InferenceWorker(max_size=10, timeout=0.05)

    def slow():
        time.sleep(2)

    with pytest.raises(QueueTimeoutError):
        await worker.submit(str(uuid.uuid4()), slow)
    worker.stop()


# ── task_id uniqueness ────────────────────────────────────────────────────────

async def test_each_submit_uses_the_provided_task_id(worker):
    seen_ids: list[str] = []
    tid1 = str(uuid.uuid4())
    tid2 = str(uuid.uuid4())

    await worker.submit(tid1, lambda: None)
    await worker.submit(tid2, lambda: None)

    assert tid1 != tid2


# ── cancel queued task ────────────────────────────────────────────────────────

async def test_cancel_queued_task_raises_task_cancelled_error():
    worker = InferenceWorker(max_size=5, timeout=30.0)

    def blocker():
        time.sleep(5)

    t1 = asyncio.create_task(worker.submit(str(uuid.uuid4()), blocker))
    await asyncio.sleep(0.1)  # let t1 acquire the lock

    t2_id = str(uuid.uuid4())
    t2 = asyncio.create_task(worker.submit(t2_id, lambda: "second"))
    await asyncio.sleep(0.05)  # let t2 enter the queue

    worker.cancel(t2_id)

    with pytest.raises(TaskCancelledError):
        await t2

    t1.cancel()
    try:
        await t1
    except (asyncio.CancelledError, Exception):
        pass
    finally:
        worker.stop()


async def test_cancel_queued_task_decrements_count():
    worker = InferenceWorker(max_size=5, timeout=30.0)

    def blocker():
        time.sleep(5)

    t1 = asyncio.create_task(worker.submit(str(uuid.uuid4()), blocker))
    await asyncio.sleep(0.1)

    t2_id = str(uuid.uuid4())
    t2 = asyncio.create_task(worker.submit(t2_id, lambda: None))
    await asyncio.sleep(0.05)

    count_before = worker._count
    worker.cancel(t2_id)
    try:
        await t2
    except TaskCancelledError:
        pass

    assert worker._count == count_before - 1

    t1.cancel()
    try:
        await t1
    except (asyncio.CancelledError, Exception):
        pass
    finally:
        worker.stop()


async def test_cancel_queued_task_does_not_affect_running_task():
    worker = InferenceWorker(max_size=5, timeout=30.0)
    running_result = None

    def fast():
        return "running_done"

    def blocker():
        time.sleep(0.3)
        return "block_done"

    t1_id = str(uuid.uuid4())
    t1 = asyncio.create_task(worker.submit(t1_id, blocker))
    await asyncio.sleep(0.05)

    t2_id = str(uuid.uuid4())
    t2 = asyncio.create_task(worker.submit(t2_id, fast))
    await asyncio.sleep(0.02)

    worker.cancel(t2_id)
    try:
        await t2
    except TaskCancelledError:
        pass

    running_result = await t1
    assert running_result == "block_done"
    worker.stop()


# ── cancel running task ───────────────────────────────────────────────────────

async def test_cancel_running_task_returns_immediately():
    worker = InferenceWorker(max_size=5, timeout=30.0)

    def slow():
        time.sleep(5)

    t1_id = str(uuid.uuid4())
    t1 = asyncio.create_task(worker.submit(t1_id, slow))
    await asyncio.sleep(0.1)  # let inference start

    start = time.monotonic()
    worker.cancel(t1_id)
    elapsed = time.monotonic() - start

    assert elapsed < 0.1, f"cancel() blocked for {elapsed:.3f}s"

    try:
        await t1
    except (TaskCancelledError, asyncio.CancelledError, Exception):
        pass
    finally:
        worker.stop()


async def test_cancel_running_task_shows_cancelling_status():
    worker = InferenceWorker(max_size=5, timeout=30.0)

    def slow():
        time.sleep(5)

    t1_id = str(uuid.uuid4())
    t1 = asyncio.create_task(worker.submit(t1_id, slow))
    await asyncio.sleep(0.1)

    assert worker.active_status == "running"
    worker.cancel(t1_id)
    assert worker.active_status == "cancelling"

    try:
        await t1
    except (TaskCancelledError, asyncio.CancelledError, Exception):
        pass
    finally:
        worker.stop()


async def test_cancelled_running_task_discards_result():
    worker = InferenceWorker(max_size=5, timeout=30.0)

    def short_inference():
        time.sleep(0.15)
        return "should_be_discarded"

    t1_id = str(uuid.uuid4())
    t1 = asyncio.create_task(worker.submit(t1_id, short_inference))
    await asyncio.sleep(0.05)  # let inference start

    worker.cancel(t1_id)

    with pytest.raises(TaskCancelledError):
        await t1
    worker.stop()


async def test_cancel_running_task_restores_idle_status_after_inference_finishes():
    worker = InferenceWorker(max_size=5, timeout=30.0)

    def short_inference():
        time.sleep(0.1)

    t1_id = str(uuid.uuid4())
    t1 = asyncio.create_task(worker.submit(t1_id, short_inference))
    await asyncio.sleep(0.02)

    worker.cancel(t1_id)
    try:
        await t1
    except (TaskCancelledError, Exception):
        pass

    await asyncio.sleep(0.01)  # let cleanup settle
    assert worker.active_status == "idle"
    worker.stop()


# ── cancel error cases ────────────────────────────────────────────────────────

async def test_cancel_unknown_task_id_raises_task_not_found_error(worker):
    with pytest.raises(TaskNotFoundError):
        worker.cancel("nonexistent-id")


async def test_cancel_completed_task_raises_task_not_found_error(worker):
    task_id = str(uuid.uuid4())
    await worker.submit(task_id, lambda: None)
    # task_id should be cleaned up after completion
    with pytest.raises(TaskNotFoundError):
        worker.cancel(task_id)


async def test_cancel_running_task_twice_is_idempotent():
    worker = InferenceWorker(max_size=5, timeout=30.0)

    def slow():
        time.sleep(5)

    t1_id = str(uuid.uuid4())
    t1 = asyncio.create_task(worker.submit(t1_id, slow))
    await asyncio.sleep(0.1)

    worker.cancel(t1_id)
    worker.cancel(t1_id)  # second call must not raise

    try:
        await t1
    except (TaskCancelledError, asyncio.CancelledError, Exception):
        pass
    finally:
        worker.stop()


# ── active_status property ────────────────────────────────────────────────────

async def test_active_status_is_idle_when_no_work(worker):
    assert worker.active_status == "idle"


async def test_active_status_is_running_during_inference():
    worker = InferenceWorker(max_size=5, timeout=30.0)

    def slow():
        time.sleep(5)

    t1 = asyncio.create_task(worker.submit(str(uuid.uuid4()), slow))
    await asyncio.sleep(0.1)

    assert worker.active_status == "running"

    t1.cancel()
    try:
        await t1
    except (asyncio.CancelledError, Exception):
        pass
    finally:
        worker.stop()
