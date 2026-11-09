from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class QueueFullError(Exception):
    pass


class QueueTimeoutError(Exception):
    pass


class TaskCancelledError(Exception):
    pass


class TaskNotFoundError(Exception):
    pass


class InferenceWorker:
    """Serializes inference calls on a single background thread."""

    def __init__(self, max_size: int, timeout: float) -> None:
        self._max_size = max_size
        self._timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._execution_lock = asyncio.Semaphore(1)
        self._count = 0
        self._active = False
        self._count_lock = asyncio.Lock()
        self._active_task_id: str | None = None
        self._cancelling: set[str] = set()
        self._pending_tasks: dict[str, asyncio.Task] = {}
        self._native_futures: dict[str, asyncio.Future] = {}

    @property
    def queue_size(self) -> int:
        return max(0, self._count - (1 if self._active else 0))

    @property
    def active(self) -> bool:
        return self._active

    @property
    def active_status(self) -> str:
        if not self._active:
            return "idle"
        if self._active_task_id in self._cancelling:
            return "cancelling"
        return "running"

    @property
    def max_size(self) -> int:
        return self._max_size

    def is_executing(self, task_id: str) -> bool:
        return self._active_task_id == task_id

    async def submit(self, task_id: str, fn: Callable[..., Any], *args: Any) -> Any:
        async with self._count_lock:
            if self._count >= self._max_size:
                raise QueueFullError(
                    f"Inference queue is full ({self._max_size} requests). Try again later."
                )
            self._count += 1

        execution = asyncio.create_task(self._run(task_id, fn, *args))
        self._pending_tasks[task_id] = execution
        execution.add_done_callback(lambda task: self._execution_done(task_id, task))
        try:
            return await asyncio.wait_for(asyncio.shield(execution), timeout=self._timeout)
        except asyncio.TimeoutError:
            if task_id in self._pending_tasks:
                self.cancel(task_id)
                execution.cancel()
            raise QueueTimeoutError(
                f"Request timed out after {self._timeout}s waiting in queue."
            ) from None
        except asyncio.CancelledError:
            if task_id in self._pending_tasks:
                self.cancel(task_id)
                execution.cancel()
            raise TaskCancelledError(f"Task {task_id} was cancelled.") from None

    async def _run(self, task_id: str, fn: Callable[..., Any], *args: Any) -> Any:
        await self._execution_lock.acquire()
        self._active = True
        self._active_task_id = task_id
        loop = asyncio.get_running_loop()
        native = loop.run_in_executor(
            self._executor, fn if not args else lambda: fn(*args)
        )
        self._native_futures[task_id] = native
        result = await asyncio.shield(native)
        if task_id in self._cancelling:
            raise asyncio.CancelledError()
        return result

    def _execution_done(self, task_id: str, execution: asyncio.Task) -> None:
        native = self._native_futures.get(task_id)
        if native is not None and not native.done():
            # An HTTP waiter can disappear; the native execution still owns the
            # resource, queue capacity and active marker until it really finishes.
            native.add_done_callback(lambda _: self._execution_done(task_id, execution))
            return
        if native is not None and not native.cancelled():
            native.exception()  # consume errors even when the waiter has gone away
        if not execution.cancelled():
            execution.exception()
        self._native_futures.pop(task_id, None)
        self._pending_tasks.pop(task_id, None)
        self._cancelling.discard(task_id)
        self._count -= 1
        if self._active_task_id == task_id:
            self._active = False
            self._active_task_id = None
            self._execution_lock.release()

    def cancel(self, task_id: str) -> None:
        if task_id not in self._pending_tasks:
            raise TaskNotFoundError(f"Task '{task_id}' not found.")
        if task_id == self._active_task_id:
            self._cancelling.add(task_id)
        else:
            task = self._pending_tasks[task_id]
            if not task.cancelled():
                task.cancel()

    def stop(self) -> None:
        self._executor.shutdown(wait=False)
