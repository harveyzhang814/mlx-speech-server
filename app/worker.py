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

    async def submit(self, task_id: str, fn: Callable[..., Any], *args: Any) -> Any:
        async with self._count_lock:
            if self._count >= self._max_size:
                raise QueueFullError(
                    f"Inference queue is full ({self._max_size} requests). Try again later."
                )
            self._count += 1

        current = asyncio.current_task()
        if current is not None:
            self._pending_tasks[task_id] = current

        try:
            try:
                return await asyncio.wait_for(
                    self._run(task_id, fn, *args), timeout=self._timeout
                )
            except asyncio.TimeoutError:
                raise QueueTimeoutError(
                    f"Request timed out after {self._timeout}s waiting in queue."
                )
            except asyncio.CancelledError:
                raise TaskCancelledError(f"Task {task_id} was cancelled.")
        finally:
            self._pending_tasks.pop(task_id, None)
            self._cancelling.discard(task_id)
            async with self._count_lock:
                self._count -= 1

    async def _run(self, task_id: str, fn: Callable[..., Any], *args: Any) -> Any:
        await self._execution_lock.acquire()
        self._active = True
        self._active_task_id = task_id
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._executor, fn if not args else lambda: fn(*args)
            )
            if task_id in self._cancelling:
                raise asyncio.CancelledError()
            return result
        finally:
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
