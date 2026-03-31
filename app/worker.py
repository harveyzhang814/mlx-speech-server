# app/worker.py
from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class QueueFullError(Exception):
    pass


class QueueTimeoutError(Exception):
    pass


class InferenceWorker:
    """Serializes inference calls on a single background thread.

    Keeps the asyncio event loop responsive during blocking MLX/Metal compute.
    """

    def __init__(self, max_size: int, timeout: float) -> None:
        self._max_size = max_size
        self._timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._execution_lock = asyncio.Semaphore(1)
        self._count = 0
        self._active = False
        self._count_lock = asyncio.Lock()

    @property
    def queue_size(self) -> int:
        return max(0, self._count - (1 if self._active else 0))

    @property
    def active(self) -> bool:
        return self._active

    @property
    def max_size(self) -> int:
        return self._max_size

    async def submit(self, fn: Callable[..., Any], *args: Any) -> Any:
        async with self._count_lock:
            if self._count >= self._max_size:
                raise QueueFullError(
                    f"Inference queue is full ({self._max_size} requests). Try again later."
                )
            self._count += 1

        try:
            try:
                return await asyncio.wait_for(
                    self._run(fn, *args), timeout=self._timeout
                )
            except asyncio.TimeoutError:
                raise QueueTimeoutError(
                    f"Request timed out after {self._timeout}s waiting in queue."
                )
        finally:
            async with self._count_lock:
                self._count -= 1

    async def _run(self, fn: Callable[..., Any], *args: Any) -> Any:
        await self._execution_lock.acquire()
        self._active = True
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._executor, fn if not args else lambda: fn(*args)
            )
            return result
        finally:
            self._active = False
            self._execution_lock.release()

    def stop(self) -> None:
        self._executor.shutdown(wait=False)
