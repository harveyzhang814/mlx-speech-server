from __future__ import annotations
import asyncio
import gc
import json
import time
from pathlib import Path
from dataclasses import replace
from threading import RLock
from uuid import uuid4
from typing import AsyncGenerator

import mlx.core as mx
import mlx_whisper
from loguru import logger

from app.handlers.base import BaseHandler, AudioCapable
from app.schemas.audio import (
    TranscriptionParams,
    TranscriptionResult,
    SegmentResult,
)
from app.schemas.common import ModelCard
from app.worker import InferenceWorker, QueueFullError, QueueTimeoutError, TaskCancelledError
from app.preparation import PreparationError, PreparationSnapshot, WhisperModelAdapter

_IDLE_TIMEOUT = 1800.0   # 30 minutes
_WATCH_INTERVAL = 300.0  # check every 5 minutes


class WhisperHandler(BaseHandler, AudioCapable):
    """Runs mlx_whisper.transcribe on the shared InferenceWorker thread."""

    def __init__(self, model_path: str, worker: InferenceWorker) -> None:
        self.model_path = model_path
        self._worker = worker
        self._model_id = model_path.rstrip("/").split("/")[-1]
        self._last_used: float = 0.0
        self._watcher_task: asyncio.Task | None = None
        self._model_adapter = WhisperModelAdapter()
        self._preparation_lock = RLock()
        self._preparation = PreparationSnapshot(operation_id=None, model_id=self._model_id)
        self._preparation_task: asyncio.Task | None = None
        self._loaded_model = None

    async def initialize(self) -> None:
        self._last_used = time.time()
        self._watcher_task = asyncio.create_task(self._watch_idle())
        logger.info(f"WhisperHandler ready (model will load on first request): {self.model_path}")

    async def cleanup(self) -> None:
        if self._watcher_task is not None:
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except asyncio.CancelledError:
                pass
        logger.info("WhisperHandler cleanup")
        gc.collect()
        mx.clear_cache()

    def model_info(self) -> ModelCard:
        return ModelCard(id=self._model_id)

    def preparation_status(self) -> PreparationSnapshot:
        with self._preparation_lock:
            return self._preparation

    async def prepare(self) -> PreparationSnapshot:
        self._last_used = time.time()
        if self._preparation_task is None or self._preparation_task.done():
            operation_id = self._preparation.operation_id or str(uuid4())
            if not self._preparation.loaded:
                operation_id = str(uuid4())
                with self._preparation_lock:
                    self._preparation = PreparationSnapshot(
                        operation_id=operation_id, model_id=self._model_id
                    )
            self._preparation_task = asyncio.create_task(self._prepare_operation(operation_id))
        # Each client owns its wait, never the shared loading operation.
        return await asyncio.shield(self._preparation_task)

    async def _prepare_operation(self, operation_id: str) -> PreparationSnapshot:
        loop = asyncio.get_running_loop()
        completed = loop.create_future()

        def load() -> PreparationSnapshot:
            snapshot = self._prepare_model()
            loop.call_soon_threadsafe(completed.set_result, snapshot)
            return snapshot

        try:
            return await self._worker.submit(operation_id, load)
        except QueueTimeoutError as exc:
            if completed.done() or self._worker.is_executing(operation_id):
                # The request deadline cannot cancel native loading or open a
                # second preparation operation while the first still runs.
                return await asyncio.shield(completed)
            return self._preparation_scheduling_failed(exc, "queue_timeout")
        except QueueFullError as exc:
            return self._preparation_scheduling_failed(exc, "queue_full")
        except Exception as exc:
            return self._preparation_failed(exc)

    def _preparation_scheduling_failed(
        self, exc: Exception, code: str
    ) -> PreparationSnapshot:
        # No model operation began: admission cannot invalidate evaluated weights.
        if self._preparation.loaded:
            return self._preparation
        return self._preparation_failed(exc, code=code)

    def _preparation_failed(
        self, exc: Exception, code: str = "model_load_failed"
    ) -> PreparationSnapshot:
        with self._preparation_lock:
            self._preparation = replace(
                self._preparation, phase="failed", loaded=False,
                error=PreparationError(code=code, message=str(exc)),
            )
        return self._preparation

    def _prepare_model(self) -> PreparationSnapshot:
        current = self._model_adapter.current_model(self.model_path)
        if self._preparation.loaded and current is not None and current is self._loaded_model:
            return self._preparation
        operation_id = self._preparation.operation_id
        with self._preparation_lock:
            self._preparation = replace(self._preparation, phase="idle", loaded=False, progress=None,
                                        downloaded_bytes=None, error=None)

        active = True

        def report(phase: str, received: int | None) -> None:
            nonlocal active
            with self._preparation_lock:
                current = self._preparation
                if not active or current.operation_id != operation_id or current.phase not in {"idle", "downloading"}:
                    return
                if phase == "downloading":
                    if received is None or received < (current.downloaded_bytes or 0):
                        return
                    self._preparation = replace(
                        current, phase="downloading", loaded=False, progress=None,
                        downloaded_bytes=received, error=None,
                    )
                elif phase == "loading":
                    active = False
                    self._preparation = replace(current, phase="loading", loaded=False, error=None)

        try:
            self._loaded_model = self._model_adapter.load(self.model_path, report=report)
        except Exception as exc:
            self._loaded_model = None
            self._model_adapter.unload()
            return self._preparation_failed(exc)
        finally:
            with self._preparation_lock:
                active = False
        with self._preparation_lock:
            self._preparation = replace(self._preparation, phase="ready", loaded=True, error=None)
        self._last_used = time.time()
        return self._preparation

    async def transcribe(
        self, audio_path: Path, params: TranscriptionParams, task_id: str
    ) -> TranscriptionResult:
        self._last_used = time.time()
        # temperature=0.0 (single float) disables Whisper's fallback schedule and
        # prevents repetition/confidence recovery. Use the full schedule when the
        # client requests 0 so that mlx_whisper can retry with higher temperatures
        # when it detects repetition loops or low-confidence segments.
        temperature: float | tuple[float, ...] = (
            (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if params.temperature == 0.0 else params.temperature
        )

        def _run() -> dict:
            raw = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=self.model_path,
                language=params.language,
                initial_prompt=params.prompt,
                temperature=temperature,
                condition_on_previous_text=False,
                fp16=True,
            )
            current = self._model_adapter.current_model(self.model_path)
            if current is not None:
                self._loaded_model = current
                self._preparation = replace(
                    self._preparation, phase="ready", loaded=True, error=None
                )
            self._last_used = time.time()
            return raw

        raw = await self._worker.submit(task_id, _run)
        return _parse_result(raw)

    async def transcribe_stream(
        self, audio_path: Path, params: TranscriptionParams, task_id: str
    ) -> AsyncGenerator[str, None]:
        result = await self.transcribe(audio_path, params, task_id)
        for segment in result.segments:
            yield f"data: {json.dumps({'text': segment.text})}\n\n"
        yield "data: [DONE]\n\n"

    async def _watch_idle(self) -> None:
        try:
            while True:
                await asyncio.sleep(_WATCH_INTERVAL)
                idle = time.time() - self._last_used
                if idle > _IDLE_TIMEOUT and not self._worker.active and self._worker.queue_size == 0:
                    await self._worker.submit(f"idle-unload-{uuid4()}", self._unload_if_idle)
                else:
                    logger.debug(f"Idle watcher: {idle / 60:.1f} min since last request, skipping")
        except TaskCancelledError:
            raise asyncio.CancelledError() from None
        except asyncio.CancelledError:
            pass

    def _unload_if_idle(self) -> None:
        # The decision is rechecked after taking the same lock as prepare/infer.
        if time.time() - self._last_used > _IDLE_TIMEOUT and self._worker.queue_size == 0:
            self._unload()

    def _unload(self) -> None:
        had_model = self._model_adapter.unload()
        self._loaded_model = None
        self._preparation = replace(self._preparation, phase="idle", loaded=False, error=None)
        if not had_model:
            return
        mx.clear_cache()
        gc.collect()
        logger.info(f"Model unloaded after {_IDLE_TIMEOUT / 60:.0f}min idle")


def _parse_result(raw: dict) -> TranscriptionResult:
    segments = [
        SegmentResult(
            id=seg["id"],
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
            no_speech_prob=seg.get("no_speech_prob", 0.0),
        )
        for seg in raw.get("segments", [])
    ]
    duration = segments[-1].end if segments else 0.0
    return TranscriptionResult(
        text=raw.get("text", ""),
        language=raw.get("language", ""),
        duration=duration,
        segments=segments,
    )
