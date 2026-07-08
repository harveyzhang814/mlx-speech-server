from __future__ import annotations
import asyncio
import gc
import json
import time
from pathlib import Path
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
from app.worker import InferenceWorker

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
            return mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=self.model_path,
                language=params.language,
                initial_prompt=params.prompt,
                temperature=temperature,
                condition_on_previous_text=False,
            )

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
                if idle > _IDLE_TIMEOUT and not self._worker.active:
                    self._unload()
                else:
                    logger.debug(f"Idle watcher: {idle / 60:.1f} min since last request, skipping")
        except asyncio.CancelledError:
            pass

    def _unload(self) -> None:
        import sys
        import mlx_whisper.transcribe  # noqa: F401 — side-effect: loads module into sys.modules
        _t = sys.modules["mlx_whisper.transcribe"]
        _t.ModelHolder.model = None
        mx.clear_cache()
        gc.collect()
        logger.info("Model unloaded after 30min idle")


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
