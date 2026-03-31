from __future__ import annotations
import gc
import json
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


class WhisperHandler(BaseHandler, AudioCapable):
    """Runs mlx_whisper.transcribe on the shared InferenceWorker thread."""

    def __init__(self, model_path: str, worker: InferenceWorker) -> None:
        self.model_path = model_path
        self._worker = worker
        self._model_id = model_path.rstrip("/").split("/")[-1]

    async def initialize(self) -> None:
        logger.info(f"WhisperHandler ready (model will load on first request): {self.model_path}")

    async def cleanup(self) -> None:
        logger.info("WhisperHandler cleanup")
        gc.collect()
        mx.clear_cache()

    def model_info(self) -> ModelCard:
        return ModelCard(id=self._model_id)

    async def transcribe(
        self, audio_path: Path, params: TranscriptionParams
    ) -> TranscriptionResult:
        def _run() -> dict:
            return mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=self.model_path,
                language=params.language,
                initial_prompt=params.prompt,
                temperature=params.temperature,
            )

        raw = await self._worker.submit(_run)
        return _parse_result(raw)

    async def transcribe_stream(
        self, audio_path: Path, params: TranscriptionParams
    ) -> AsyncGenerator[str, None]:
        result = await self.transcribe(audio_path, params)
        for segment in result.segments:
            yield f"data: {json.dumps({'text': segment.text})}\n\n"
        yield "data: [DONE]\n\n"


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
