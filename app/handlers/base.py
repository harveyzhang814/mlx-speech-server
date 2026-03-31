from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator
from app.schemas.common import ModelCard
from app.schemas.audio import TranscriptionParams, TranscriptionResult


class BaseHandler(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        """Load model and any other startup work."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Release resources on shutdown."""

    @abstractmethod
    def model_info(self) -> ModelCard:
        """Return model metadata for /v1/models."""


class AudioCapable(ABC):
    """Mixin for handlers that support audio transcription."""

    @abstractmethod
    async def transcribe(
        self, audio_path: Path, params: TranscriptionParams
    ) -> TranscriptionResult:
        """Run full transcription and return complete result."""

    @abstractmethod
    async def transcribe_stream(
        self, audio_path: Path, params: TranscriptionParams
    ) -> AsyncGenerator[str, None]:
        """Yield SSE-formatted strings, one per segment, ending with 'data: [DONE]\\n\\n'."""
