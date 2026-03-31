import pytest
from pathlib import Path
from typing import AsyncGenerator
from app.handlers.base import BaseHandler, AudioCapable
from app.schemas.common import ModelCard
from app.schemas.audio import TranscriptionParams, TranscriptionResult


def test_base_handler_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseHandler()


def test_audio_capable_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AudioCapable()


def test_concrete_handler_must_implement_all_methods():
    class PartialHandler(BaseHandler):
        async def initialize(self) -> None:
            pass

    with pytest.raises(TypeError):
        PartialHandler()


def test_concrete_handler_with_audio_capable_can_be_instantiated():
    class ConcreteHandler(BaseHandler, AudioCapable):
        async def initialize(self) -> None:
            pass
        async def cleanup(self) -> None:
            pass
        def model_info(self) -> ModelCard:
            return ModelCard(id="test-model")
        async def transcribe(self, audio_path: Path, params: TranscriptionParams) -> TranscriptionResult:
            return TranscriptionResult(text="", language="en", duration=0.0)
        async def transcribe_stream(self, audio_path: Path, params: TranscriptionParams) -> AsyncGenerator[str, None]:
            yield "data: [DONE]\n\n"

    handler = ConcreteHandler()
    assert handler is not None
