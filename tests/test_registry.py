import pytest
from app.registry import ModelRegistry, ModelNotFoundError
from app.handlers.base import BaseHandler, AudioCapable
from app.schemas.common import ModelCard
from app.schemas.audio import TranscriptionParams, TranscriptionResult
from pathlib import Path
from typing import AsyncGenerator


class _FakeHandler(BaseHandler, AudioCapable):
    async def initialize(self) -> None: pass
    async def cleanup(self) -> None: pass
    def model_info(self) -> ModelCard:
        return ModelCard(id="fake-model")
    async def transcribe(self, audio_path: Path, params: TranscriptionParams) -> TranscriptionResult:
        return TranscriptionResult(text="", language="en", duration=0.0)
    async def transcribe_stream(self, audio_path: Path, params: TranscriptionParams) -> AsyncGenerator[str, None]:
        yield "data: [DONE]\n\n"


def test_register_and_get():
    registry = ModelRegistry()
    handler = _FakeHandler()
    registry.register("fake-model", handler)
    assert registry.get("fake-model") is handler


def test_get_unknown_model_raises():
    registry = ModelRegistry()
    with pytest.raises(ModelNotFoundError, match="fake-model"):
        registry.get("fake-model")


def test_list_models_returns_model_cards():
    registry = ModelRegistry()
    handler = _FakeHandler()
    registry.register("fake-model", handler)
    models = registry.list_models()
    assert len(models) == 1
    assert models[0].id == "fake-model"


def test_list_models_empty_registry():
    registry = ModelRegistry()
    assert registry.list_models() == []
