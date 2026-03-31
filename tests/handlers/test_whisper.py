import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from app.handlers.whisper import WhisperHandler
from app.schemas.audio import TranscriptionParams, TranscriptionResult
from app.schemas.common import ModelCard
from app.worker import InferenceWorker


MLX_WHISPER_RESULT = {
    "text": " Hello world",
    "language": "en",
    "segments": [
        {"id": 0, "start": 0.0, "end": 2.1, "text": " Hello world", "no_speech_prob": 0.01}
    ],
}


@pytest.fixture
def worker():
    w = InferenceWorker(max_size=5, timeout=30.0)
    yield w
    w.stop()


@pytest.fixture
def handler(worker):
    return WhisperHandler(
        model_path="mlx-community/whisper-large-v3-turbo",
        worker=worker,
    )


@pytest.mark.asyncio
async def test_initialize_sets_model_path(handler):
    with patch("app.handlers.whisper.mlx_whisper") as mock_mlx:
        await handler.initialize()
        assert handler.model_path == "mlx-community/whisper-large-v3-turbo"


@pytest.mark.asyncio
async def test_transcribe_returns_result(handler, tmp_wav_file):
    with patch("app.handlers.whisper.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe = MagicMock(return_value=MLX_WHISPER_RESULT)
        params = TranscriptionParams(language="en", temperature=0.0)
        result = await handler.transcribe(tmp_wav_file, params)

    assert isinstance(result, TranscriptionResult)
    assert result.text == " Hello world"
    assert result.language == "en"
    assert len(result.segments) == 1
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 2.1
    assert result.segments[0].no_speech_prob == 0.01


@pytest.mark.asyncio
async def test_transcribe_passes_params_to_mlx_whisper(handler, tmp_wav_file):
    with patch("app.handlers.whisper.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe = MagicMock(return_value=MLX_WHISPER_RESULT)
        params = TranscriptionParams(language="zh", prompt="hint", temperature=0.2)
        await handler.transcribe(tmp_wav_file, params)

    call_kwargs = mock_mlx.transcribe.call_args[1]
    assert call_kwargs["language"] == "zh"
    assert call_kwargs["initial_prompt"] == "hint"
    assert call_kwargs["temperature"] == 0.2


@pytest.mark.asyncio
async def test_transcribe_stream_yields_sse_segments(handler, tmp_wav_file):
    with patch("app.handlers.whisper.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe = MagicMock(return_value=MLX_WHISPER_RESULT)
        params = TranscriptionParams()
        chunks = []
        async for chunk in handler.transcribe_stream(tmp_wav_file, params):
            chunks.append(chunk)

    assert len(chunks) == 2  # 1 segment + [DONE]
    assert '"text": " Hello world"' in chunks[0]
    assert chunks[-1] == "data: [DONE]\n\n"


def test_model_info_returns_model_card(handler):
    card = handler.model_info()
    assert isinstance(card, ModelCard)
    assert card.id == "whisper-large-v3-turbo"


@pytest.mark.asyncio
async def test_cleanup_runs_gc(handler):
    with patch("app.handlers.whisper.gc") as mock_gc, \
         patch("app.handlers.whisper.mx") as mock_mx:
        await handler.cleanup()
        mock_gc.collect.assert_called_once()
        mock_mx.clear_cache.assert_called_once()
