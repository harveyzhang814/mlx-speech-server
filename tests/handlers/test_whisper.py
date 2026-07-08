import pytest
import time
import asyncio
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
    with patch("app.handlers.whisper.mlx_whisper"):
        await handler.initialize()
        assert handler.model_path == "mlx-community/whisper-large-v3-turbo"
        await handler.cleanup()


@pytest.mark.asyncio
async def test_transcribe_returns_result(handler, tmp_wav_file):
    with patch("app.handlers.whisper.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe = MagicMock(return_value=MLX_WHISPER_RESULT)
        params = TranscriptionParams(language="en", temperature=0.0)
        result = await handler.transcribe(tmp_wav_file, params, task_id="test-id")

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
        await handler.transcribe(tmp_wav_file, params, task_id="test-id")

    call_kwargs = mock_mlx.transcribe.call_args[1]
    assert call_kwargs["language"] == "zh"
    assert call_kwargs["initial_prompt"] == "hint"
    assert call_kwargs["temperature"] == 0.2
    assert call_kwargs["condition_on_previous_text"] is False


@pytest.mark.asyncio
async def test_temperature_zero_uses_fallback_schedule(handler, tmp_wav_file):
    """temperature=0.0 must expand to the full tuple so Whisper can retry
    with higher temperatures when it detects repetition or low confidence."""
    with patch("app.handlers.whisper.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe = MagicMock(return_value=MLX_WHISPER_RESULT)
        params = TranscriptionParams(temperature=0.0)
        await handler.transcribe(tmp_wav_file, params, task_id="test-id")

    call_kwargs = mock_mlx.transcribe.call_args[1]
    assert call_kwargs["temperature"] == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@pytest.mark.asyncio
async def test_temperature_nonzero_passed_as_is(handler, tmp_wav_file):
    with patch("app.handlers.whisper.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe = MagicMock(return_value=MLX_WHISPER_RESULT)
        params = TranscriptionParams(temperature=0.5)
        await handler.transcribe(tmp_wav_file, params, task_id="test-id")

    call_kwargs = mock_mlx.transcribe.call_args[1]
    assert call_kwargs["temperature"] == 0.5


@pytest.mark.asyncio
async def test_condition_on_previous_text_always_false(handler, tmp_wav_file):
    """condition_on_previous_text=False prevents cascading hallucinations
    where one bad segment poisons all subsequent segments."""
    with patch("app.handlers.whisper.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe = MagicMock(return_value=MLX_WHISPER_RESULT)
        params = TranscriptionParams(temperature=0.0)
        await handler.transcribe(tmp_wav_file, params, task_id="test-id")

    call_kwargs = mock_mlx.transcribe.call_args[1]
    assert call_kwargs["condition_on_previous_text"] is False


@pytest.mark.asyncio
async def test_transcribe_stream_yields_sse_segments(handler, tmp_wav_file):
    with patch("app.handlers.whisper.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe = MagicMock(return_value=MLX_WHISPER_RESULT)
        params = TranscriptionParams()
        chunks = []
        async for chunk in handler.transcribe_stream(tmp_wav_file, params, task_id="test-id"):
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


@pytest.mark.asyncio
async def test_idle_watcher_unloads_model(worker):
    import sys
    import mlx_whisper.transcribe  # noqa: F401 — side-effect: loads module into sys.modules
    _mlx_t = sys.modules["mlx_whisper.transcribe"]
    _mlx_t.ModelHolder.model = object()  # simulate loaded model

    with patch("app.handlers.whisper._IDLE_TIMEOUT", 0.0), \
         patch("app.handlers.whisper._WATCH_INTERVAL", 0.001), \
         patch("app.handlers.whisper.mx") as mock_mx, \
         patch("app.handlers.whisper.gc"):
        handler = WhisperHandler(model_path="mlx-community/whisper-large-v3-turbo", worker=worker)
        await handler.initialize()
        handler._last_used = 0.0  # force past idle threshold
        await asyncio.sleep(0.05)  # let watcher trigger
        await handler.cleanup()

    assert _mlx_t.ModelHolder.model is None
    mock_mx.clear_cache.assert_called()


@pytest.mark.asyncio
async def test_idle_watcher_skips_while_active(worker):
    worker._active = True  # simulate in-flight inference

    with patch("app.handlers.whisper._IDLE_TIMEOUT", 0.0), \
         patch("app.handlers.whisper._WATCH_INTERVAL", 0.001):
        handler = WhisperHandler(model_path="mlx-community/whisper-large-v3-turbo", worker=worker)
        with patch.object(handler, "_unload") as mock_unload:
            await handler.initialize()
            handler._last_used = 0.0
            await asyncio.sleep(0.05)
            await handler.cleanup()

    mock_unload.assert_not_called()


@pytest.mark.asyncio
async def test_last_used_updated_on_transcribe(handler, tmp_wav_file):
    before = time.time()
    with patch("app.handlers.whisper.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe = MagicMock(return_value=MLX_WHISPER_RESULT)
        params = TranscriptionParams(language="en", temperature=0.0)
        await handler.transcribe(tmp_wav_file, params, task_id="test-id")

    assert handler._last_used >= before


@pytest.mark.asyncio
async def test_watcher_cancelled_on_cleanup(worker):
    handler = WhisperHandler(model_path="mlx-community/whisper-large-v3-turbo", worker=worker)
    with patch("app.handlers.whisper._WATCH_INTERVAL", 3600.0):
        await handler.initialize()
        task = handler._watcher_task
        assert not task.done()
        await handler.cleanup()
    assert task.cancelled()


@pytest.mark.asyncio
async def test_initial_last_used_prevents_early_unload(worker):
    with patch("app.handlers.whisper._IDLE_TIMEOUT", 1800.0), \
         patch("app.handlers.whisper._WATCH_INTERVAL", 0.001):
        handler = WhisperHandler(model_path="mlx-community/whisper-large-v3-turbo", worker=worker)
        with patch.object(handler, "_unload") as mock_unload:
            await handler.initialize()
            # _last_used = time.time() set in initialize() — well under 1800s
            await asyncio.sleep(0.05)
            await handler.cleanup()

    mock_unload.assert_not_called()
