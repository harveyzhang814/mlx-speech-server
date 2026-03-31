import pytest
from app.schemas.common import ModelCard, ErrorResponse, QueueStats
from app.schemas.audio import (
    TranscriptionRequest,
    TranscriptionParams,
    TranscriptionResult,
    SegmentResult,
    ResponseFormat,
)


def test_model_card_fields():
    card = ModelCard(id="whisper-large-v3-turbo")
    assert card.id == "whisper-large-v3-turbo"
    assert card.object == "model"
    assert card.owned_by == "local"


def test_error_response_structure():
    err = ErrorResponse(message="bad request", type="invalid_request_error", code="model_not_found")
    assert err.error["message"] == "bad request"
    assert err.error["type"] == "invalid_request_error"
    assert err.error["code"] == "model_not_found"


def test_queue_stats_fields():
    stats = QueueStats(queue_size=2, queue_max_size=10, active=True)
    assert stats.queue_size == 2
    assert stats.queue_max_size == 10
    assert stats.active is True


def test_transcription_params_defaults():
    params = TranscriptionParams()
    assert params.language is None
    assert params.prompt is None
    assert params.temperature == 0.0


def test_transcription_result():
    seg = SegmentResult(id=0, start=0.0, end=2.1, text=" Hello", no_speech_prob=0.01)
    result = TranscriptionResult(
        text=" Hello", language="en", duration=2.1, segments=[seg]
    )
    assert result.text == " Hello"
    assert len(result.segments) == 1
    assert result.segments[0].start == 0.0


def test_response_format_values():
    assert ResponseFormat.JSON == "json"
    assert ResponseFormat.TEXT == "text"
    assert ResponseFormat.VERBOSE_JSON == "verbose_json"
    assert ResponseFormat.SRT == "srt"
    assert ResponseFormat.VTT == "vtt"


def test_transcription_request_defaults():
    req = TranscriptionRequest(model="whisper-large-v3-turbo")
    assert req.response_format == ResponseFormat.JSON
    assert req.temperature == 0.0
    assert req.stream is False
    assert req.language is None
    assert req.prompt is None
