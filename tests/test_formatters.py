import json
import pytest
from app.formatters import format_transcription
from app.schemas.audio import ResponseFormat, TranscriptionResult, SegmentResult


@pytest.fixture
def result():
    return TranscriptionResult(
        text=" Hello world",
        language="en",
        duration=4.2,
        segments=[
            SegmentResult(id=0, start=0.0, end=2.1, text=" Hello", no_speech_prob=0.01),
            SegmentResult(id=1, start=2.1, end=4.2, text=" world", no_speech_prob=0.02),
        ],
    )


def test_format_json(result):
    output = format_transcription(result, ResponseFormat.JSON)
    data = json.loads(output)
    assert data == {"text": " Hello world"}


def test_format_text(result):
    output = format_transcription(result, ResponseFormat.TEXT)
    assert output == " Hello world"


def test_format_verbose_json(result):
    output = format_transcription(result, ResponseFormat.VERBOSE_JSON)
    data = json.loads(output)
    assert data["task"] == "transcribe"
    assert data["language"] == "en"
    assert data["duration"] == 4.2
    assert data["text"] == " Hello world"
    assert len(data["segments"]) == 2
    assert data["segments"][0]["start"] == 0.0
    assert data["segments"][0]["end"] == 2.1
    assert data["segments"][0]["text"] == " Hello"
    assert data["segments"][0]["no_speech_prob"] == 0.01


def test_format_srt(result):
    output = format_transcription(result, ResponseFormat.SRT)
    lines = output.strip().split("\n")
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:02,100"
    assert lines[2] == " Hello"
    assert lines[4] == "2"
    assert lines[5] == "00:00:02,100 --> 00:00:04,200"
    assert lines[6] == " world"


def test_format_vtt(result):
    output = format_transcription(result, ResponseFormat.VTT)
    assert output.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:02.100" in output
    assert " Hello" in output
    assert "00:00:02.100 --> 00:00:04.200" in output
    assert " world" in output


def test_format_srt_timestamp_precision():
    result = TranscriptionResult(
        text="test",
        language="en",
        duration=3.5,
        segments=[SegmentResult(id=0, start=1.234, end=3.567, text="test", no_speech_prob=0.0)],
    )
    output = format_transcription(result, ResponseFormat.SRT)
    assert "00:00:01,234 --> 00:00:03,567" in output
