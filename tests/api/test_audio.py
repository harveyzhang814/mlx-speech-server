from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from app.api.audio import create_audio_router
from app.registry import ModelRegistry
from app.handlers.base import BaseHandler, AudioCapable
from app.schemas.common import ModelCard
from app.schemas.audio import TranscriptionParams, TranscriptionResult, SegmentResult
from typing import AsyncGenerator


FAKE_RESULT = TranscriptionResult(
    text=" Hello world",
    language="en",
    duration=2.1,
    segments=[SegmentResult(id=0, start=0.0, end=2.1, text=" Hello world", no_speech_prob=0.01)],
)


class FakeAudioHandler(BaseHandler, AudioCapable):
    async def initialize(self) -> None: pass
    async def cleanup(self) -> None: pass
    def model_info(self) -> ModelCard:
        return ModelCard(id="whisper-large-v3-turbo")
    async def transcribe(self, audio_path: Path, params: TranscriptionParams) -> TranscriptionResult:
        return FAKE_RESULT
    async def transcribe_stream(self, audio_path: Path, params: TranscriptionParams) -> AsyncGenerator[str, None]:
        yield 'data: {"text": " Hello world"}\n\n'
        yield "data: [DONE]\n\n"


class CapturingAudioHandler(FakeAudioHandler):
    def __init__(self) -> None:
        self.last_params: TranscriptionParams | None = None

    async def transcribe(self, audio_path: Path, params: TranscriptionParams) -> TranscriptionResult:
        self.last_params = params
        return FAKE_RESULT


def _make_client() -> TestClient:
    registry = ModelRegistry()
    registry.register("whisper-large-v3-turbo", FakeAudioHandler())
    worker = MagicMock()
    app = FastAPI()
    app.include_router(create_audio_router(registry, worker))
    return TestClient(app)


def _audio_form(tmp_wav_file: Path, extra: dict = None) -> dict:
    data = {"model": "whisper-large-v3-turbo"}
    if extra:
        data.update(extra)
    files = {"file": ("audio.wav", tmp_wav_file.read_bytes(), "audio/wav")}
    return {"data": data, "files": files}


def test_transcription_json_format(tmp_wav_file):
    client = _make_client()
    resp = client.post("/v1/audio/transcriptions", **_audio_form(tmp_wav_file))
    assert resp.status_code == 200
    assert resp.json() == {"text": " Hello world"}


def test_transcription_text_format(tmp_wav_file):
    client = _make_client()
    form = _audio_form(tmp_wav_file, {"response_format": "text"})
    resp = client.post("/v1/audio/transcriptions", **form)
    assert resp.status_code == 200
    assert resp.text == " Hello world"


def test_transcription_verbose_json_format(tmp_wav_file):
    client = _make_client()
    form = _audio_form(tmp_wav_file, {"response_format": "verbose_json"})
    resp = client.post("/v1/audio/transcriptions", **form)
    assert resp.status_code == 200
    data = resp.json()
    assert "segments" in data
    assert data["language"] == "en"


def test_transcription_srt_format(tmp_wav_file):
    client = _make_client()
    form = _audio_form(tmp_wav_file, {"response_format": "srt"})
    resp = client.post("/v1/audio/transcriptions", **form)
    assert resp.status_code == 200
    assert "00:00:00,000 --> 00:00:02,100" in resp.text


def test_transcription_vtt_format(tmp_wav_file):
    client = _make_client()
    form = _audio_form(tmp_wav_file, {"response_format": "vtt"})
    resp = client.post("/v1/audio/transcriptions", **form)
    assert resp.status_code == 200
    assert resp.text.startswith("WEBVTT")


def test_transcription_normalizes_zh_cn_language_param(tmp_wav_file):
    registry = ModelRegistry()
    handler = CapturingAudioHandler()
    registry.register("whisper-large-v3-turbo", handler)
    worker = MagicMock()
    app = FastAPI()
    app.include_router(create_audio_router(registry, worker))
    client = TestClient(app)
    resp = client.post("/v1/audio/transcriptions", **_audio_form(tmp_wav_file, {"language": "zh-CN"}))
    assert resp.status_code == 200
    assert handler.last_params is not None
    assert handler.last_params.language == "zh"


def test_transcription_unsupported_language_returns_400(tmp_wav_file):
    client = _make_client()
    resp = client.post("/v1/audio/transcriptions", **_audio_form(tmp_wav_file, {"language": "qq"}))
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "unsupported_language"
    assert "invalid_request_error" == body["error"]["type"]


def test_transcription_unknown_model_returns_400(tmp_wav_file):
    registry = ModelRegistry()
    worker = MagicMock()
    app = FastAPI()
    app.include_router(create_audio_router(registry, worker))
    client = TestClient(app)
    resp = client.post("/v1/audio/transcriptions", **_audio_form(tmp_wav_file, {"model": "gpt-4"}))
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "model_not_found"


def test_transcription_unsupported_audio_format(tmp_wav_file):
    client = _make_client()
    files = {"file": ("audio.xyz", b"fake", "application/octet-stream")}
    data = {"model": "whisper-large-v3-turbo"}
    resp = client.post("/v1/audio/transcriptions", data=data, files=files)
    assert resp.status_code == 415


def test_transcription_streaming_returns_sse(tmp_wav_file):
    client = _make_client()
    form = _audio_form(tmp_wav_file, {"stream": "true"})
    with client.stream("POST", "/v1/audio/transcriptions", **form) as resp:
        assert resp.status_code == 200
        content = resp.read().decode()
    assert "data:" in content
    assert "[DONE]" in content
