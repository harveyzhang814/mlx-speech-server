from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel


class ResponseFormat(str, Enum):
    JSON = "json"
    TEXT = "text"
    VERBOSE_JSON = "verbose_json"
    SRT = "srt"
    VTT = "vtt"


class TranscriptionRequest(BaseModel):
    model: str
    language: str | None = None
    prompt: str | None = None
    response_format: ResponseFormat = ResponseFormat.JSON
    temperature: float = 0.0
    stream: bool = False


@dataclass
class TranscriptionParams:
    language: str | None = None
    prompt: str | None = None
    temperature: float = 0.0


@dataclass
class SegmentResult:
    id: int
    start: float
    end: float
    text: str
    no_speech_prob: float


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    segments: list[SegmentResult] = field(default_factory=list)
