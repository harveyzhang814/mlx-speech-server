from __future__ import annotations
import json
from app.schemas.audio import ResponseFormat, TranscriptionResult


def format_transcription(result: TranscriptionResult, fmt: ResponseFormat) -> str:
    if fmt == ResponseFormat.JSON:
        return json.dumps({"text": result.text})
    if fmt == ResponseFormat.TEXT:
        return result.text
    if fmt == ResponseFormat.VERBOSE_JSON:
        return _to_verbose_json(result)
    if fmt == ResponseFormat.SRT:
        return _to_srt(result)
    if fmt == ResponseFormat.VTT:
        return _to_vtt(result)
    raise ValueError(f"Unknown response format: {fmt}")


def _to_verbose_json(result: TranscriptionResult) -> str:
    data = {
        "task": "transcribe",
        "language": result.language,
        "duration": result.duration,
        "text": result.text,
        "segments": [
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "no_speech_prob": seg.no_speech_prob,
            }
            for seg in result.segments
        ],
    }
    return json.dumps(data)


def _to_srt(result: TranscriptionResult) -> str:
    lines: list[str] = []
    for i, seg in enumerate(result.segments, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_ts(seg.start)} --> {_srt_ts(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def _to_vtt(result: TranscriptionResult) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for seg in result.segments:
        lines.append(f"{_vtt_ts(seg.start)} --> {_vtt_ts(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def _srt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = round((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = round((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
