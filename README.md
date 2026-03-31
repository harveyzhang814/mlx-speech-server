# mlx-whisper-server

OpenAI-compatible Whisper transcription API server, running natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

## Features

- **OpenAI API compatible** — drop-in replacement for `POST /v1/audio/transcriptions`
- **All response formats** — `json`, `text`, `verbose_json`, `srt`, `vtt`
- **Streaming** — segment-level SSE via `stream=true`
- **Apple Silicon optimized** — Metal GPU acceleration, periodic `mx.clear_cache()` to manage unified memory
- **Request queue** — configurable max size and timeout, with `GET /v1/queue/stats` monitoring
- **Extensible** — handler abstraction supports future model types (LLM, embeddings, etc.)

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+

## Installation

```bash
git clone https://github.com/your-org/mlx-whisper-server.git
cd mlx-whisper-server
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

```bash
# Start with defaults (whisper-large-v3-turbo on port 8000)
python main.py

# Custom port and model
python main.py --port 9000 --model-path mlx-community/whisper-large-v3-turbo

# 4-bit quantized model (lower memory usage)
python main.py --model-path mlx-community/whisper-large-v3-turbo-q4
```

The model downloads automatically from HuggingFace on first request.

## API Reference

### Health Check

```
GET /health
```

```json
{"status": "ok", "model": "whisper-large-v3-turbo"}
```

### List Models

```
GET /v1/models
```

```json
{
  "object": "list",
  "data": [
    {"id": "whisper-large-v3-turbo", "object": "model", "owned_by": "local"}
  ]
}
```

### Queue Stats

```
GET /v1/queue/stats
```

```json
{"queue_size": 0, "queue_max_size": 10, "active": false}
```

### Audio Transcription

```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

**Parameters:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | Audio file (mp3, wav, m4a, ogg, flac, aac, webm) |
| `model` | string | Yes | Model ID, e.g. `whisper-large-v3-turbo` |
| `language` | string | No | ISO 639-1 code (auto-detect if omitted) |
| `prompt` | string | No | Context hint for transcription |
| `response_format` | string | No | `json` (default), `text`, `verbose_json`, `srt`, `vtt` |
| `temperature` | float | No | 0.0-1.0, default 0.0 |
| `stream` | bool | No | Enable SSE streaming, default false |

**Examples:**

```bash
# JSON (default)
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-large-v3-turbo

# SRT subtitles
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-large-v3-turbo \
  -F response_format=srt

# Streaming
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-large-v3-turbo \
  -F stream=true

# With language hint
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-large-v3-turbo \
  -F language=zh
```

**Response formats:**

`json`:
```json
{"text": "Hello world"}
```

`verbose_json`:
```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 5.52,
  "text": "Hello world",
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.1, "text": " Hello world", "no_speech_prob": 0.01}
  ]
}
```

`srt`:
```
1
00:00:00,000 --> 00:00:02,100
 Hello world
```

`vtt`:
```
WEBVTT

00:00:00.000 --> 00:00:02.100
 Hello world
```

**Streaming SSE:**
```
data: {"text": " Hello world"}

data: [DONE]
```

**Errors:**

| Status | Code | Cause |
|---|---|---|
| 400 | `model_not_found` | Unknown model ID |
| 400 | `invalid_response_format` | Unsupported format |
| 415 | `unsupported_audio_format` | File type not supported |
| 503 | `queue_full` | Too many concurrent requests |
| 503 | `queue_timeout` | Request waited too long |

All errors follow OpenAI format:
```json
{"error": {"message": "...", "type": "...", "code": "..."}}
```

## Configuration

Configuration via CLI flags or environment variables (CLI takes priority):

| CLI Flag | Env Var | Default | Description |
|---|---|---|---|
| `--host` | `WHISPER_HOST` | `0.0.0.0` | Bind address |
| `--port` | `WHISPER_PORT` | `8000` | Bind port |
| `--model-path` | `WHISPER_MODEL_PATH` | `mlx-community/whisper-large-v3-turbo` | HuggingFace repo or local path |
| `--quantize` | `WHISPER_QUANTIZE` | None | Use a pre-quantized model (4/8) |
| `--queue-max-size` | `WHISPER_QUEUE_MAX_SIZE` | `10` | Max queued requests before 503 |
| `--queue-timeout` | `WHISPER_QUEUE_TIMEOUT` | `300` | Seconds before queue timeout |
| `--memory-cleanup-interval` | `WHISPER_MEMORY_CLEANUP_INTERVAL` | `20` | Clear Metal cache every N requests |
| `--log-level` | `WHISPER_LOG_LEVEL` | `info` | Log level (debug/info/warning/error) |

## Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo",
        file=f,
        response_format="verbose_json",
    )
print(result.text)
```

## Architecture

```
main.py (CLI)
  -> app/server.py (FastAPI app factory, lifespan, Metal cleanup middleware)
       -> app/api/audio.py     POST /v1/audio/transcriptions
       -> app/api/models.py    GET  /v1/models
       -> app/api/queue.py     GET  /v1/queue/stats
       -> app/registry.py      model_id -> handler lookup
       -> app/worker.py        single-thread inference queue
       -> app/handlers/
            base.py            BaseHandler ABC + AudioCapable mixin
            whisper.py         WhisperHandler (mlx_whisper.transcribe)
       -> app/schemas/         Pydantic models + dataclasses
       -> app/formatters.py    json/text/verbose_json/srt/vtt conversion
       -> app/audio.py         upload save/validate/cleanup
```

**Extensibility:** To add a new model type, implement `BaseHandler` + a capability mixin (e.g. `ChatCapable`), add an API router, and register in the lifespan. Existing code requires no changes.

## Development

```bash
# Run tests
pytest -v

# Run with auto-reload (development)
uvicorn app.server:create_app --factory --reload

# Lint
ruff check .
```

## License

MIT
