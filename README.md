# mlx-whisper-server

<p align="center">
  <a href="README_zh.md">中文文档</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#api-reference">API</a> ·
  <a href="#configuration">Configuration</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/API-OpenAI%20Compatible-412991?logo=openai&logoColor=white" alt="OpenAI Compatible">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</p>

OpenAI-compatible Whisper transcription API server running natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

## Features

- **OpenAI API compatible** — drop-in replacement for `POST /v1/audio/transcriptions`
- **All response formats** — `json`, `text`, `verbose_json`, `srt`, `vtt`
- **Streaming** — segment-level SSE via `stream=true`
- **Apple Silicon optimized** — Metal GPU acceleration with periodic `mx.clear_cache()` for unified memory management
- **Request queue** — configurable max size and timeout, with `GET /v1/queue/stats` monitoring
- **Extensible** — handler abstraction supports future model types (LLM, embeddings, etc.)

> [!NOTE]
> `mlx_whisper` does not support token-level streaming. Full inference runs first, then segments are yielded sequentially over SSE.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+

## Installation

```bash
git clone https://github.com/harveyzhang814/mlx-whisper-server.git
cd mlx-whisper-server
```

## Quick Start

### Managed service (recommended)

Install the CLI, then use `mlx-speech-server` subcommands to manage the launchd service (auto-start on login, auto-restart on crash).

**From local clone:**
```bash
git clone https://github.com/your-org/mlx-whisper-server.git
cd mlx-whisper-server
pipx install .
mlx-speech-server install
mlx-speech-server start
mlx-speech-server status
```

**From PyPI** (once published):
```bash
pip install mlx-speech-server
mlx-speech-server install
mlx-speech-server start
```

**All service commands:**

| Command | Description |
| :--- | :--- |
| `mlx-speech-server install` | Create service venv, install deps, register launchd agent |
| `mlx-speech-server uninstall` | Remove launchd agent (venv kept) |
| `mlx-speech-server upgrade` | Local clone: git pull + reinstall if updated; PyPI: pip upgrade |
| `mlx-speech-server start` | Start service (auto-installs if needed) |
| `mlx-speech-server stop` | Stop service |
| `mlx-speech-server restart` | Restart service |
| `mlx-speech-server status` | Show PID, health check, queue stats |
| `mlx-speech-server logs` | Show recent log output |

Default paths:
- Service venv: `~/.local/venvs/mlx-speech-server/`
- Logs: `~/.local/logs/mlx-speech-server/`

### Manual start

```bash
python3 -m venv ~/.local/venvs/mlx-whisper-server
source ~/.local/venvs/mlx-whisper-server/bin/activate
pip install -e "."

# Start with defaults (whisper-large-v3-turbo on port 8000)
python main.py

# Custom port and model
python main.py --port 9000 --model-path mlx-community/whisper-large-v3-turbo

# 4-bit quantized model (lower memory usage)
python main.py --model-path mlx-community/whisper-large-v3-turbo-q4
```

The model downloads automatically from HuggingFace on first request.

## Supported Models

All [mlx-community](https://huggingface.co/mlx-community) Whisper models are supported — just change `--model-path`:

| Model | Size | Min RAM | Recommended | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| `mlx-community/whisper-large-v3-turbo` | ~1.6 GB | 8 GB | M1 Pro+ | **Default** — best speed/quality balance |
| `mlx-community/whisper-large-v3-mlx` | ~3 GB | 16 GB | M1 Pro+ | Highest quality, slower |
| `mlx-community/whisper-large-v3-mlx-4bit` | ~0.9 GB | 8 GB | M1+ | Low memory, slightly lower quality |
| `mlx-community/whisper-large-v3-turbo-8bit` | ~0.8 GB | 8 GB | M1+ | Turbo quantized, lower memory |
| `mlx-community/distil-whisper-large-v3` | ~1.5 GB | 8 GB | M1 Pro+ | Distilled, faster inference |
| `mlx-community/whisper-medium-mlx` | ~1.5 GB | 8 GB | M1+ | Mid-size, multilingual |
| `mlx-community/whisper-small-mlx` | ~0.5 GB | 8 GB | M1+ | Lightweight, good for real-time |
| `mlx-community/whisper-tiny-mlx` | ~0.15 GB | 8 GB | M1+ | Smallest and fastest, limited quality |

> [!TIP]
> RAM requirements include model weights + inference overhead (~2–3× model size). On 8 GB devices, prefer quantized or small models to leave headroom for the OS. Apple Silicon uses unified memory shared between CPU and GPU — model weights, inference buffers, and system memory all compete for the same pool.

Additional variants: English-only (`.en`), quantized (2/4/8-bit), FP32, language-specific fine-tunes (German, etc.). See the full list at [mlx-community on HuggingFace](https://huggingface.co/collections/mlx-community/whisper).

## API Reference

Once running, interactive docs are available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

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
| :--- | :--- | :---: | :--- |
| `file` | file | ✓ | Audio file (mp3, wav, m4a, ogg, flac, aac, webm) |
| `model` | string | ✓ | Model ID, e.g. `whisper-large-v3-turbo` |
| `language` | string | — | ISO 639-1 code (auto-detect if omitted) |
| `prompt` | string | — | Context hint for transcription |
| `response_format` | string | — | `json` (default), `text`, `verbose_json`, `srt`, `vtt` |
| `temperature` | float | — | 0.0–1.0, default `0.0` |
| `stream` | bool | — | Enable SSE streaming, default `false` |

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

<details>
<summary><strong>Response format examples</strong></summary>

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

Streaming SSE:
```
data: {"text": " Hello world"}

data: [DONE]
```

</details>

**Error codes:**

| Status | Code | Cause |
| :---: | :--- | :--- |
| 400 | `model_not_found` | Unknown model ID |
| 400 | `invalid_response_format` | Unsupported format value |
| 415 | `unsupported_audio_format` | File type not supported |
| 503 | `queue_full` | Too many concurrent requests |
| 503 | `queue_timeout` | Request waited too long |

All errors follow OpenAI format:
```json
{"error": {"message": "...", "type": "...", "code": "..."}}
```

## Configuration

Configure via CLI flags or environment variables. CLI flags take priority.

| CLI Flag | Env Var | Default | Description |
| :--- | :--- | :--- | :--- |
| `--host` | `WHISPER_HOST` | `0.0.0.0` | Bind address |
| `--port` | `WHISPER_PORT` | `8000` | Bind port |
| `--model-path` | `WHISPER_MODEL_PATH` | `mlx-community/whisper-large-v3-turbo` | HuggingFace repo or local path |
| `--quantize` | `WHISPER_QUANTIZE` | — | Pre-quantized model bits (`4` or `8`) |
| `--queue-max-size` | `WHISPER_QUEUE_MAX_SIZE` | `10` | Max queued requests before 503 |
| `--queue-timeout` | `WHISPER_QUEUE_TIMEOUT` | `300` | Seconds before queue timeout |
| `--memory-cleanup-interval` | `WHISPER_MEMORY_CLEANUP_INTERVAL` | `20` | Clear Metal cache every N requests |
| `--log-level` | `WHISPER_LOG_LEVEL` | `info` | Log level (`debug`/`info`/`warning`/`error`) |

Create `~/.config/mlx-speech-server/config.env` (auto-created by `mlx-speech-server install`):

```bash
WHISPER_PORT=8000
WHISPER_MODEL_PATH=mlx-community/whisper-large-v3-turbo
WHISPER_QUEUE_MAX_SIZE=10
```

After editing, restart the service:

```bash
mlx-speech-server restart
```

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

The API is fully compatible with OpenAI's `/v1/audio/transcriptions` — any client that supports the OpenAI SDK just needs a `base_url` change.

## Architecture

<details>
<summary><strong>Request flow and module map</strong></summary>

```
HTTP → Router → Registry → Handler → Worker → mlx_whisper → Formatter → Response

main.py (CLI)
  └─ app/server.py         FastAPI app factory, lifespan, Metal cleanup middleware
       ├─ app/api/audio.py       POST /v1/audio/transcriptions
       ├─ app/api/models.py      GET  /v1/models
       ├─ app/api/queue.py       GET  /v1/queue/stats
       ├─ app/registry.py        model_id → handler lookup
       ├─ app/worker.py          single-thread inference queue
       ├─ app/handlers/
       │    ├─ base.py           BaseHandler ABC + AudioCapable mixin
       │    └─ whisper.py        WhisperHandler (mlx_whisper.transcribe)
       ├─ app/schemas/           Pydantic models + dataclasses
       ├─ app/formatters.py      json/text/verbose_json/srt/vtt conversion
       └─ app/audio.py           upload save/validate/cleanup
```

**Extensibility:** To add a new model type, implement `BaseHandler` + a capability mixin (e.g. `ChatCapable`), add an API router, and register it in the lifespan. Existing code requires no changes.

</details>

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Run a single test
pytest tests/api/test_audio.py::test_transcription_json_format -v

# Lint
ruff check .
ruff check . --fix

# Dev server with auto-reload
uvicorn app.server:create_app --factory --reload
```

## License

[MIT](LICENSE)
