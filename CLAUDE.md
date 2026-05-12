# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (dev)
pip install -e ".[dev]"

# Run all tests
pytest -v

# Run single test file / specific test
pytest tests/test_config.py -v
pytest tests/api/test_audio.py::test_transcription_json_format -v

# Lint
ruff check .
ruff check . --fix

# Start server (direct, for development)
python main.py
python main.py --port 9000 --model-path mlx-community/whisper-large-v3-turbo

# macOS service management (launchd)
mlx-speech-server install    # create venv at ~/.local/venvs/mlx-speech-server, register plist
mlx-speech-server start      # bootstrap + kickstart; auto-installs if needed
mlx-speech-server stop
mlx-speech-server restart
mlx-speech-server status     # PID, port, /health, /v1/queue/stats
mlx-speech-server logs       # last 30 lines of stdout/stderr
mlx-speech-server upgrade    # git pull (local) or pip install --upgrade (PyPI)
mlx-speech-server uninstall  # removes plist, keeps venv
```

## Service Configuration

The launchd service reads config from `~/.config/mlx-speech-server/config.env` (KEY=VALUE format). `app/server.py:run_from_env()` is the entry point — it calls `ServerConfig.from_env()` and starts uvicorn. For dev use `python main.py` with CLI flags instead.

`WHISPER_*` env vars (all optional, match `ServerConfig` fields):
- `WHISPER_HOST`, `WHISPER_PORT` (default 8000)
- `WHISPER_MODEL_PATH` (default `mlx-community/whisper-large-v3-turbo`)
- `WHISPER_QUANTIZE` (int, optional)
- `WHISPER_MEMORY_CLEANUP_INTERVAL` (default 20 requests)
- `WHISPER_QUEUE_MAX_SIZE` (default 10), `WHISPER_QUEUE_TIMEOUT` (default 300s)
- `WHISPER_LOG_LEVEL` (default `info`)

## Architecture

Request flow: `HTTP → Router → Registry → Handler → Worker → mlx_whisper → Formatter → Response`

**Handler abstraction** (`app/handlers/base.py`): `BaseHandler` ABC defines lifecycle (initialize/cleanup/model_info). Capability mixins (`AudioCapable`) define what a handler can do. API routers check capabilities via `isinstance`. To add a new model type: implement `BaseHandler` + mixin, add a router, register in lifespan. Existing code unchanged.

**InferenceWorker** (`app/worker.py`): Single-thread `ThreadPoolExecutor` + `asyncio.Semaphore(1)` serializes all inference. Blocking `mlx_whisper.transcribe()` runs via `loop.run_in_executor()` to keep the event loop responsive. `QueueFullError` raised immediately when `_count >= max_size`; `QueueTimeoutError` via `asyncio.wait_for()`.

**Router factory pattern** (`app/api/*.py`): Each endpoint module exports a `create_*_router(dependencies)` function returning `APIRouter`. Dependencies (registry, worker) injected as constructor args, not FastAPI DI. This makes routers testable with mocks/fakes.

**App factory** (`app/server.py`): `create_app(config, registry, worker)` wires routers, exception handlers, health endpoint, and memory cleanup middleware. `run(config)` creates all components and starts uvicorn. Lifespan handles model registration on startup and cleanup on shutdown.

**macOS service** (`app/service.py` + `app/cli.py`): Manages a dedicated venv at `~/.local/venvs/mlx-speech-server` and a launchd plist at `~/Library/LaunchAgents/com.local.mlx-speech-server.plist`. Detects local vs PyPI install via `direct_url.json` metadata — local installs `pip install -e <project_dir>`, PyPI installs `pip install mlx-speech-server`. CLI commands are registered as `mlx-speech-server` and `mlx-speech-server-run` entry points in `pyproject.toml`.

**Schemas** (`app/schemas/`): `audio.py` defines `TranscriptionRequest` (Pydantic, for HTTP layer), `TranscriptionParams` (dataclass, passed to handler), `TranscriptionResult`/`SegmentResult` (dataclass, returned from handler). `common.py` defines `ModelCard` (returned by `model_info()`).

## Key Conventions

- **Metal cleanup**: `mx.clear_cache()` + `gc.collect()` on startup, shutdown, and every N requests (middleware in `server.py`). Required to prevent unified memory exhaustion on Apple Silicon.
- **Streaming is segment-level**: `mlx_whisper` has no token-level streaming. `transcribe_stream()` runs full inference then yields segments as SSE.
- **Temp file lifecycle**: `save_upload_file()` creates temp file, caller cleans up in `finally` block via `cleanup_temp_file()`.
- **Error format**: All errors return `{"error": {"message", "type", "code"}}` (OpenAI-compatible). Custom exceptions (`QueueFullError`, `ModelNotFoundError`, etc.) mapped to HTTP codes in `server.py` exception handlers. The audio router returns `JSONResponse` directly (not `HTTPException`) to control the response body shape.
- **Testing**: Mock `mlx_whisper` with `patch("app.handlers.whisper.mlx_whisper")`. Use `FakeAudioHandler` (concrete test double) for API tests. `tmp_wav_file` fixture in `conftest.py` creates valid WAV files. pytest `asyncio_mode = "auto"`.
- **Config precedence**: CLI flags > `WHISPER_*` env vars > dataclass defaults.
- **Language normalization** (`app/whisper_language.py`): BCP 47 / locale codes are normalized to ISO 639-1 before passing to `mlx_whisper` — `zh-TW` → `zh`, `en-US` → `en`. Validated against `mlx_whisper.tokenizer.LANGUAGES`. `None` / blank = auto-detect. Returns `(code, error_message)` tuple; error is non-`None` only for invalid codes.
