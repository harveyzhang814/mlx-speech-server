# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run all tests
pytest -v

# Run single test file / specific test
pytest tests/test_config.py -v
pytest tests/api/test_audio.py::test_transcription_json_format -v

# Lint
ruff check .
ruff check . --fix

# Start server
python main.py
python main.py --port 9000 --model-path mlx-community/whisper-large-v3-turbo
```

## Architecture

Request flow: `HTTP → Router → Registry → Handler → Worker → mlx_whisper → Formatter → Response`

**Handler abstraction** (`app/handlers/base.py`): `BaseHandler` ABC defines lifecycle (initialize/cleanup/model_info). Capability mixins (`AudioCapable`) define what a handler can do. API routers check capabilities via `isinstance`. To add a new model type: implement `BaseHandler` + mixin, add a router, register in lifespan. Existing code unchanged.

**InferenceWorker** (`app/worker.py`): Single-thread `ThreadPoolExecutor` + `asyncio.Semaphore(1)` serializes all inference. Blocking `mlx_whisper.transcribe()` runs via `loop.run_in_executor()` to keep the event loop responsive. `QueueFullError` raised immediately when `_count >= max_size`; `QueueTimeoutError` via `asyncio.wait_for()`.

**Router factory pattern** (`app/api/*.py`): Each endpoint module exports a `create_*_router(dependencies)` function returning `APIRouter`. Dependencies (registry, worker) injected as constructor args, not FastAPI DI. This makes routers testable with mocks/fakes.

**App factory** (`app/server.py`): `create_app(config, registry, worker)` wires routers, exception handlers, health endpoint, and memory cleanup middleware. `run(config)` creates all components and starts uvicorn. Lifespan handles model registration on startup and cleanup on shutdown.

## Key Conventions

- **Metal cleanup**: `mx.clear_cache()` + `gc.collect()` on startup, shutdown, and every N requests (middleware in `server.py`). Required to prevent unified memory exhaustion on Apple Silicon.
- **Streaming is segment-level**: `mlx_whisper` has no token-level streaming. `transcribe_stream()` runs full inference then yields segments as SSE.
- **Temp file lifecycle**: `save_upload_file()` creates temp file, caller cleans up in `finally` block via `cleanup_temp_file()`.
- **Error format**: All errors return `{"error": {"message", "type", "code"}}` (OpenAI-compatible). Custom exceptions (`QueueFullError`, `ModelNotFoundError`, etc.) mapped to HTTP codes in `server.py` exception handlers. The audio router returns `JSONResponse` directly (not `HTTPException`) to control the response body shape.
- **Testing**: Mock `mlx_whisper` with `patch("app.handlers.whisper.mlx_whisper")`. Use `FakeAudioHandler` (concrete test double) for API tests. `tmp_wav_file` fixture in `conftest.py` creates valid WAV files. pytest `asyncio_mode = "auto"`.
- **Config precedence**: CLI flags > `WHISPER_*` env vars > dataclass defaults.
