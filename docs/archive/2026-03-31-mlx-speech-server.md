# mlx-whisper-server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an OpenAI-compatible HTTP API server that runs `whisper-large-v3-turbo` via MLX on Apple Silicon, with streaming, all response formats, request queue management, and a clean handler abstraction for future model types.

**Architecture:** FastAPI app with a handler abstraction layer (`BaseHandler` + capability Mixins) backed by a single-thread `InferenceWorker` that serializes Metal GPU work. A `ModelRegistry` maps model IDs to handlers. All endpoints are OpenAI-compatible. macOS-specific optimizations (Metal cache clearing, thread isolation) are baked in throughout.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, mlx-whisper, mlx, pydantic v2, click, loguru, pytest, pytest-asyncio, httpx

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Project metadata, dependencies, dev tools |
| `main.py` | CLI entry point (click) |
| `app/__init__.py` | Package init |
| `app/config.py` | `ServerConfig` dataclass, env var + CLI config |
| `app/worker.py` | `InferenceWorker`: asyncio queue + single ThreadPoolExecutor |
| `app/registry.py` | `ModelRegistry`: model_id → handler mapping |
| `app/audio.py` | Temp file save, format validation, cleanup |
| `app/formatters.py` | Convert `TranscriptionResult` to json/text/verbose_json/srt/vtt |
| `app/handlers/__init__.py` | Re-exports |
| `app/handlers/base.py` | `BaseHandler` ABC + `AudioCapable` Mixin |
| `app/handlers/whisper.py` | `WhisperHandler`: wraps `mlx_whisper.transcribe` |
| `app/schemas/__init__.py` | Re-exports |
| `app/schemas/common.py` | `ModelCard`, `ErrorResponse`, `QueueStats` |
| `app/schemas/audio.py` | `TranscriptionRequest`, `TranscriptionParams`, `TranscriptionResult`, `SegmentResult` |
| `app/api/__init__.py` | Re-exports |
| `app/api/models.py` | `GET /v1/models` router factory |
| `app/api/queue.py` | `GET /v1/queue/stats` router factory |
| `app/api/audio.py` | `POST /v1/audio/transcriptions` router factory |
| `app/server.py` | `create_app()`, lifespan, exception handlers, memory cleanup middleware |
| `tests/conftest.py` | Shared pytest fixtures |
| `tests/test_config.py` | Config tests |
| `tests/test_schemas.py` | Schema validation tests |
| `tests/test_audio.py` | Audio utils tests |
| `tests/test_worker.py` | InferenceWorker tests |
| `tests/test_registry.py` | ModelRegistry tests |
| `tests/test_formatters.py` | Response format conversion tests |
| `tests/handlers/test_whisper.py` | WhisperHandler tests (mocked mlx_whisper) |
| `tests/api/test_models.py` | /v1/models endpoint tests |
| `tests/api/test_queue.py` | /v1/queue/stats endpoint tests |
| `tests/api/test_audio.py` | /v1/audio/transcriptions endpoint tests |
| `tests/api/test_server.py` | Health endpoint + exception handler tests |

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `app/__init__.py`
- Create: `app/handlers/__init__.py`
- Create: `app/schemas/__init__.py`
- Create: `app/api/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/handlers/__init__.py`
- Create: `tests/api/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mlx-whisper-server"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.29",
    "mlx-whisper>=0.4",
    "mlx>=0.16",
    "pydantic>=2.0",
    "python-multipart",
    "click>=8.0",
    "loguru>=0.7",
]

[project.scripts]
mlx-whisper-server = "main:cli"

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",
    "ruff>=0.4",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 2: Create all `__init__.py` files and `tests/conftest.py`**

```python
# app/__init__.py  (empty)
# app/handlers/__init__.py  (empty)
# app/schemas/__init__.py  (empty)
# app/api/__init__.py  (empty)
# tests/__init__.py  (empty)
# tests/handlers/__init__.py  (empty)
# tests/api/__init__.py  (empty)
```

```python
# tests/conftest.py
import pytest
from pathlib import Path
import tempfile
import wave
import struct


@pytest.fixture
def tmp_wav_file() -> Path:
    """Create a minimal valid WAV file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    # Write a minimal 1-second 16kHz mono WAV
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        samples = struct.pack("<" + "h" * 16000, *([0] * 16000))
        wav.writeframes(samples)
    yield path
    path.unlink(missing_ok=True)
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: No errors. `pytest --collect-only` shows 0 tests collected.

- [ ] **Step 4: Commit**

```bash
git init
git add pyproject.toml app/ tests/
git commit -m "chore: project scaffold with dependencies and test infrastructure"
```

---

## Task 2: Config

**Files:**
- Create: `app/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config.py
import os
import pytest
from app.config import ServerConfig


def test_default_values():
    config = ServerConfig()
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.model_path == "mlx-community/whisper-large-v3-turbo"
    assert config.quantize is None
    assert config.memory_cleanup_interval == 20
    assert config.queue_max_size == 10
    assert config.queue_timeout == 300.0
    assert config.log_level == "info"


def test_from_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("WHISPER_PORT", "9000")
    monkeypatch.setenv("WHISPER_MODEL_PATH", "/local/whisper")
    monkeypatch.setenv("WHISPER_QUANTIZE", "4")
    monkeypatch.setenv("WHISPER_QUEUE_MAX_SIZE", "5")
    monkeypatch.setenv("WHISPER_QUEUE_TIMEOUT", "60.0")
    monkeypatch.setenv("WHISPER_MEMORY_CLEANUP_INTERVAL", "10")
    monkeypatch.setenv("WHISPER_LOG_LEVEL", "debug")

    config = ServerConfig.from_env()
    assert config.port == 9000
    assert config.model_path == "/local/whisper"
    assert config.quantize == 4
    assert config.queue_max_size == 5
    assert config.queue_timeout == 60.0
    assert config.memory_cleanup_interval == 10
    assert config.log_level == "debug"


def test_from_env_uses_defaults_when_vars_absent(monkeypatch):
    for key in ["WHISPER_PORT", "WHISPER_MODEL_PATH", "WHISPER_QUANTIZE"]:
        monkeypatch.delenv(key, raising=False)
    config = ServerConfig.from_env()
    assert config.port == 8000
    assert config.model_path == "mlx-community/whisper-large-v3-turbo"
    assert config.quantize is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 3: Implement `app/config.py`**

```python
# app/config.py
from __future__ import annotations
import os
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "mlx-community/whisper-large-v3-turbo"
    quantize: int | None = None
    memory_cleanup_interval: int = 20
    queue_max_size: int = 10
    queue_timeout: float = 300.0
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> ServerConfig:
        def _int(key: str, default: int) -> int:
            val = os.environ.get(key)
            return int(val) if val is not None else default

        def _float(key: str, default: float) -> float:
            val = os.environ.get(key)
            return float(val) if val is not None else default

        def _optional_int(key: str) -> int | None:
            val = os.environ.get(key)
            return int(val) if val is not None else None

        return cls(
            host=os.environ.get("WHISPER_HOST", "0.0.0.0"),
            port=_int("WHISPER_PORT", 8000),
            model_path=os.environ.get(
                "WHISPER_MODEL_PATH", "mlx-community/whisper-large-v3-turbo"
            ),
            quantize=_optional_int("WHISPER_QUANTIZE"),
            memory_cleanup_interval=_int("WHISPER_MEMORY_CLEANUP_INTERVAL", 20),
            queue_max_size=_int("WHISPER_QUEUE_MAX_SIZE", 10),
            queue_timeout=_float("WHISPER_QUEUE_TIMEOUT", 300.0),
            log_level=os.environ.get("WHISPER_LOG_LEVEL", "info"),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add ServerConfig with env var support"
```

---

## Task 3: Schemas

**Files:**
- Create: `app/schemas/common.py`
- Create: `app/schemas/audio.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_schemas.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_schemas.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.schemas.common'`

- [ ] **Step 3: Implement `app/schemas/common.py`**

```python
# app/schemas/common.py
from __future__ import annotations
from pydantic import BaseModel, model_validator


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str


class ErrorResponse(BaseModel):
    error: dict

    def __init__(self, message: str, type: str, code: str):
        super().__init__(error={"message": message, "type": type, "code": code})


class QueueStats(BaseModel):
    queue_size: int
    queue_max_size: int
    active: bool
```

- [ ] **Step 4: Implement `app/schemas/audio.py`**

```python
# app/schemas/audio.py
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_schemas.py -v
```

Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add app/schemas/ tests/test_schemas.py
git commit -m "feat: add schemas for API request/response and internal types"
```

---

## Task 4: Audio Utils

**Files:**
- Create: `app/audio.py`
- Create: `tests/test_audio.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_audio.py
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from app.audio import (
    save_upload_file,
    cleanup_temp_file,
    SUPPORTED_FORMATS,
    UnsupportedAudioFormatError,
)


def test_supported_formats_includes_common_types():
    assert ".wav" in SUPPORTED_FORMATS
    assert ".mp3" in SUPPORTED_FORMATS
    assert ".m4a" in SUPPORTED_FORMATS
    assert ".flac" in SUPPORTED_FORMATS
    assert ".ogg" in SUPPORTED_FORMATS
    assert ".aac" in SUPPORTED_FORMATS
    assert ".webm" in SUPPORTED_FORMATS


@pytest.mark.asyncio
async def test_save_upload_file_returns_temp_path(tmp_wav_file):
    mock_upload = AsyncMock()
    mock_upload.filename = "audio.wav"
    mock_upload.read = AsyncMock(return_value=tmp_wav_file.read_bytes())

    result = await save_upload_file(mock_upload)
    try:
        assert result.exists()
        assert result.suffix == ".wav"
        assert result.read_bytes() == tmp_wav_file.read_bytes()
    finally:
        result.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_save_upload_file_raises_for_unsupported_format():
    mock_upload = AsyncMock()
    mock_upload.filename = "audio.xyz"
    mock_upload.read = AsyncMock(return_value=b"fake data")

    with pytest.raises(UnsupportedAudioFormatError):
        await save_upload_file(mock_upload)


@pytest.mark.asyncio
async def test_save_upload_file_raises_for_no_extension():
    mock_upload = AsyncMock()
    mock_upload.filename = "audiofile"
    mock_upload.read = AsyncMock(return_value=b"fake data")

    with pytest.raises(UnsupportedAudioFormatError):
        await save_upload_file(mock_upload)


def test_cleanup_temp_file_removes_file(tmp_path):
    f = tmp_path / "test.wav"
    f.write_bytes(b"data")
    assert f.exists()
    cleanup_temp_file(f)
    assert not f.exists()


def test_cleanup_temp_file_ignores_missing_file(tmp_path):
    f = tmp_path / "nonexistent.wav"
    # Should not raise
    cleanup_temp_file(f)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_audio.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.audio'`

- [ ] **Step 3: Implement `app/audio.py`**

```python
# app/audio.py
from __future__ import annotations
import tempfile
from pathlib import Path
from fastapi import UploadFile


SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".webm"}


class UnsupportedAudioFormatError(Exception):
    pass


async def save_upload_file(upload_file: UploadFile) -> Path:
    """Save an uploaded audio file to a temp path. Caller must clean up."""
    filename = upload_file.filename or ""
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_FORMATS:
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        content = await upload_file.read()
        tmp.write(content)
        return Path(tmp.name)


def cleanup_temp_file(path: Path) -> None:
    """Delete a temp file, silently ignoring errors."""
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_audio.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add app/audio.py tests/test_audio.py
git commit -m "feat: add audio upload handling with format validation"
```

---

## Task 5: Handler Base

**Files:**
- Create: `app/handlers/base.py`
- Create: `tests/handlers/test_base.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/handlers/test_base.py
import pytest
from pathlib import Path
from typing import AsyncGenerator
from app.handlers.base import BaseHandler, AudioCapable
from app.schemas.common import ModelCard
from app.schemas.audio import TranscriptionParams, TranscriptionResult


def test_base_handler_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseHandler()  # type: ignore


def test_audio_capable_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AudioCapable()  # type: ignore


def test_concrete_handler_must_implement_all_methods():
    """A class that only partially implements BaseHandler cannot be instantiated."""
    class PartialHandler(BaseHandler):
        async def initialize(self) -> None:
            pass
        # Missing cleanup() and model_info()

    with pytest.raises(TypeError):
        PartialHandler()


def test_concrete_handler_with_audio_capable_can_be_instantiated():
    class ConcreteHandler(BaseHandler, AudioCapable):
        async def initialize(self) -> None:
            pass

        async def cleanup(self) -> None:
            pass

        def model_info(self) -> ModelCard:
            return ModelCard(id="test-model")

        async def transcribe(self, audio_path: Path, params: TranscriptionParams) -> TranscriptionResult:
            return TranscriptionResult(text="", language="en", duration=0.0)

        async def transcribe_stream(
            self, audio_path: Path, params: TranscriptionParams
        ) -> AsyncGenerator[str, None]:
            yield "data: [DONE]\n\n"

    handler = ConcreteHandler()
    assert handler is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/handlers/test_base.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.handlers.base'`

- [ ] **Step 3: Implement `app/handlers/base.py`**

```python
# app/handlers/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator
from app.schemas.common import ModelCard
from app.schemas.audio import TranscriptionParams, TranscriptionResult


class BaseHandler(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        """Load model and any other startup work."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Release resources on shutdown."""

    @abstractmethod
    def model_info(self) -> ModelCard:
        """Return model metadata for /v1/models."""


class AudioCapable(ABC):
    """Mixin for handlers that support audio transcription.

    Implement this alongside BaseHandler to expose the handler
    to the /v1/audio/transcriptions endpoint.
    """

    @abstractmethod
    async def transcribe(
        self, audio_path: Path, params: TranscriptionParams
    ) -> TranscriptionResult:
        """Run full transcription and return complete result."""

    @abstractmethod
    async def transcribe_stream(
        self, audio_path: Path, params: TranscriptionParams
    ) -> AsyncGenerator[str, None]:
        """Yield SSE-formatted strings, one per segment, ending with 'data: [DONE]\\n\\n'."""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/handlers/test_base.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add app/handlers/base.py tests/handlers/test_base.py
git commit -m "feat: add BaseHandler and AudioCapable abstractions"
```

---

## Task 6: InferenceWorker

**Files:**
- Create: `app/worker.py`
- Create: `tests/test_worker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_worker.py
import asyncio
import pytest
from app.worker import InferenceWorker, QueueFullError, QueueTimeoutError


@pytest.mark.asyncio
async def test_submit_runs_function_and_returns_result():
    worker = InferenceWorker(max_size=5, timeout=5.0)
    try:
        result = await worker.submit(lambda: 42)
        assert result == 42
    finally:
        worker.stop()


@pytest.mark.asyncio
async def test_submit_propagates_exception():
    worker = InferenceWorker(max_size=5, timeout=5.0)
    try:
        with pytest.raises(ValueError, match="oops"):
            await worker.submit(lambda: (_ for _ in ()).throw(ValueError("oops")))
    finally:
        worker.stop()


@pytest.mark.asyncio
async def test_queue_full_raises_immediately():
    worker = InferenceWorker(max_size=1, timeout=30.0)
    # Use a long-running task to fill the executor
    import time
    blocker_started = asyncio.Event()

    def slow():
        blocker_started.set()  # signal in thread - not safe but ok for test
        time.sleep(5)
        return "done"

    # Submit slow task to occupy the single executor slot
    task = asyncio.create_task(worker.submit(slow))
    await asyncio.sleep(0.05)  # let executor pick up the task

    # Now queue is full (1 running, max_size=1), second submit should fail
    with pytest.raises(QueueFullError):
        await worker.submit(lambda: "second")

    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    finally:
        worker.stop()


@pytest.mark.asyncio
async def test_queue_size_reflects_pending_count():
    worker = InferenceWorker(max_size=10, timeout=10.0)
    assert worker.queue_size == 0
    assert worker.active is False
    worker.stop()


@pytest.mark.asyncio
async def test_timeout_raises_queue_timeout_error():
    worker = InferenceWorker(max_size=10, timeout=0.05)
    import time

    def slow():
        time.sleep(2)
        return "done"

    with pytest.raises(QueueTimeoutError):
        await worker.submit(slow)
    worker.stop()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_worker.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.worker'`

- [ ] **Step 3: Implement `app/worker.py`**

```python
# app/worker.py
from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class QueueFullError(Exception):
    pass


class QueueTimeoutError(Exception):
    pass


class InferenceWorker:
    """Serializes inference calls on a single background thread.

    Keeps the asyncio event loop responsive during blocking MLX/Metal compute.
    Enforces a maximum number of concurrent+waiting requests (max_size) and a
    per-request timeout so the server never silently stalls under load.
    """

    def __init__(self, max_size: int, timeout: float) -> None:
        self._max_size = max_size
        self._timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=1)
        # _execution_lock ensures only one fn runs at a time
        self._execution_lock = asyncio.Semaphore(1)
        self._count = 0      # total in system: waiting + executing
        self._active = False
        self._count_lock = asyncio.Lock()

    @property
    def queue_size(self) -> int:
        """Number of requests waiting (not yet executing)."""
        return max(0, self._count - (1 if self._active else 0))

    @property
    def active(self) -> bool:
        """True if a request is currently being inferred."""
        return self._active

    @property
    def max_size(self) -> int:
        return self._max_size

    async def submit(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Submit a blocking function for execution on the inference thread.

        Raises QueueFullError if max_size is reached.
        Raises QueueTimeoutError if the request waits longer than timeout.
        """
        async with self._count_lock:
            if self._count >= self._max_size:
                raise QueueFullError(
                    f"Inference queue is full ({self._max_size} requests). "
                    "Try again later."
                )
            self._count += 1

        try:
            try:
                await asyncio.wait_for(
                    self._execution_lock.acquire(), timeout=self._timeout
                )
            except asyncio.TimeoutError:
                raise QueueTimeoutError(
                    f"Request timed out after {self._timeout}s waiting in queue."
                )

            self._active = True
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self._executor, fn if not args else lambda: fn(*args)
                )
                return result
            finally:
                self._active = False
                self._execution_lock.release()
        finally:
            async with self._count_lock:
                self._count -= 1

    def stop(self) -> None:
        """Shut down the executor. Call during server lifespan shutdown."""
        self._executor.shutdown(wait=False)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_worker.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add app/worker.py tests/test_worker.py
git commit -m "feat: add InferenceWorker with queue size limit and timeout"
```

---

## Task 7: ModelRegistry

**Files:**
- Create: `app/registry.py`
- Create: `tests/test_registry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_registry.py
import pytest
from app.registry import ModelRegistry, ModelNotFoundError
from app.handlers.base import BaseHandler, AudioCapable
from app.schemas.common import ModelCard
from app.schemas.audio import TranscriptionParams, TranscriptionResult
from pathlib import Path
from typing import AsyncGenerator


class _FakeHandler(BaseHandler, AudioCapable):
    async def initialize(self) -> None: pass
    async def cleanup(self) -> None: pass
    def model_info(self) -> ModelCard:
        return ModelCard(id="fake-model")
    async def transcribe(self, audio_path: Path, params: TranscriptionParams) -> TranscriptionResult:
        return TranscriptionResult(text="", language="en", duration=0.0)
    async def transcribe_stream(self, audio_path: Path, params: TranscriptionParams) -> AsyncGenerator[str, None]:
        yield "data: [DONE]\n\n"


def test_register_and_get():
    registry = ModelRegistry()
    handler = _FakeHandler()
    registry.register("fake-model", handler)
    assert registry.get("fake-model") is handler


def test_get_unknown_model_raises():
    registry = ModelRegistry()
    with pytest.raises(ModelNotFoundError, match="fake-model"):
        registry.get("fake-model")


def test_list_models_returns_model_cards():
    registry = ModelRegistry()
    handler = _FakeHandler()
    registry.register("fake-model", handler)
    models = registry.list_models()
    assert len(models) == 1
    assert models[0].id == "fake-model"


def test_list_models_empty_registry():
    registry = ModelRegistry()
    assert registry.list_models() == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_registry.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.registry'`

- [ ] **Step 3: Implement `app/registry.py`**

```python
# app/registry.py
from __future__ import annotations
from app.handlers.base import BaseHandler
from app.schemas.common import ModelCard


class ModelNotFoundError(Exception):
    pass


class ModelRegistry:
    """Maps model IDs to handler instances."""

    def __init__(self) -> None:
        self._handlers: dict[str, BaseHandler] = {}

    def register(self, model_id: str, handler: BaseHandler) -> None:
        self._handlers[model_id] = handler

    def get(self, model_id: str) -> BaseHandler:
        try:
            return self._handlers[model_id]
        except KeyError:
            raise ModelNotFoundError(f"Model '{model_id}' not found.")

    def list_models(self) -> list[ModelCard]:
        return [h.model_info() for h in self._handlers.values()]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_registry.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add app/registry.py tests/test_registry.py
git commit -m "feat: add ModelRegistry for handler lookup and model listing"
```

---

## Task 8: WhisperHandler

**Files:**
- Create: `app/handlers/whisper.py`
- Create: `tests/handlers/test_whisper.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/handlers/test_whisper.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
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
def handler(worker, tmp_path):
    return WhisperHandler(
        model_path="mlx-community/whisper-large-v3-turbo",
        worker=worker,
    )


@pytest.mark.asyncio
async def test_initialize_loads_model(handler):
    with patch("app.handlers.whisper.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe = MagicMock(return_value=MLX_WHISPER_RESULT)
        # initialize doesn't load — model is loaded lazily on first transcribe
        # so just verify handler is in correct state
        await handler.initialize()
        assert handler.model_path == "mlx-community/whisper-large-v3-turbo"


@pytest.mark.asyncio
async def test_transcribe_returns_result(handler, tmp_wav_file):
    with patch("app.handlers.whisper.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe = MagicMock(return_value=MLX_WHISPER_RESULT)
        params = TranscriptionParams(language="en", temperature=0.0)
        result = await handler.transcribe(tmp_wav_file, params)

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
        await handler.transcribe(tmp_wav_file, params)

    call_kwargs = mock_mlx.transcribe.call_args[1]
    assert call_kwargs["language"] == "zh"
    assert call_kwargs["initial_prompt"] == "hint"
    assert call_kwargs["temperature"] == 0.2


@pytest.mark.asyncio
async def test_transcribe_stream_yields_sse_segments(handler, tmp_wav_file):
    with patch("app.handlers.whisper.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe = MagicMock(return_value=MLX_WHISPER_RESULT)
        params = TranscriptionParams()
        chunks = []
        async for chunk in handler.transcribe_stream(tmp_wav_file, params):
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
    import gc
    with patch("app.handlers.whisper.gc") as mock_gc, \
         patch("app.handlers.whisper.mx") as mock_mx:
        await handler.cleanup()
        mock_gc.collect.assert_called_once()
        mock_mx.clear_cache.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/handlers/test_whisper.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.handlers.whisper'`

- [ ] **Step 3: Implement `app/handlers/whisper.py`**

```python
# app/handlers/whisper.py
from __future__ import annotations
import gc
import json
from pathlib import Path
from typing import AsyncGenerator

import mlx.core as mx
import mlx_whisper
from loguru import logger

from app.handlers.base import BaseHandler, AudioCapable
from app.schemas.audio import (
    TranscriptionParams,
    TranscriptionResult,
    SegmentResult,
)
from app.schemas.common import ModelCard
from app.worker import InferenceWorker


class WhisperHandler(BaseHandler, AudioCapable):
    """Runs mlx_whisper.transcribe on the shared InferenceWorker thread."""

    def __init__(self, model_path: str, worker: InferenceWorker) -> None:
        self.model_path = model_path
        self._worker = worker
        # Derive a short model name for the ModelCard
        self._model_id = model_path.rstrip("/").split("/")[-1]

    async def initialize(self) -> None:
        logger.info(f"WhisperHandler ready (model will load on first request): {self.model_path}")

    async def cleanup(self) -> None:
        logger.info("WhisperHandler cleanup")
        gc.collect()
        mx.clear_cache()

    def model_info(self) -> ModelCard:
        return ModelCard(id=self._model_id)

    async def transcribe(
        self, audio_path: Path, params: TranscriptionParams
    ) -> TranscriptionResult:
        def _run() -> dict:
            return mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=self.model_path,
                language=params.language,
                initial_prompt=params.prompt,
                temperature=params.temperature,
            )

        raw = await self._worker.submit(_run)
        return _parse_result(raw)

    async def transcribe_stream(
        self, audio_path: Path, params: TranscriptionParams
    ) -> AsyncGenerator[str, None]:
        # mlx_whisper has no token-level streaming; we stream at segment granularity.
        result = await self.transcribe(audio_path, params)
        for segment in result.segments:
            yield f"data: {json.dumps({'text': segment.text})}\n\n"
        yield "data: [DONE]\n\n"


def _parse_result(raw: dict) -> TranscriptionResult:
    segments = [
        SegmentResult(
            id=seg["id"],
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
            no_speech_prob=seg.get("no_speech_prob", 0.0),
        )
        for seg in raw.get("segments", [])
    ]
    duration = segments[-1].end if segments else 0.0
    return TranscriptionResult(
        text=raw.get("text", ""),
        language=raw.get("language", ""),
        duration=duration,
        segments=segments,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/handlers/test_whisper.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add app/handlers/whisper.py tests/handlers/test_whisper.py
git commit -m "feat: add WhisperHandler wrapping mlx_whisper.transcribe"
```

---

## Task 9: Response Formatters

**Files:**
- Create: `app/formatters.py`
- Create: `tests/test_formatters.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_formatters.py
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
    """Milliseconds should be correctly extracted from seconds."""
    result = TranscriptionResult(
        text="test",
        language="en",
        duration=3.5,
        segments=[SegmentResult(id=0, start=1.234, end=3.567, text="test", no_speech_prob=0.0)],
    )
    output = format_transcription(result, ResponseFormat.SRT)
    assert "00:00:01,234 --> 00:00:03,567" in output
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_formatters.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.formatters'`

- [ ] **Step 3: Implement `app/formatters.py`**

```python
# app/formatters.py
from __future__ import annotations
import json
from app.schemas.audio import ResponseFormat, TranscriptionResult, SegmentResult


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_formatters.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add app/formatters.py tests/test_formatters.py
git commit -m "feat: add response formatters for json/text/verbose_json/srt/vtt"
```

---

## Task 10: API Routers — Models and Queue

**Files:**
- Create: `app/api/models.py`
- Create: `app/api/queue.py`
- Create: `tests/api/test_models.py`
- Create: `tests/api/test_queue.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/api/test_models.py
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from app.api.models import create_models_router
from app.schemas.common import ModelCard


def _make_client() -> TestClient:
    registry = MagicMock()
    registry.list_models.return_value = [ModelCard(id="whisper-large-v3-turbo")]
    app = FastAPI()
    app.include_router(create_models_router(registry))
    return TestClient(app)


def test_list_models_returns_openai_format():
    client = _make_client()
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "whisper-large-v3-turbo"
    assert data["data"][0]["object"] == "model"


def test_list_models_empty_registry():
    registry = MagicMock()
    registry.list_models.return_value = []
    app = FastAPI()
    app.include_router(create_models_router(registry))
    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"] == []
```

```python
# tests/api/test_queue.py
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, PropertyMock
from app.api.queue import create_queue_router


def _make_client(queue_size: int = 2, active: bool = True, max_size: int = 10) -> TestClient:
    worker = MagicMock()
    type(worker).queue_size = PropertyMock(return_value=queue_size)
    type(worker).active = PropertyMock(return_value=active)
    type(worker).max_size = PropertyMock(return_value=max_size)
    app = FastAPI()
    app.include_router(create_queue_router(worker))
    return TestClient(app)


def test_queue_stats_returns_correct_fields():
    client = _make_client(queue_size=2, active=True, max_size=10)
    resp = client.get("/v1/queue/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["queue_size"] == 2
    assert data["queue_max_size"] == 10
    assert data["active"] is True


def test_queue_stats_idle():
    client = _make_client(queue_size=0, active=False, max_size=10)
    resp = client.get("/v1/queue/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["queue_size"] == 0
    assert data["active"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/api/test_models.py tests/api/test_queue.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.api.models'`

- [ ] **Step 3: Implement `app/api/models.py`**

```python
# app/api/models.py
from __future__ import annotations
from fastapi import APIRouter
from app.registry import ModelRegistry


def create_models_router(registry: ModelRegistry) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    async def list_models():
        models = registry.list_models()
        return {
            "object": "list",
            "data": [m.model_dump() for m in models],
        }

    return router
```

- [ ] **Step 4: Implement `app/api/queue.py`**

```python
# app/api/queue.py
from __future__ import annotations
from fastapi import APIRouter
from app.worker import InferenceWorker
from app.schemas.common import QueueStats


def create_queue_router(worker: InferenceWorker) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/queue/stats", response_model=QueueStats)
    async def queue_stats():
        return QueueStats(
            queue_size=worker.queue_size,
            queue_max_size=worker.max_size,
            active=worker.active,
        )

    return router
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/api/test_models.py tests/api/test_queue.py -v
```

Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add app/api/models.py app/api/queue.py tests/api/test_models.py tests/api/test_queue.py
git commit -m "feat: add /v1/models and /v1/queue/stats endpoints"
```

---

## Task 11: API Router — Audio Transcription

**Files:**
- Create: `app/api/audio.py`
- Create: `tests/api/test_audio.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/api/test_audio.py
import json
import pytest
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch, AsyncMock
from app.api.audio import create_audio_router
from app.registry import ModelRegistry
from app.worker import InferenceWorker
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/api/test_audio.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.api.audio'`

- [ ] **Step 3: Implement `app/api/audio.py`**

```python
# app/api/audio.py
from __future__ import annotations
import json
from fastapi import APIRouter, Depends, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from loguru import logger

from app.audio import UnsupportedAudioFormatError, cleanup_temp_file, save_upload_file
from app.formatters import format_transcription
from app.handlers.base import AudioCapable
from app.registry import ModelNotFoundError, ModelRegistry
from app.schemas.audio import ResponseFormat, TranscriptionParams
from app.worker import InferenceWorker


async def _get_api_key(authorization: str | None = Header(None)) -> None:
    """Auth extension point. Replace this function to enable Bearer Token validation."""
    pass


def create_audio_router(registry: ModelRegistry, worker: InferenceWorker) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/audio/transcriptions")
    async def transcribe(
        file: UploadFile,
        model: str = Form(...),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        stream: bool = Form(False),
        _auth: None = Depends(_get_api_key),
    ):
        # Validate response_format
        try:
            fmt = ResponseFormat(response_format)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"Invalid response_format '{response_format}'.",
                        "type": "invalid_request_error",
                        "code": "invalid_response_format",
                    }
                },
            )

        # Resolve handler
        try:
            handler = registry.get(model)
        except ModelNotFoundError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"Model '{model}' not found.",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
            )

        if not isinstance(handler, AudioCapable):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"Model '{model}' does not support audio transcription.",
                        "type": "invalid_request_error",
                        "code": "capability_not_supported",
                    }
                },
            )

        # Save upload
        try:
            audio_path = await save_upload_file(file)
        except UnsupportedAudioFormatError as e:
            raise HTTPException(
                status_code=415,
                detail={
                    "error": {
                        "message": str(e),
                        "type": "invalid_request_error",
                        "code": "unsupported_audio_format",
                    }
                },
            )

        params = TranscriptionParams(
            language=language,
            prompt=prompt,
            temperature=temperature,
        )

        if stream:
            async def sse_generator():
                try:
                    async for chunk in handler.transcribe_stream(audio_path, params):
                        yield chunk
                finally:
                    cleanup_temp_file(audio_path)

            return StreamingResponse(sse_generator(), media_type="text/event-stream")

        # Non-streaming
        try:
            result = await handler.transcribe(audio_path, params)
            formatted = format_transcription(result, fmt)
        finally:
            cleanup_temp_file(audio_path)

        if fmt == ResponseFormat.TEXT:
            return PlainTextResponse(formatted)
        if fmt in (ResponseFormat.SRT, ResponseFormat.VTT):
            return PlainTextResponse(formatted, media_type="text/plain")
        return JSONResponse(content=json.loads(formatted))

    return router
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/api/test_audio.py -v
```

Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add app/api/audio.py tests/api/test_audio.py
git commit -m "feat: add /v1/audio/transcriptions endpoint with streaming and all response formats"
```

---

## Task 12: Server — App Factory, Lifespan, Metal Cleanup

**Files:**
- Create: `app/server.py`
- Create: `tests/api/test_server.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/api/test_server.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from app.config import ServerConfig
from app.registry import ModelRegistry
from app.worker import InferenceWorker
from app.server import create_app


def _make_test_app():
    config = ServerConfig(model_path="mlx-community/whisper-large-v3-turbo", memory_cleanup_interval=2)
    registry = ModelRegistry()
    worker = InferenceWorker(max_size=5, timeout=10.0)
    return create_app(config, registry, worker), worker


def test_health_endpoint():
    app, worker = _make_test_app()
    try:
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model" in data
    finally:
        worker.stop()


def test_unknown_route_returns_404():
    app, worker = _make_test_app()
    try:
        client = TestClient(app)
        resp = client.get("/nonexistent")
        assert resp.status_code == 404
    finally:
        worker.stop()


def test_metal_cleanup_triggered_at_interval():
    """Memory cleanup middleware clears Metal cache every N requests."""
    with patch("app.server.mx") as mock_mx, patch("app.server.gc") as mock_gc:
        config = ServerConfig(memory_cleanup_interval=2)
        registry = ModelRegistry()
        worker = InferenceWorker(max_size=5, timeout=10.0)
        app = create_app(config, registry, worker)
        try:
            client = TestClient(app)
            # 2 requests should trigger one cleanup (interval=2)
            client.get("/health")
            client.get("/health")
            assert mock_mx.clear_cache.call_count >= 1
        finally:
            worker.stop()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/api/test_server.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.server'`

- [ ] **Step 3: Implement `app/server.py`**

```python
# app/server.py
from __future__ import annotations
import gc
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import mlx.core as mx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.audio import create_audio_router
from app.api.models import create_models_router
from app.api.queue import create_queue_router
from app.audio import UnsupportedAudioFormatError
from app.config import ServerConfig
from app.handlers.whisper import WhisperHandler
from app.registry import ModelNotFoundError, ModelRegistry
from app.worker import InferenceWorker, QueueFullError, QueueTimeoutError


def create_app(
    config: ServerConfig,
    registry: ModelRegistry,
    worker: InferenceWorker,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Startup
        logger.info("Server starting up")
        mx.clear_cache()
        gc.collect()

        handler = WhisperHandler(model_path=config.model_path, worker=worker)
        await handler.initialize()
        registry.register(handler.model_info().id, handler)
        logger.info(f"Registered model: {handler.model_info().id}")

        yield

        # Shutdown
        logger.info("Server shutting down")
        await handler.cleanup()
        worker.stop()
        mx.clear_cache()
        gc.collect()

    app = FastAPI(title="mlx-whisper-server", version="0.1.0", lifespan=lifespan)

    # Routers
    app.include_router(create_models_router(registry))
    app.include_router(create_audio_router(registry, worker))
    app.include_router(create_queue_router(worker))

    # Health
    @app.get("/health")
    async def health():
        models = registry.list_models()
        model_id = models[0].id if models else config.model_path
        return {"status": "ok", "model": model_id}

    # Exception handlers
    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(exc), "type": "invalid_request_error", "code": "model_not_found"}},
        )

    @app.exception_handler(QueueFullError)
    async def queue_full_handler(request: Request, exc: QueueFullError):
        return JSONResponse(
            status_code=503,
            content={"error": {"message": str(exc), "type": "api_error", "code": "queue_full"}},
        )

    @app.exception_handler(QueueTimeoutError)
    async def queue_timeout_handler(request: Request, exc: QueueTimeoutError):
        return JSONResponse(
            status_code=503,
            content={"error": {"message": str(exc), "type": "api_error", "code": "queue_timeout"}},
        )

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error.", "type": "api_error", "code": "internal_error"}},
        )

    # Memory cleanup middleware
    request_count = 0

    @app.middleware("http")
    async def memory_cleanup_middleware(request: Request, call_next):
        nonlocal request_count
        response = await call_next(request)
        request_count += 1
        if request_count % config.memory_cleanup_interval == 0:
            logger.debug(f"Metal cache cleanup after {request_count} requests")
            mx.clear_cache()
            gc.collect()
        return response

    return app


def run(config: ServerConfig) -> None:
    import uvicorn

    registry = ModelRegistry()
    worker = InferenceWorker(max_size=config.queue_max_size, timeout=config.queue_timeout)
    app = create_app(config, registry, worker)
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/api/test_server.py -v
```

Expected: 3 passed

- [ ] **Step 5: Run full test suite**

```bash
pytest -v
```

Expected: All tests pass (no failures)

- [ ] **Step 6: Commit**

```bash
git add app/server.py tests/api/test_server.py
git commit -m "feat: add server app factory with lifespan, Metal cleanup middleware, and exception handlers"
```

---

## Task 13: CLI Entry Point

**Files:**
- Create: `main.py`

- [ ] **Step 1: Implement `main.py`**

```python
# main.py
import click
from app.config import ServerConfig
from app.server import run


@click.command()
@click.option("--host", default=None, help="Bind host (default: 0.0.0.0)")
@click.option("--port", default=None, type=int, help="Bind port (default: 8000)")
@click.option("--model-path", default=None, help="HuggingFace repo or local path for the Whisper model")
@click.option("--quantize", default=None, type=click.Choice(["4", "8"]), help="Quantization bits (use a pre-quantized model path)")
@click.option("--queue-max-size", default=None, type=int, help="Max concurrent+waiting requests (default: 10)")
@click.option("--queue-timeout", default=None, type=float, help="Queue wait timeout in seconds (default: 300)")
@click.option("--memory-cleanup-interval", default=None, type=int, help="Clear Metal cache every N requests (default: 20)")
@click.option("--log-level", default=None, type=click.Choice(["debug", "info", "warning", "error"]))
def cli(host, port, model_path, quantize, queue_max_size, queue_timeout, memory_cleanup_interval, log_level):
    """mlx-whisper-server: OpenAI-compatible Whisper API on Apple Silicon."""
    config = ServerConfig.from_env()

    # CLI args override env vars
    if host is not None:
        config.host = host
    if port is not None:
        config.port = port
    if model_path is not None:
        config.model_path = model_path
    if quantize is not None:
        config.quantize = int(quantize)
    if queue_max_size is not None:
        config.queue_max_size = queue_max_size
    if queue_timeout is not None:
        config.queue_timeout = queue_timeout
    if memory_cleanup_interval is not None:
        config.memory_cleanup_interval = memory_cleanup_interval
    if log_level is not None:
        config.log_level = log_level

    run(config)


if __name__ == "__main__":
    cli()
```

- [ ] **Step 2: Verify CLI help works**

```bash
python main.py --help
```

Expected output:
```
Usage: main.py [OPTIONS]

  mlx-whisper-server: OpenAI-compatible Whisper API on Apple Silicon.

Options:
  --host TEXT
  --port INTEGER
  --model-path TEXT
  ...
```

- [ ] **Step 3: Run full test suite one final time**

```bash
pytest -v
```

Expected: All tests pass

- [ ] **Step 4: Final commit**

```bash
git add main.py
git commit -m "feat: add CLI entry point with click"
```

---

## Verification Checklist

After all tasks complete, verify end-to-end manually:

```bash
# Start server (will download model on first run)
python main.py --port 8000

# In another terminal:

# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Queue stats
curl http://localhost:8000/v1/queue/stats

# Transcription (replace audio.mp3 with a real file)
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-large-v3-turbo" \
  -F "response_format=json"

# Verbose JSON
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-large-v3-turbo" \
  -F "response_format=verbose_json"

# SRT subtitles
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-large-v3-turbo" \
  -F "response_format=srt"

# Streaming
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-large-v3-turbo" \
  -F "stream=true"
```
