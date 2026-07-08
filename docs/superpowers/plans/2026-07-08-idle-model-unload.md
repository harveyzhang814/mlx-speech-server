# Idle Model Unload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically unload the Whisper model from unified memory after 30 minutes of inactivity, then silently reload it on the next request.

**Architecture:** `WhisperHandler` gains two module-level constants (`_IDLE_TIMEOUT`, `_WATCH_INTERVAL`), a `_last_used` float updated on every `transcribe()` call, and an asyncio background task started in `initialize()` that wakes every 5 minutes to check idle time and call `_unload()` when the threshold is exceeded.

**Tech Stack:** Python asyncio, `mlx_whisper.transcribe.ModelHolder` (internal model cache), `mlx.core.mx`, `loguru`

## Global Constraints

- Only two files change: `app/handlers/whisper.py` and `tests/handlers/test_whisper.py`
- No new dependencies
- `asyncio_mode = "auto"` in pytest — no `@pytest.mark.asyncio` needed for async tests (but the existing tests already use it; keep for consistency)
- Mock target for mlx_whisper is `app.handlers.whisper.mlx_whisper`
- `_IDLE_TIMEOUT` and `_WATCH_INTERVAL` are **module-level names**, patchable via `patch("app.handlers.whisper._IDLE_TIMEOUT", ...)`

---

### Task 1: Idle watcher — tests then implementation

**Files:**
- Modify: `app/handlers/whisper.py`
- Modify: `tests/handlers/test_whisper.py`

**Interfaces:**
- New module constants: `_IDLE_TIMEOUT: float = 1800.0`, `_WATCH_INTERVAL: float = 300.0`
- New instance state: `_last_used: float`, `_watcher_task: asyncio.Task | None`
- New methods: `_watch_idle(self) -> Coroutine`, `_unload(self) -> None`
- Updated: `initialize()` sets `_last_used = time.time()` and starts watcher task; `cleanup()` cancels it; `transcribe()` updates `_last_used`

---

- [ ] **Step 1: Add failing tests**

Add these imports at the top of `tests/handlers/test_whisper.py` (after existing imports):

```python
import time
import asyncio
```

Update the existing `test_initialize_sets_model_path` to clean up the watcher task it now creates:

```python
@pytest.mark.asyncio
async def test_initialize_sets_model_path(handler):
    with patch("app.handlers.whisper.mlx_whisper"):
        await handler.initialize()
        assert handler.model_path == "mlx-community/whisper-large-v3-turbo"
        await handler.cleanup()
```

Add 5 new tests at the end of `tests/handlers/test_whisper.py`:

```python
@pytest.mark.asyncio
async def test_idle_watcher_unloads_model(worker):
    import mlx_whisper.transcribe as _mlx_t
    _mlx_t.ModelHolder.model = object()  # simulate loaded model

    with patch("app.handlers.whisper._IDLE_TIMEOUT", 0.0), \
         patch("app.handlers.whisper._WATCH_INTERVAL", 0.001), \
         patch("app.handlers.whisper.mx") as mock_mx, \
         patch("app.handlers.whisper.gc"):
        handler = WhisperHandler(model_path="mlx-community/whisper-large-v3-turbo", worker=worker)
        await handler.initialize()
        handler._last_used = 0.0  # force past idle threshold
        await asyncio.sleep(0.05)  # let watcher trigger
        await handler.cleanup()

    assert _mlx_t.ModelHolder.model is None
    mock_mx.clear_cache.assert_called()


@pytest.mark.asyncio
async def test_idle_watcher_skips_while_active(worker):
    worker._active = True  # simulate in-flight inference

    with patch("app.handlers.whisper._IDLE_TIMEOUT", 0.0), \
         patch("app.handlers.whisper._WATCH_INTERVAL", 0.001):
        handler = WhisperHandler(model_path="mlx-community/whisper-large-v3-turbo", worker=worker)
        with patch.object(handler, "_unload") as mock_unload:
            await handler.initialize()
            handler._last_used = 0.0
            await asyncio.sleep(0.05)
            await handler.cleanup()

    mock_unload.assert_not_called()


@pytest.mark.asyncio
async def test_last_used_updated_on_transcribe(handler, tmp_wav_file):
    before = time.time()
    with patch("app.handlers.whisper.mlx_whisper") as mock_mlx:
        mock_mlx.transcribe = MagicMock(return_value=MLX_WHISPER_RESULT)
        params = TranscriptionParams(language="en", temperature=0.0)
        await handler.transcribe(tmp_wav_file, params, task_id="test-id")

    assert handler._last_used >= before


@pytest.mark.asyncio
async def test_watcher_cancelled_on_cleanup(worker):
    handler = WhisperHandler(model_path="mlx-community/whisper-large-v3-turbo", worker=worker)
    with patch("app.handlers.whisper._WATCH_INTERVAL", 3600.0):
        await handler.initialize()
        task = handler._watcher_task
        assert not task.done()
        await handler.cleanup()
    assert task.cancelled()


@pytest.mark.asyncio
async def test_initial_last_used_prevents_early_unload(worker):
    with patch("app.handlers.whisper._IDLE_TIMEOUT", 1800.0), \
         patch("app.handlers.whisper._WATCH_INTERVAL", 0.001):
        handler = WhisperHandler(model_path="mlx-community/whisper-large-v3-turbo", worker=worker)
        with patch.object(handler, "_unload") as mock_unload:
            await handler.initialize()
            # _last_used = time.time() set in initialize() — well under 1800s
            await asyncio.sleep(0.05)
            await handler.cleanup()

    mock_unload.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/handlers/test_whisper.py -v
```

Expected: new tests fail with `AttributeError` (`_last_used`, `_watcher_task` not found) and `test_initialize_sets_model_path` may generate a task-not-cleaned-up warning.

- [ ] **Step 3: Implement changes in `app/handlers/whisper.py`**

Replace the entire file with:

```python
from __future__ import annotations
import asyncio
import gc
import json
import time
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

_IDLE_TIMEOUT = 1800.0   # 30 minutes
_WATCH_INTERVAL = 300.0  # check every 5 minutes


class WhisperHandler(BaseHandler, AudioCapable):
    """Runs mlx_whisper.transcribe on the shared InferenceWorker thread."""

    def __init__(self, model_path: str, worker: InferenceWorker) -> None:
        self.model_path = model_path
        self._worker = worker
        self._model_id = model_path.rstrip("/").split("/")[-1]
        self._last_used: float = 0.0
        self._watcher_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        self._last_used = time.time()
        self._watcher_task = asyncio.create_task(self._watch_idle())
        logger.info(f"WhisperHandler ready (model will load on first request): {self.model_path}")

    async def cleanup(self) -> None:
        if self._watcher_task is not None:
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except asyncio.CancelledError:
                pass
        logger.info("WhisperHandler cleanup")
        gc.collect()
        mx.clear_cache()

    def model_info(self) -> ModelCard:
        return ModelCard(id=self._model_id)

    async def transcribe(
        self, audio_path: Path, params: TranscriptionParams, task_id: str
    ) -> TranscriptionResult:
        self._last_used = time.time()
        # temperature=0.0 (single float) disables Whisper's fallback schedule and
        # prevents repetition/confidence recovery. Use the full schedule when the
        # client requests 0 so that mlx_whisper can retry with higher temperatures
        # when it detects repetition loops or low-confidence segments.
        temperature: float | tuple[float, ...] = (
            (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if params.temperature == 0.0 else params.temperature
        )

        def _run() -> dict:
            return mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=self.model_path,
                language=params.language,
                initial_prompt=params.prompt,
                temperature=temperature,
                condition_on_previous_text=False,
            )

        raw = await self._worker.submit(task_id, _run)
        return _parse_result(raw)

    async def transcribe_stream(
        self, audio_path: Path, params: TranscriptionParams, task_id: str
    ) -> AsyncGenerator[str, None]:
        result = await self.transcribe(audio_path, params, task_id)
        for segment in result.segments:
            yield f"data: {json.dumps({'text': segment.text})}\n\n"
        yield "data: [DONE]\n\n"

    async def _watch_idle(self) -> None:
        try:
            while True:
                await asyncio.sleep(_WATCH_INTERVAL)
                idle = time.time() - self._last_used
                if idle > _IDLE_TIMEOUT and not self._worker.active:
                    self._unload()
                else:
                    logger.debug(f"Idle watcher: {idle / 60:.1f} min since last request, skipping")
        except asyncio.CancelledError:
            pass

    def _unload(self) -> None:
        import mlx_whisper.transcribe as _t
        _t.ModelHolder.model = None
        mx.clear_cache()
        gc.collect()
        logger.info("Model unloaded after 30min idle")


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

- [ ] **Step 4: Run all tests**

```bash
pytest tests/handlers/test_whisper.py -v
```

Expected: all tests pass including the 5 new ones and the updated `test_initialize_sets_model_path`.

Then run the full suite to check for regressions:

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 5: Lint**

```bash
ruff check app/handlers/whisper.py tests/handlers/test_whisper.py
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add app/handlers/whisper.py tests/handlers/test_whisper.py
git commit -m "feat(handlers): add idle model unload after 30min inactivity"
```
