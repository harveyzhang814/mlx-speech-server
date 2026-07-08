"""
E2E test: idle model unload after 1 minute of inactivity.

Run explicitly (takes ~72 seconds):
    pytest tests/e2e/test_idle_unload.py -v -s

Skipped in normal `pytest` runs (marked e2e).
"""
import asyncio
import sys
import time
from unittest.mock import patch

import pytest

from app.handlers.whisper import WhisperHandler
from app.worker import InferenceWorker

pytestmark = pytest.mark.e2e

_TIMEOUT_SECONDS = 60
_CHECK_INTERVAL_SECONDS = 10


@pytest.fixture
def worker():
    w = InferenceWorker(max_size=5, timeout=30.0)
    yield w
    w.stop()


@pytest.mark.asyncio
async def test_model_unloads_after_one_minute_idle(worker):
    """
    Starts a WhisperHandler with a simulated loaded model and no incoming requests.
    Expects the idle watcher to unload the model after ~1 minute.

    Total runtime: ~72 seconds.
    Watch for the log line: "Model unloaded after 1min idle"
    """
    import mlx_whisper.transcribe  # noqa: F401
    _t = sys.modules["mlx_whisper.transcribe"]
    _t.ModelHolder.model = object()  # simulate a loaded model

    with patch("app.handlers.whisper._IDLE_TIMEOUT", float(_TIMEOUT_SECONDS)), \
         patch("app.handlers.whisper._WATCH_INTERVAL", float(_CHECK_INTERVAL_SECONDS)):
        handler = WhisperHandler(
            model_path="mlx-community/whisper-large-v3-turbo",
            worker=worker,
        )
        await handler.initialize()

        wait = _TIMEOUT_SECONDS + _CHECK_INTERVAL_SECONDS + 2
        print(f"\n[e2e] Handler initialized. Waiting {wait}s for idle unload...")
        start = time.monotonic()

        await asyncio.sleep(wait)

        elapsed = time.monotonic() - start
        print(f"[e2e] {elapsed:.1f}s elapsed. Checking model state...")

        await handler.cleanup()

    assert _t.ModelHolder.model is None, (
        f"Expected ModelHolder.model to be None after {_TIMEOUT_SECONDS}s idle, but it still has a value"
    )
    print("[e2e] Model was unloaded as expected.")
