"""Preparation state and the mlx-whisper 0.4 ModelHolder compatibility boundary."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal

import mlx.core as mx
from huggingface_hub import snapshot_download
from huggingface_hub.utils import tqdm as HubProgress


@dataclass(frozen=True)
class PreparationError:
    code: str
    message: str


@dataclass(frozen=True)
class PreparationSnapshot:
    operation_id: str | None
    model_id: str
    phase: Literal["idle", "downloading", "loading", "ready", "failed"] = "idle"
    loaded: bool = False
    progress: float | None = None
    downloaded_bytes: int | None = None
    error: PreparationError | None = None


class WhisperModelAdapter:
    """Keep private upstream holder access here; use only under the worker lock.

    mlx-whisper 0.4 transcribe defaults fp16=True and caches by model path.
    Explicit preparation must use that same dtype and evaluate lazy parameters.
    """

    @staticmethod
    def _holder():
        return import_module("mlx_whisper.transcribe").ModelHolder

    def current_model(self, model_path: str) -> Any | None:
        holder = self._holder()
        return holder.model if holder.model_path == model_path else None

    def load(
        self, model_path: str, *, report: Callable[[str, int | None], None] = lambda *_: None
    ) -> Any:
        holder = self._holder()
        model = self.current_model(model_path)
        resolved_path = model_path
        if model is None and not Path(model_path).exists():
            report("downloading", 0)

            class DownloadProgress(HubProgress):
                # HF 1.8 aggregates real byte updates separately from file count.
                # Keep counting even when console progress bars are disabled.
                def __init__(self, *args, **kwargs):
                    self._bytes = kwargs.get("unit") == "B"
                    self._received = 0
                    self._progress_lock = Lock()
                    kwargs["disable"] = True
                    super().__init__(*args, **kwargs)

                def update(self, n=1):
                    result = super().update(n)
                    if self._bytes and n is not None and n > 0:
                        with self._progress_lock:
                            self._received += int(n)
                            report("downloading", self._received)
                    return result

            # Metadata's aggregate total grows as files are discovered: no truthful
            # percentage is available. Only real byte updates signal activity.
            resolved_path = snapshot_download(repo_id=model_path, tqdm_class=DownloadProgress)
        report("loading", None)
        if model is None:
            # Loading the resolved directory cannot re-enter an unobserved download.
            model = holder.get_model(resolved_path, mx.float16)
        mx.eval(model.parameters())
        if holder.model is not model or holder.model_path not in {model_path, resolved_path}:
            raise RuntimeError("Loaded model no longer matches the configured model holder")
        # Transcription uses the original configured key. Preserve its cache hit
        # while retaining the very same evaluated object loaded from the snapshot.
        holder.model_path = model_path
        return model

    def unload(self) -> bool:
        holder = self._holder()
        had_model = holder.model is not None
        holder.model = None
        holder.model_path = None
        return had_model
