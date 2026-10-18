"""Preparation state and the mlx-whisper 0.4 ModelHolder compatibility boundary."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal

import mlx.core as mx


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

    def load(self, model_path: str) -> Any:
        model = self._holder().get_model(model_path, mx.float16)
        mx.eval(model.parameters())
        if self.current_model(model_path) is not model:
            raise RuntimeError("Loaded model no longer matches the configured model holder")
        return model

    def unload(self) -> bool:
        holder = self._holder()
        had_model = holder.model is not None
        holder.model = None
        holder.model_path = None
        return had_model
