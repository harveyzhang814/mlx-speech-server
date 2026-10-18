"""Preparation state and the mlx-whisper 0.4 ModelHolder compatibility boundary."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal

import mlx.core as mx
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import DryRunError
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
            resolved_path = self._prefetch(model_path, report)
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

    @staticmethod
    def _prefetch(model_path: str, report: Callable[[str, int | None], None]) -> str:
        report("downloading", 0)

        class QuietProgress(HubProgress):
            def __init__(self, *args, **kwargs):
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)

        try:
            # Public metadata discovery selects exactly one immutable revision.
            # Snapshot aggregation conflates resume offsets with new bytes, so
            # only the following per-file transfers may report byte activity.
            files = snapshot_download(repo_id=model_path, dry_run=True, tqdm_class=QuietProgress)
        except DryRunError:
            # Match snapshot_download's original offline cache fallback. Loading
            # will still reject incomplete or damaged cached weights normally.
            return snapshot_download(repo_id=model_path, local_files_only=True)
        if not files or len({file.commit_hash for file in files}) != 1:
            raise RuntimeError("Model snapshot metadata does not identify one configured revision")
        commit = files[0].commit_hash
        received = 0
        progress_lock = Lock()

        def download(file) -> None:
            highest = 0

            class FileProgress(QuietProgress):
                def __init__(self, *args, **kwargs):
                    nonlocal highest
                    self._position = max(0, int(kwargs.get("initial") or 0))
                    self._bytes = kwargs.get("unit") == "B"
                    self._active = True
                    with progress_lock:
                        # Already present bytes are a baseline, never activity.
                        highest = max(highest, self._position)
                    super().__init__(*args, **kwargs)

                def update(self, n=1):
                    nonlocal received, highest
                    result = super().update(n)
                    if self._bytes and n is not None and n > 0:
                        with progress_lock:
                            if not self._active:
                                return result
                            self._position += int(n)
                            # Retries may recreate a bar below its prior high
                            # water mark. Do not count those bytes a second time.
                            if self._position > highest:
                                received += self._position - highest
                                highest = self._position
                                report("downloading", received)
                    return result

                def close(self):
                    with progress_lock:
                        self._active = False
                    super().close()

            hf_hub_download(repo_id=model_path, filename=file.filename, revision=commit,
                            tqdm_class=FileProgress)

        with ThreadPoolExecutor(max_workers=8) as transfers:
            list(transfers.map(download, files))
        # Resolve the exact completed revision locally; no second acquisition can
        # hide inside the subsequent model-loading phase.
        return snapshot_download(repo_id=model_path, revision=commit, local_files_only=True)

    def unload(self) -> bool:
        holder = self._holder()
        had_model = holder.model is not None
        holder.model = None
        holder.model_path = None
        return had_model
