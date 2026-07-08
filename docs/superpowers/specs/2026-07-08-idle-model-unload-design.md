# Idle Model Unload Design

**Date:** 2026-07-08
**Branch:** feat/idle-model-unload

## Problem

After the first transcription request, mlx_whisper caches the Whisper model (~1.5 GB for large-v3-turbo) in `ModelHolder.model` (a module-level class variable). The model stays in unified memory indefinitely — even during long idle periods — because `mx.clear_cache()` only frees temporary Metal buffers, not model weights.

## Goal

Automatically unload the model after 30 minutes of inactivity to free unified memory, then silently reload on the next request.

## Decisions

| Question | Decision |
|----------|----------|
| Timeout duration | Fixed 1800s (30 min), module-level constant |
| Configurable? | No (tracked as future todo: `WHISPER_IDLE_UNLOAD_TIMEOUT` env var) |
| Cold-start behavior | Silent wait — request queues normally, user experiences slower first response |
| Where watcher lives | Inside `WhisperHandler` (self-contained lifecycle) |

## Architecture

Only `app/handlers/whisper.py` changes. No other files are modified.

```
WhisperHandler
├── _IDLE_TIMEOUT = 1800.0          # module-level constant
├── _last_used: float               # updated by transcribe() on each call
├── _watcher_task: asyncio.Task
│
├── initialize()   → starts _watch_idle() background task
├── cleanup()      → cancels _watcher_task
├── transcribe()   → updates _last_used before submitting to worker
├── _watch_idle()  → loop: sleep 5min → check → unload if idle
└── _unload()      → ModelHolder.model = None + mx.clear_cache() + gc.collect()
```

## Data Flow

1. `transcribe()` called → `_last_used = time.time()` → submit to `InferenceWorker`
2. Worker calls `mlx_whisper.transcribe()` → `ModelHolder.get_model()` loads model if needed
3. `_watch_idle()` wakes every 5 minutes:
   - `now - _last_used > 1800` AND `not worker.active` → call `_unload()`
   - otherwise skip
4. After unload, next `transcribe()` → `ModelHolder.get_model()` reloads from disk (~3–8s cold start)

## Edge Cases

| Scenario | Handling |
|----------|----------|
| Request arrives while watcher is checking | `worker.active` is True → watcher skips this cycle |
| No requests ever received | `_last_used` initialized to `time.time()` at `initialize()`, preventing early unload |
| New request arrives during `_unload()` | Request queues in `InferenceWorker`; `ModelHolder.get_model()` reloads on worker thread |
| `cleanup()` cancels watcher mid-sleep | `CancelledError` caught in `_watch_idle()`, exits cleanly without unloading |

## Logging

```
INFO  Model unloaded after 30min idle
DEBUG Idle watcher: {n:.1f} min since last request, skipping
```

Reload is detectable via existing mlx_whisper log output when `load_model()` runs.

## Tests

All in `tests/handlers/test_whisper.py`, using `patch("app.handlers.whisper.mlx_whisper")`:

1. `test_idle_watcher_unloads_model` — patch `_IDLE_TIMEOUT` to near-zero, assert `ModelHolder.model = None` and `mx.clear_cache` called
2. `test_idle_watcher_skips_while_active` — worker active → no unload
3. `test_last_used_updated_on_transcribe` — `_last_used` advances after `transcribe()`
4. `test_watcher_cancelled_on_cleanup` — `_watcher_task.cancelled()` is True after `cleanup()`
5. `test_initial_last_used_prevents_early_unload` — no requests received → watcher does not unload

## Files Changed

- `app/handlers/whisper.py` — add `_IDLE_TIMEOUT`, `_last_used`, `_watcher_task`, `_watch_idle()`, `_unload()`; update `initialize()`, `cleanup()`, `transcribe()`
- `tests/handlers/test_whisper.py` — 5 new test cases

## Future Work

- `WHISPER_IDLE_UNLOAD_TIMEOUT` env var to make timeout configurable (tracked in task #7)
