# API Reference

**Base URL:** `http://localhost:47300` (default; configurable via `WHISPER_HOST` / `WHISPER_PORT`)

**OpenAI compatibility:** The transcription and models endpoints follow the OpenAI API schema. Clients written for OpenAI's `/v1/audio/transcriptions` and `/v1/models` work without modification.

---

## Authentication

All endpoints accept an optional `Authorization: Bearer <token>` header. Authentication is a no-op by default; replace `_get_api_key` in `app/api/audio.py` to enforce token validation.

---

## Endpoints

### POST /v1/audio/transcriptions

Transcribe an audio file.

**Content-Type:** `multipart/form-data`

#### Request fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | yes | — | Audio file to transcribe. Supported formats: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`, `.webm` |
| `model` | string | yes | — | Model ID as returned by `GET /v1/models`, or the exact `configured_model_path` from preparation capabilities |
| `language` | string | no | `null` | Source language. ISO 639-1 code (e.g. `en`, `zh`). BCP 47 / locale codes (e.g. `zh-CN`, `en-US`) are normalized automatically. `null` or omitted = auto-detect |
| `prompt` | string | no | `null` | Optional text to condition the transcription |
| `response_format` | string | no | `json` | Output format. One of: `json`, `text`, `verbose_json`, `srt`, `vtt` |
| `temperature` | float | no | `0.0` | Sampling temperature (0.0–1.0) |
| `stream` | bool | no | `false` | Stream output as Server-Sent Events |

#### Response headers

| Header | Description |
|--------|-------------|
| `X-Task-ID` | UUID assigned to this transcription task. Use it to cancel an in-progress job |

#### Response body — `json` format

```json
{
  "text": "Hello world."
}
```

#### Response body — `verbose_json` format

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 3.5,
  "text": "Hello world.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "Hello world.",
      "no_speech_prob": 0.01
    }
  ]
}
```

#### Response body — `text` format

Plain text, `Content-Type: text/plain`.

#### Response body — `srt` / `vtt` formats

Plain text subtitles, `Content-Type: text/plain`.

#### Streaming response (`stream=true`)

`Content-Type: text/event-stream`. Each segment is emitted as one SSE event. The stream ends with a `[DONE]` sentinel.

```
data: {"text": " Hello"}

data: {"text": " world."}

data: [DONE]
```

---

### DELETE /v1/audio/transcriptions/{task_id}

Cancel a single in-progress or queued transcription.

#### Path parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_id` | string | UUID from `X-Task-ID` response header |

#### Response — 200

```json
{
  "status": "cancelled",
  "task_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```

#### Response — 404

```json
{
  "error": {
    "message": "Task '...' not found.",
    "type": "invalid_request_error",
    "code": "task_not_found"
  }
}
```

---

### DELETE /v1/audio/transcriptions

Batch-cancel multiple transcriptions in one request.

**Content-Type:** `application/json`

#### Request body

```json
{
  "task_ids": [
    "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "7c9e6679-7425-40de-944b-e07fc1f90ae7"
  ]
}
```

#### Response — 200

```json
{
  "cancelled": ["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
  "not_found": ["7c9e6679-7425-40de-944b-e07fc1f90ae7"]
}
```

`cancelled` lists IDs that were successfully cancelled. `not_found` lists IDs that did not match any task. Both arrays are always present (may be empty).

---

### GET /v1/models

List registered models. Registration does not mean weights are loaded.

#### Response — 200

```json
{
  "object": "list",
  "data": [
    {
      "id": "whisper-large-v3-turbo",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

The `id` value is the last path component of the model path (e.g. `mlx-community/whisper-large-v3-turbo` → `whisper-large-v3-turbo`). Pass this value as `model` in transcription requests.

The audio endpoint also accepts the exact configured path advertised by capabilities.
Another repository or path with the same basename is not an alias.

---

### GET /v1/preparation/capabilities

Negotiate explicit preparation protocol version 1 and identify this server process.

#### Response — 200

```json
{
  "protocol_version": 1,
  "instance_id": "e95ab965-3142-4fc5-9e4e-fd34e2228114",
  "server_version": "0.3.0",
  "configured_model_id": "whisper-large-v3-turbo",
  "configured_model_path": "mlx-community/whisper-large-v3-turbo",
  "explicit_prepare": true
}
```

`instance_id` changes when the process restarts. `server_version` is the installed
package version, or `"unknown"` when package metadata is unavailable. The configured
model is fixed by server configuration; this API cannot select a different model
or replace shared configuration. Older services may return 404 for this endpoint.

### POST /v1/models/{model_id}/prepare

Start preparing the configured model without an audio upload. `model_id` must
equal `configured_model_id`, including when configuration uses a full repository
or local path. The full path alias accepted by audio requests is not accepted here.
Another target returns HTTP 409 with `model_mismatch`.

Returns HTTP **202** while preparation is admitted or running, and HTTP **200**
when the model is already ready. Both responses use the preparation status schema
below. A 202 response is not proof of loaded weights; the phase may initially be
`idle` while work is scheduled. Repeated requests join the current operation;
they do not duplicate a running load or reload a ready model. This endpoint does
not extend readiness indefinitely or change the server's idle-unload policy.

### GET /v1/models/{model_id}/preparation

Return a snapshot of preparation for the same canonical configured model ID.

#### Response — 200

```json
{
  "instance_id": "e95ab965-3142-4fc5-9e4e-fd34e2228114",
  "operation_id": "710d6fef-eaca-4533-acb4-4adc2aa44bbc",
  "model_id": "whisper-large-v3-turbo",
  "phase": "ready",
  "loaded": true,
  "progress": null,
  "downloaded_bytes": null,
  "error": null
}
```

| Field | Meaning |
|-------|---------|
| `instance_id` | Server process identity; correlate it with capabilities |
| `operation_id` | Preparation attempt identity, initially `null`; shared by duplicate requests for the current attempt |
| `model_id` | Canonical configured model ID |
| `phase` | `idle`, `downloading`, `loading`, `ready`, or `failed` |
| `loaded` | True after the actual cached model parameters have been loaded and evaluated; false before loading or after idle unload |
| `progress` | Nullable progress; no percentage is fabricated when a trustworthy total is unavailable |
| `downloaded_bytes` | Nullable count of newly received bytes in this acquisition, excluding existing resume offsets and retry overlap; zero does not prove new transfer activity |
| `error` | `null`, or `{ "code": "model_load_failed", "message": "..." }` |

Only `phase: "ready"` together with `loaded: true` reports a prepared model.
Remote acquisition reports `downloading` before loading; a local path can proceed
directly to `loading`. A warm cache is not evidence of network-transfer progress.
Preparation failures appear in the status body's `error`, with codes including
`model_load_failed`, `queue_full`, and `queue_timeout`; polling itself can still
return HTTP 200. An active native load can outlive the queue deadline: it retains
the worker and operation until native execution completes.

An audio request uses the same serialized worker and model holder as explicit
preparation. The first audio request can still load lazily without prior prepare.
The existing 1800-second idle threshold and 300-second watcher interval remain
in effect, so readiness can later return to `idle` / `loaded: false`.

---

### GET /v1/queue/stats

Return the current state of the inference queue.

#### Response — 200

```json
{
  "queue_size": 2,
  "queue_max_size": 10,
  "active": true,
  "active_status": "running"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `queue_size` | integer | Number of requests waiting in the queue (excludes the request currently being processed) |
| `queue_max_size` | integer | Maximum queue capacity. New requests return `503 queue_full` when this limit is reached |
| `active` | boolean | `true` while native work (transcription or preparation) owns the worker, including work still running after an HTTP/queue deadline |
| `active_status` | string | `"idle"` / `"running"` / `"cancelling"` |

---

### GET /health

Liveness check. `status: "ok"` does not prove model weights are loaded; use
preparation status for readiness.

#### Response — 200

```json
{
  "status": "ok",
  "model": "whisper-large-v3-turbo"
}
```

---

## Error format

All error responses use a consistent envelope compatible with the OpenAI error schema:

```json
{
  "error": {
    "message": "Human-readable description.",
    "type": "invalid_request_error",
    "code": "machine_readable_code"
  }
}
```

### Error codes

| HTTP status | `code` | `type` | Cause |
|-------------|--------|--------|-------|
| 400 | `invalid_response_format` | `invalid_request_error` | `response_format` is not one of the supported values |
| 400 | `model_not_found` | `invalid_request_error` | `model` field does not match any registered model |
| 400 | `capability_not_supported` | `invalid_request_error` | The model exists but does not support audio transcription |
| 400 | `unsupported_language` | `invalid_request_error` | `language` code cannot be normalized to a Whisper-supported ISO 639-1 code |
| 409 | `model_mismatch` | `invalid_request_error` | Preparation target differs from the configured canonical model ID |
| 404 | `task_not_found` | `invalid_request_error` | `task_id` does not match any active or queued task |
| 415 | `unsupported_audio_format` | `invalid_request_error` | Uploaded file extension is not in the supported set |
| 500 | `internal_error` | `api_error` | Unhandled server-side error |
| 503 | `queue_full` | `api_error` | Inference queue is at capacity |
| 503 | `queue_timeout` | `api_error` | Request waited longer than `WHISPER_QUEUE_TIMEOUT` seconds |

---

## Language codes

The `language` field accepts:

- **ISO 639-1** two-letter codes: `en`, `zh`, `ja`, `fr`, `de`, … (full list: [`mlx_whisper.tokenizer.LANGUAGES`](https://github.com/ml-explore/mlx-examples/blob/main/whisper/mlx_whisper/tokenizer.py))
- **BCP 47 / locale codes**: `en-US`, `zh-CN`, `zh-TW`, `pt-BR`, etc. — normalized to the base language code automatically
- `null` / omitted: auto-detect language

Chinese variants (`zh-CN`, `zh-TW`, `zh-Hans`, `zh-Hant`) all map to `zh`; Whisper has a single Chinese track.

---

## Supported audio formats

`.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.aac`, `.webm`

---

## Service CLI readiness contract

`mlx ensure-running --json` reports service availability separately from model
readiness:

```json
{
  "running": true,
  "pid": 12345,
  "port": 47300,
  "health": {"status": "ok", "model": "whisper-large-v3-turbo"},
  "service_label": "com.local.mlx-speech-server"
}
```

Failures add an `error` object containing `code` and `message`. The command is
idempotent for an existing healthy service and does not restart it to apply
configuration. `--model`, `--port`, and `--use-current-runtime` support first
installation; existing configuration is preserved. Neither `running` nor the
legacy launchd `loaded` status means that model weights are loaded. Existing
`start`, `stop`, and `restart` commands remain available. The CLI uses the shared
service label/config paths and exposes no independent label/config override.

For isolated real-model acceptance, see
[Verify explicit preparation](../how-to/verify-explicit-preparation.md).
