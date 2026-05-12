from __future__ import annotations
import json
import uuid
from fastapi import APIRouter, Body, Depends, Form, Header, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel


class BatchCancelRequest(BaseModel):
    task_ids: list[str]

from app.audio import UnsupportedAudioFormatError, cleanup_temp_file, save_upload_file
from app.formatters import format_transcription
from app.handlers.base import AudioCapable
from app.registry import ModelNotFoundError, ModelRegistry
from app.schemas.audio import ResponseFormat, TranscriptionParams
from app.whisper_language import resolve_whisper_language
from app.worker import InferenceWorker, TaskNotFoundError


async def _get_api_key(authorization: str | None = Header(None)) -> None:
    """Auth extension point. Replace this function to enable Bearer Token validation."""
    pass


def _error_response(status_code: int, message: str, error_type: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        },
    )


def create_audio_router(registry: ModelRegistry, worker: InferenceWorker) -> APIRouter:
    router = APIRouter()

    @router.delete("/v1/audio/transcriptions")
    async def batch_cancel_transcriptions(body: BatchCancelRequest):
        cancelled = []
        not_found = []
        for task_id in body.task_ids:
            try:
                worker.cancel(task_id)
                cancelled.append(task_id)
            except TaskNotFoundError:
                not_found.append(task_id)
        return JSONResponse({"cancelled": cancelled, "not_found": not_found})

    @router.delete("/v1/audio/transcriptions/{task_id}")
    async def cancel_transcription(task_id: str):
        try:
            worker.cancel(task_id)
        except TaskNotFoundError:
            return _error_response(
                404,
                f"Task '{task_id}' not found.",
                "invalid_request_error",
                "task_not_found",
            )
        return JSONResponse({"status": "cancelled", "task_id": task_id})

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
            return _error_response(
                400,
                f"Invalid response_format '{response_format}'.",
                "invalid_request_error",
                "invalid_response_format",
            )

        # Resolve handler
        try:
            handler = registry.get(model)
        except ModelNotFoundError:
            return _error_response(
                400,
                f"Model '{model}' not found.",
                "invalid_request_error",
                "model_not_found",
            )

        if not isinstance(handler, AudioCapable):
            return _error_response(
                400,
                f"Model '{model}' does not support audio transcription.",
                "invalid_request_error",
                "capability_not_supported",
            )

        normalized_language, language_error = resolve_whisper_language(language)
        if language_error is not None:
            return _error_response(
                400,
                language_error,
                "invalid_request_error",
                "unsupported_language",
            )

        # Save upload
        try:
            audio_path = await save_upload_file(file)
        except UnsupportedAudioFormatError as e:
            return _error_response(
                415,
                str(e),
                "invalid_request_error",
                "unsupported_audio_format",
            )

        task_id = str(uuid.uuid4())
        params = TranscriptionParams(
            language=normalized_language,
            prompt=prompt,
            temperature=temperature,
        )

        if stream:
            async def sse_generator():
                try:
                    async for chunk in handler.transcribe_stream(audio_path, params, task_id):
                        yield chunk
                finally:
                    cleanup_temp_file(audio_path)

            return StreamingResponse(
                sse_generator(),
                media_type="text/event-stream",
                headers={"X-Task-ID": task_id},
            )

        # Non-streaming
        try:
            result = await handler.transcribe(audio_path, params, task_id)
            formatted = format_transcription(result, fmt)
        finally:
            cleanup_temp_file(audio_path)

        if fmt == ResponseFormat.TEXT:
            return PlainTextResponse(formatted, headers={"X-Task-ID": task_id})
        if fmt in (ResponseFormat.SRT, ResponseFormat.VTT):
            return PlainTextResponse(formatted, media_type="text/plain", headers={"X-Task-ID": task_id})
        return JSONResponse(content=json.loads(formatted), headers={"X-Task-ID": task_id})

    return router
