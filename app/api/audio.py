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
