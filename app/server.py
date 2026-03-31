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
        logger.info("Server starting up")
        mx.clear_cache()
        gc.collect()

        handler = WhisperHandler(model_path=config.model_path, worker=worker)
        await handler.initialize()
        registry.register(handler.model_info().id, handler)
        logger.info(f"Registered model: {handler.model_info().id}")

        yield

        logger.info("Server shutting down")
        await handler.cleanup()
        worker.stop()
        mx.clear_cache()
        gc.collect()

    app = FastAPI(title="mlx-speech-server", version="0.1.0", lifespan=lifespan)

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
