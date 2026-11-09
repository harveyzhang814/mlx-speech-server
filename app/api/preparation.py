"""Nonblocking preparation of the server's configured model only."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from typing import cast
from uuid import uuid4

from fastapi import APIRouter, Response
from loguru import logger

from app.handlers.whisper import WhisperHandler
from app.preparation import PreparationError, PreparationSnapshot
from app.registry import ModelRegistry
from app.schemas.preparation import PreparationCapabilities, PreparationStatus
from app.worker import QueueFullError, QueueTimeoutError


def create_preparation_router(
    registry: ModelRegistry,
    configured_model_id: str,
    configured_model_path: str,
    instance_id: str,
    server_version: str,
) -> APIRouter:
    task: asyncio.Task | None = None
    background_failure: PreparationSnapshot | None = None
    background_failure_source: PreparationSnapshot | None = None

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            # Router shutdown runs before the server tears down its handler and
            # worker. Native loading cannot be cancelled: drain its owning task.
            if task is not None:
                await asyncio.shield(task)

    router = APIRouter(lifespan=lifespan)

    def handler_for(model_id: str) -> WhisperHandler:
        return cast(WhisperHandler, registry.get_configured(model_id, configured_model_id))

    def status_for(handler: WhisperHandler) -> PreparationStatus:
        nonlocal background_failure, background_failure_source
        snapshot = handler.preparation_status()
        # Handler snapshots are immutable. An unexpected exception's fallback
        # applies only while the handler still exposes that same failed attempt;
        # inference, idle unload, and new preparation all publish newer snapshots.
        if background_failure is not None:
            if snapshot is background_failure_source and not snapshot.loaded:
                snapshot = background_failure
            else:
                background_failure = None
                background_failure_source = None
        return PreparationStatus(instance_id=instance_id, **asdict(snapshot))

    async def prepare(handler: WhisperHandler) -> None:
        nonlocal background_failure, background_failure_source
        try:
            await handler.prepare()
        except Exception as exc:
            # Handler normally reports errors itself. Also consume unexpected
            # background exceptions and expose a terminal state to polling clients.
            code = "model_load_failed"
            if isinstance(exc, QueueFullError):
                code = "queue_full"
            elif isinstance(exc, QueueTimeoutError):
                code = "queue_timeout"
            snapshot = handler.preparation_status()
            background_failure_source = snapshot
            background_failure = replace(
                snapshot,
                operation_id=snapshot.operation_id or str(uuid4()),
                phase="failed",
                loaded=False,
                error=PreparationError(code=code, message=str(exc)),
            )
            logger.exception("Background model preparation failed")

    @router.get("/v1/preparation/capabilities", response_model=PreparationCapabilities)
    async def capabilities() -> PreparationCapabilities:
        return PreparationCapabilities(
            instance_id=instance_id,
            server_version=server_version,
            configured_model_id=configured_model_id,
            configured_model_path=configured_model_path,
        )

    @router.post(
        "/v1/models/{model_id:path}/prepare",
        response_model=PreparationStatus,
        responses={202: {"model": PreparationStatus}},
    )
    async def prepare_model(model_id: str, response: Response) -> PreparationStatus:
        nonlocal task, background_failure, background_failure_source
        handler = handler_for(model_id)
        current = status_for(handler)
        if current.loaded:
            response.status_code = 200
            return current
        if task is None or task.done():
            background_failure = None
            background_failure_source = None
            task = asyncio.create_task(prepare(handler))
        # Give the handler one turn to create its shared operation/snapshot. Do
        # not wait for worker admission, download, or weight loading to complete.
        await asyncio.sleep(0)
        response.status_code = 202
        return status_for(handler)

    @router.get("/v1/models/{model_id:path}/preparation", response_model=PreparationStatus)
    async def preparation(model_id: str) -> PreparationStatus:
        return status_for(handler_for(model_id))

    return router
