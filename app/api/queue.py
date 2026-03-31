from __future__ import annotations
from fastapi import APIRouter
from app.worker import InferenceWorker
from app.schemas.common import QueueStats


def create_queue_router(worker: InferenceWorker) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/queue/stats", response_model=QueueStats)
    async def queue_stats():
        return QueueStats(
            queue_size=worker.queue_size,
            queue_max_size=worker.max_size,
            active=worker.active,
        )

    return router
