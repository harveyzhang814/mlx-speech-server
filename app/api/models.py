from __future__ import annotations
from fastapi import APIRouter
from app.registry import ModelRegistry


def create_models_router(registry: ModelRegistry) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    async def list_models():
        models = registry.list_models()
        return {
            "object": "list",
            "data": [m.model_dump() for m in models],
        }

    return router
