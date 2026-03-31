from __future__ import annotations
from pydantic import BaseModel


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class ErrorResponse(BaseModel):
    error: dict

    def __init__(self, message: str, type: str, code: str):
        super().__init__(error={"message": message, "type": type, "code": code})


class QueueStats(BaseModel):
    queue_size: int
    queue_max_size: int
    active: bool
