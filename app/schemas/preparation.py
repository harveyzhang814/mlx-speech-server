"""Version 1 of the explicit model preparation protocol."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PreparationCapabilities(BaseModel):
    protocol_version: Literal[1] = 1
    instance_id: str
    server_version: str
    configured_model_id: str
    configured_model_path: str
    explicit_prepare: Literal[True] = True


class PreparationErrorResponse(BaseModel):
    code: str
    message: str


class PreparationStatus(BaseModel):
    instance_id: str
    operation_id: str | None
    model_id: str
    phase: Literal["idle", "downloading", "loading", "ready", "failed"]
    loaded: bool
    progress: float | None
    downloaded_bytes: int | None = Field(default=None, ge=0)
    error: PreparationErrorResponse | None
