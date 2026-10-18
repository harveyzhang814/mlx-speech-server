from __future__ import annotations
from app.handlers.base import BaseHandler
from app.schemas.common import ModelCard


class ModelNotFoundError(Exception):
    pass


class ModelMismatchError(Exception):
    pass


class ModelRegistry:
    """Maps model IDs to handler instances."""

    def __init__(self) -> None:
        self._handlers: dict[str, BaseHandler] = {}

    def register(self, model_id: str, handler: BaseHandler) -> None:
        self._handlers[model_id] = handler

    def get(self, model_id: str) -> BaseHandler:
        try:
            return self._handlers[model_id]
        except KeyError:
            raise ModelNotFoundError(f"Model '{model_id}' not found.")

    def get_configured(self, model_id: str, configured_model_id: str) -> BaseHandler:
        """Preparation cannot grant access to any model except the configured ID."""
        if model_id != configured_model_id:
            raise ModelMismatchError(
                f"Model '{model_id}' does not match configured model '{configured_model_id}'."
            )
        return self.get(configured_model_id)

    def list_models(self) -> list[ModelCard]:
        return [h.model_info() for h in self._handlers.values()]
