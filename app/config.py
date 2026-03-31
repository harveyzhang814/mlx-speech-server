from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "mlx-community/whisper-large-v3-turbo"
    quantize: int | None = None
    memory_cleanup_interval: int = 20
    queue_max_size: int = 10
    queue_timeout: float = 300.0
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> ServerConfig:
        def _int(key: str, default: int) -> int:
            val = os.environ.get(key)
            return int(val) if val is not None else default

        def _float(key: str, default: float) -> float:
            val = os.environ.get(key)
            return float(val) if val is not None else default

        def _optional_int(key: str) -> int | None:
            val = os.environ.get(key)
            return int(val) if val is not None else None

        return cls(
            host=os.environ.get("WHISPER_HOST", "0.0.0.0"),
            port=_int("WHISPER_PORT", 8000),
            model_path=os.environ.get(
                "WHISPER_MODEL_PATH", "mlx-community/whisper-large-v3-turbo"
            ),
            quantize=_optional_int("WHISPER_QUANTIZE"),
            memory_cleanup_interval=_int("WHISPER_MEMORY_CLEANUP_INTERVAL", 20),
            queue_max_size=_int("WHISPER_QUEUE_MAX_SIZE", 10),
            queue_timeout=_float("WHISPER_QUEUE_TIMEOUT", 300.0),
            log_level=os.environ.get("WHISPER_LOG_LEVEL", "info"),
        )
