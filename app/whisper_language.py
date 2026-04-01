from __future__ import annotations

from mlx_whisper.tokenizer import LANGUAGES


def normalize_whisper_language(raw: str | None) -> str | None:
    """Map common locale / BCP 47 forms to Whisper language codes (ISO 639-1 style).

    Returns None if *raw* is None or blank (caller should treat as auto-detect).
    """
    if raw is None:
        return None
    s = raw.strip().replace("_", "-").lower()
    if not s:
        return None
    if s == "zh" or s.startswith("zh-"):
        return "zh"
    if "-" in s:
        return s.split("-", 1)[0]
    return s


def resolve_whisper_language(raw: str | None) -> tuple[str | None, str | None]:
    """Validate and normalize *raw* for mlx_whisper.

    Returns (code, None) where *code* is None for auto-detect, or a key in
    :data:`mlx_whisper.tokenizer.LANGUAGES`.

    On failure returns (None, message) with a short client-facing *message*.
    """
    normalized = normalize_whisper_language(raw)
    if normalized is None:
        return (None, None)
    if normalized not in LANGUAGES:
        return (
            None,
            f"Unsupported language '{raw}'. Use an ISO 639-1 code supported by Whisper (e.g. en, zh).",
        )
    return (normalized, None)
