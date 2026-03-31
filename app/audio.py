from __future__ import annotations
import tempfile
from pathlib import Path
from fastapi import UploadFile


SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".webm"}


class UnsupportedAudioFormatError(Exception):
    pass


async def save_upload_file(upload_file: UploadFile) -> Path:
    """Save an uploaded audio file to a temp path. Caller must clean up."""
    filename = upload_file.filename or ""
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_FORMATS:
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        content = await upload_file.read()
        tmp.write(content)
        return Path(tmp.name)


def cleanup_temp_file(path: Path) -> None:
    """Delete a temp file, silently ignoring errors."""
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
