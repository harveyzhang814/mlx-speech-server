import pytest
from pathlib import Path
from unittest.mock import AsyncMock
from app.audio import (
    save_upload_file,
    cleanup_temp_file,
    SUPPORTED_FORMATS,
    UnsupportedAudioFormatError,
)


def test_supported_formats_includes_common_types():
    assert ".wav" in SUPPORTED_FORMATS
    assert ".mp3" in SUPPORTED_FORMATS
    assert ".m4a" in SUPPORTED_FORMATS
    assert ".flac" in SUPPORTED_FORMATS
    assert ".ogg" in SUPPORTED_FORMATS
    assert ".aac" in SUPPORTED_FORMATS
    assert ".webm" in SUPPORTED_FORMATS


@pytest.mark.asyncio
async def test_save_upload_file_returns_temp_path(tmp_wav_file):
    mock_upload = AsyncMock()
    mock_upload.filename = "audio.wav"
    mock_upload.read = AsyncMock(return_value=tmp_wav_file.read_bytes())

    result = await save_upload_file(mock_upload)
    try:
        assert result.exists()
        assert result.suffix == ".wav"
        assert result.read_bytes() == tmp_wav_file.read_bytes()
    finally:
        result.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_save_upload_file_raises_for_unsupported_format():
    mock_upload = AsyncMock()
    mock_upload.filename = "audio.xyz"
    mock_upload.read = AsyncMock(return_value=b"fake data")

    with pytest.raises(UnsupportedAudioFormatError):
        await save_upload_file(mock_upload)


@pytest.mark.asyncio
async def test_save_upload_file_raises_for_no_extension():
    mock_upload = AsyncMock()
    mock_upload.filename = "audiofile"
    mock_upload.read = AsyncMock(return_value=b"fake data")

    with pytest.raises(UnsupportedAudioFormatError):
        await save_upload_file(mock_upload)


def test_cleanup_temp_file_removes_file(tmp_path):
    f = tmp_path / "test.wav"
    f.write_bytes(b"data")
    assert f.exists()
    cleanup_temp_file(f)
    assert not f.exists()


def test_cleanup_temp_file_ignores_missing_file(tmp_path):
    f = tmp_path / "nonexistent.wav"
    cleanup_temp_file(f)
