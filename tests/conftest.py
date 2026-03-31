import pytest
from pathlib import Path
import tempfile
import wave
import struct


@pytest.fixture
def tmp_wav_file() -> Path:
    """Create a minimal valid WAV file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = Path(f.name)
    # Write a minimal 1-second 16kHz mono WAV
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        samples = struct.pack("<" + "h" * 16000, *([0] * 16000))
        wav.writeframes(samples)
    yield path
    path.unlink(missing_ok=True)
