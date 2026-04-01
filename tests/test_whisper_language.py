import pytest

from app.whisper_language import normalize_whisper_language, resolve_whisper_language


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("zh", "zh"),
        ("ZH", "zh"),
        ("zh-CN", "zh"),
        ("zh_tw", "zh"),
        ("zh-Hant", "zh"),
        ("en", "en"),
        ("en-US", "en"),
        ("en_GB", "en"),
        ("pt-BR", "pt"),
    ],
)
def test_normalize_whisper_language(raw, expected):
    assert normalize_whisper_language(raw) == expected


def test_resolve_auto_detect():
    code, err = resolve_whisper_language(None)
    assert code is None and err is None
    code, err = resolve_whisper_language("  ")
    assert code is None and err is None


def test_resolve_zh_cn():
    code, err = resolve_whisper_language("zh-CN")
    assert code == "zh" and err is None


def test_resolve_invalid_language():
    code, err = resolve_whisper_language("qq")
    assert code is None
    assert err is not None
    assert "qq" in err or "Unsupported" in err
