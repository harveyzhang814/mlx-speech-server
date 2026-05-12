# Whisper 语言参数规范化 — 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 API 层将 `language` 规范为 mlx_whisper 支持的语种码，并对不支持的语言返回 400（OpenAI 风格 error），避免 `Unsupported language` 导致 500。

**Architecture:** 新增小模块实现 `normalize_whisper_language` 与基于 `mlx_whisper.tokenizer.LANGUAGES` 的校验；在 `app/api/audio.py` 构造 `TranscriptionParams` 前调用。`WhisperHandler` 保持不变。

**Tech Stack:** Python 3.x, FastAPI, mlx_whisper（`LANGUAGES` 字典）, pytest, ruff。

---

### Task 1: 语言规范化模块

**Files:**
- Create: `app/whisper_language.py`
- Test: `tests/test_whisper_language.py`

**Step 1:** 实现 `normalize_whisper_language(raw: str | None) -> str | None`：处理 `None`、空白、`strip`、小写、中文 locale 映射、`xx-YY` 取主标签。

**Step 2:** 实现 `validate_whisper_language(code: str | None) -> str | None`：若 `code` 为 `None` 则返回 `None`；若非空且不在 `LANGUAGES` 中则抛 `ValueError` 或返回可区分结果 — **推荐** 返回 `tuple[str|None, str|None]` `(ok, err_msg)` 或专用小异常类供 router 转 400，避免与 `ValueError` 混用。设计文档倾向在 router 内 `return _error_response`，故函数可返回 `tuple[str | None, str | None]` 为 `(normalized, None)` 或 `(None, "message")`。

**Step 3:** 单元测试：`zh-CN` → `zh`，`en-gb` → `en`，空 → `None`，杜撰 `qq` → 错误。

**Step 4:** `pytest tests/test_whisper_language.py -v`；`ruff check app/whisper_language.py`。

---

### Task 2: 接入 audio 路由

**Files:**
- Modify: `app/api/audio.py`（构造 `TranscriptionParams` 之前调用规范化与校验）
- Test: `tests/api/test_audio.py`

**Step 1:** 在 `params = TranscriptionParams(...)` 之前调用规范化；失败时 `return _error_response(400, ..., "invalid_request_error", "unsupported_language")`（`code` 与 `message` 与设计一致）。

**Step 2:** 流式与非流式路径共用同一 `params` 构造块，避免漏接。

**Step 3:** API 测试：非法 `language` 返回 400 与 `error` 结构；合法 `zh-CN` 在 mock 下成功（若现有测试已 mock handler，断言传入的 language）。

**Step 4:** `pytest tests/api/test_audio.py -v` 与全量 `pytest -v`。

---

### Task 3: 文档

**Files:**
- Modify: `README.md`, `README_zh.md`（`language` 表格行）

**Step 1:** 各增加一句关于 ISO 639-1 与 locale 自动规范化的说明。

---

### Task 4: 收尾

**Step 1:** `ruff check .`

**Step 2:** `pytest -v`

**Step 3:** 提交： `feat: normalize transcription language for mlx_whisper compatibility`（或 `fix:` 视团队约定）。
