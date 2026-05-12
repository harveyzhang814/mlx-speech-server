# Whisper / mlx_whisper 语言参数规范化 — 设计说明

**日期:** 2026-04-01  
**状态:** 已确认（按推荐方案：服务端规范化 + 非法语种 400）

## 背景与目标

客户端常传入 BCP 47 / locale 形式（如 `zh-CN`），`mlx_whisper` 在 `get_tokenizer` 中会使用小写后的字符串查表，导致 `ValueError: Unsupported language: zh-cn`。OpenAI 文档建议的 `language` 为 ISO 639-1 两字母码（如 `zh`），与 Whisper 支持的短码一致。

**目标：** 在 API 边界将常见输入规范为 mlx_whisper 可接受的语种码；对规范化后仍不支持的语言返回 **400** 与现有 OpenAI 风格 `error` 体，避免 **500**。

## 方案对比

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| A | 仅在 `audio.py` 内联 `if/else` 映射 | 改动最少 | 难测、难扩展、与校验耦合 |
| B | 独立模块 `normalize` + 用 `mlx_whisper.tokenizer.LANGUAGES` 校验 | 单测友好、与上游支持列表同步 | 多一个小文件 |
| C | 在 `WhisperHandler` 内规范化 | 底层始终干净 | HTTP 400 需在 handler 抛自定义异常再注册，与当前「router 直接 `JSONResponse`」风格不一致 |

**推荐：方案 B** — 与现有 `audio.py` 中 `_error_response` 模式一致，在构造 `TranscriptionParams` 之前完成规范化与校验。

## 架构与数据流

1. 请求进入 `POST /v1/audio/transcriptions`，`language` 为可选表单字段。
2. 若 `language` 为 `None`、仅空白或空字符串 → 按 **`None` 处理**（保持自动检测行为）。
3. 否则：`strip` → **locale / 别名映射**（见下）→ **小写** → 可选 **BCP 47 主语言**（例如 `pt-br` → `pt`，在 `LANGUAGES` 中存在时保留）。
4. 若最终非空字符串 **不在** `mlx_whisper.tokenizer.LANGUAGES` 的键集合中 → 返回 **400**，`code` 建议为 `unsupported_language`（或 `invalid_language`），`message` 简短说明并提示使用 ISO 639-1 / 两字母码。
5. 将规范化后的 `str | None` 写入 `TranscriptionParams`，`WhisperHandler` **无需修改**（除非后续希望防御性再校验一层，本次 YAGNI）。

## 规范化规则（首版，YAGNI）

- **显式映射（常见误用）：** `zh-cn`、`zh-tw`、`zh-hans`、`zh-hant` 等 → `zh`（Whisper 对中文仅 `zh`）。
- **通用：** 全串 `lower()` 后，若含 `-`，取 **第一段** 作为候选语种码（`pt-br` → `pt`），再查 `LANGUAGES`。
- **顺序建议：** 先应用显式映射表，再拆主语言子标签，避免 `zh-cn` 被拆成 `zh` 已覆盖；显式表可覆盖 `zh-*` → `zh`。

## 错误处理

- **不**引入未捕获的 `ValueError` 传到全局 `generic_handler`。
- 与 `invalid_response_format`、`model_not_found` 一致：在 `audio.py` 内 **`return _error_response(...)`**，不新增全局 `exception_handler`（除非团队更希望用自定义异常类统一音频域错误，本次不必要）。

## 测试

- **单元测试：** 新模块的映射与边界（`None`、空串、`zh-CN` → `zh`、`en-US` → `en`、未知 `xx` → 校验失败）。
- **校验：** 使用运行时 `LANGUAGES`（与 mlx_whisper 版本一致）；可对「规范化后为合法码」做断言。
- **API 测试：** 在现有 `FakeAudioHandler` / mock 路径上增加：带 `language=zh-CN` 时若 handler 被调用，传入的 params.language 为 `zh`；非法语言返回 400 且 body 含 `error.code`。

## 文档

- `README.md` / `README_zh.md`：`language` 行补充「支持 ISO 639-1；常见 locale（如 `zh-CN`）会被规范为 Whisper 语种码」。

## 非目标（本期不做）

- 不维护与 OpenAI 云端完全一致的报错文案。
- 不尝试区分繁体地区专用模型（Whisper 单 `zh`）。
