# mlx-whisper-server 设计文档

**日期：** 2026-03-31
**状态：** 已批准
**参考项目：** https://github.com/cubist38/mlx-openai-server

---

## 1. 背景与目标

基于 MLX 框架在 Apple Silicon Mac 上运行 `whisper-large-v3-turbo` 模型，提供 OpenAI 兼容的 HTTP API。参考项目功能繁杂（LLM/VLM/Whisper/Embeddings/图像生成），本项目以 Whisper 场景为起点，设计可扩展至其他模型类型的轻量服务。

**核心目标：**
- OpenAI 兼容的 `/v1/audio/transcriptions` 接口
- 流式响应（segment 级 SSE）
- 全部 response_format：`json` / `text` / `verbose_json` / `srt` / `vtt`
- 针对 Apple Silicon 统一内存的优化
- 可扩展的 handler 抽象，方便未来接入其他模型类型

---

## 2. 目录结构

```
mlx-whisper-server/
├── main.py                    # CLI 入口（click）
├── app/
│   ├── __init__.py
│   ├── config.py              # ServerConfig dataclass，环境变量 + CLI 双通道
│   ├── worker.py              # InferenceWorker：单线程 ThreadPoolExecutor + 显式 asyncio.Queue
│   ├── registry.py            # ModelRegistry：model_id → handler 实例
│   ├── audio.py               # 音频临时文件保存、格式校验、清理
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── base.py            # BaseHandler + AudioCapable 抽象类
│   │   └── whisper.py         # WhisperHandler 实现
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── common.py          # ModelCard、ErrorResponse、QueueStats
│   │   └── audio.py           # TranscriptionRequest、TranscriptionResponse
│   ├── api/
│   │   ├── __init__.py
│   │   ├── models.py          # GET /v1/models
│   │   ├── audio.py           # POST /v1/audio/transcriptions
│   │   └── queue.py           # GET /v1/queue/stats
│   └── server.py              # create_app()、lifespan、全局异常处理器
└── pyproject.toml
```

---

## 3. 架构与数据流

### 请求链路

```
HTTP POST /v1/audio/transcriptions
        │
        ▼
   api/audio.py (FastAPI Router)
     ├─ 解析 multipart form-data
     ├─ audio.py 保存临时文件 + 格式校验
     └─ registry.get(model) → 检查 isinstance(handler, AudioCapable)
        │
        ▼
   worker.py (InferenceWorker)
     ├─ asyncio.Queue 检查：队列已满 → 立即返回 503
     ├─ 入队，等待推理线程（带超时）→ 超时 → 返回 503
     └─ loop.run_in_executor(executor, fn)  ← 单线程池，保护事件循环
        │
        ▼
   handlers/whisper.py (WhisperHandler)
     └─ mlx_whisper.transcribe(audio_path, **params)
        │
        ▼
   api/audio.py
     ├─ 普通：格式化后返回 JSONResponse / PlainTextResponse
     └─ 流式：StreamingResponse，按 segment yield SSE
        │
        ▼
   HTTP Response（+ 清理临时文件）
```

### 关键决策

- **InferenceWorker** 使用显式 `asyncio.Queue` + `ThreadPoolExecutor(max_workers=1)`：队列满时立即 503，避免无限堆积；推理是 Metal 密集型，串行处理避免统一内存争抢
- **模型在 lifespan 启动时一次性加载**，之后全程复用，不重复加载
- **流式粒度为 segment 级**：`mlx_whisper` 不支持 token 级流式，转录完成后按 segment 逐块 yield SSE，行为与参考项目一致
- **临时文件在响应结束后（含异常）统一清理**，使用 `finally` 块保证执行

---

## 4. Handler 抽象层

### BaseHandler + 能力 Mixin

```python
# handlers/base.py

class BaseHandler(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def cleanup(self) -> None: ...

    @abstractmethod
    def model_info(self) -> ModelCard: ...


class AudioCapable(ABC):
    """实现此 Mixin 表示该 handler 支持音频转录能力。"""

    @abstractmethod
    async def transcribe(
        self, audio_path: Path, params: TranscriptionParams
    ) -> TranscriptionResult: ...

    @abstractmethod
    async def transcribe_stream(
        self, audio_path: Path, params: TranscriptionParams
    ) -> AsyncGenerator[str, None]: ...
```

### 扩展模式

未来新增模型只需：
1. `handlers/<new_type>.py` — 实现 `BaseHandler` + 对应能力 Mixin
2. `api/<endpoint>.py` — 新 Router，`isinstance` 检查能力
3. `server.py` lifespan — 注册一行

**已有代码无需修改。**

### 能力 Mixin 扩展预留（不在本期实现）

```python
# 示例，供未来参考
class ChatCapable(ABC): ...       # /v1/chat/completions
class EmbeddingCapable(ABC): ...  # /v1/embeddings
```

---

## 5. API 契约

### POST /v1/audio/transcriptions

**请求（multipart/form-data）：**

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | UploadFile | ✅ | 支持 mp3/wav/m4a/ogg/flac/aac/webm |
| `model` | str | ✅ | 接受 `whisper-large-v3-turbo`，其他返回 400 |
| `language` | str | 可选 | ISO 639-1，默认 `null`（自动检测） |
| `prompt` | str | 可选 | 上下文提示词 |
| `response_format` | str | 可选 | `json`（默认）/ `text` / `verbose_json` / `srt` / `vtt` |
| `temperature` | float | 可选 | 0.0–1.0，默认 0.0 |
| `stream` | bool | 可选 | 默认 false |

**响应示例：**

`json`：
```json
{ "text": "Hello world" }
```

`verbose_json`：
```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 12.4,
  "text": "Hello world",
  "segments": [
    { "id": 0, "start": 0.0, "end": 2.1, "text": " Hello world", "no_speech_prob": 0.01 }
  ]
}
```

`text`：纯字符串

`srt` / `vtt`：标准字幕格式字符串

**流式 SSE（`stream=true`）：**
```
data: {"text": " Hello"}
data: {"text": " world"}
data: [DONE]
```

### GET /v1/models

返回已注册模型列表，OpenAI 兼容格式：
```json
{
  "object": "list",
  "data": [
    { "id": "whisper-large-v3-turbo", "object": "model", "owned_by": "local" }
  ]
}
```

### GET /v1/queue/stats

返回当前推理队列状态：
```json
{
  "queue_size": 2,
  "queue_max_size": 10,
  "active": true
}
```

| 字段 | 说明 |
|---|---|
| `queue_size` | 当前等待中的请求数 |
| `queue_max_size` | 队列容量上限（来自配置） |
| `active` | 当前是否有请求正在推理 |

### GET /health

```json
{ "status": "ok", "model": "whisper-large-v3-turbo" }
```

### 鉴权扩展点

```python
# api/audio.py
async def get_api_key(authorization: str | None = Header(None)) -> None:
    # 默认：直接 pass，不校验
    # 后续只需替换此函数实现 Bearer Token 校验
    pass
```

---

## 6. macOS / Apple Silicon 优化

### Metal 缓存清理

| 时机 | 实现位置 |
|---|---|
| 服务启动前 | `server.py` lifespan startup |
| 服务关闭后 | `server.py` lifespan shutdown |
| 每 N 个请求（可配置） | `server.py` 请求计数中间件 |

```python
import mlx.core as mx
import gc

mx.clear_cache()
gc.collect()
```

### InferenceWorker 线程隔离

MLX/Metal 推理在独立线程执行，FastAPI 事件循环始终响应：

```python
# worker.py
executor = ThreadPoolExecutor(max_workers=1)
result = await loop.run_in_executor(executor, inference_fn)
```

### 量化支持

`quantize: int | None` 配置项支持 4-bit / 8-bit 量化，降低统一内存占用（`whisper-large-v3-turbo` fp16 约 3GB）。

---

## 7. 错误处理

统一 OpenAI 兼容错误格式：

```json
{
  "error": {
    "message": "Model 'gpt-4' not found",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

| 场景 | 状态码 | type |
|---|---|---|
| 不支持的模型名 | 400 | `invalid_request_error` |
| 不支持的 response_format | 400 | `invalid_request_error` |
| 音频格式不支持 | 415 | `invalid_request_error` |
| 音频文件损坏 | 422 | `invalid_request_error` |
| handler 不支持该能力 | 400 | `invalid_request_error` |
| 推理内部异常 | 500 | `api_error` |
| 队列已满（超过 queue_max_size） | 503 | `api_error` |
| 队列等待超时（超过 queue_timeout） | 503 | `api_error` |

全局异常处理器在 `server.py` 注册，handler 只抛标准 Python 异常，不耦合 HTTP 层。

---

## 8. 配置

```python
@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "mlx-community/whisper-large-v3-turbo"
    quantize: int | None = None          # None / 4 / 8
    memory_cleanup_interval: int = 20    # 每 N 个请求清理 Metal 缓存
    queue_max_size: int = 10             # 队列容量上限，超出返回 503
    queue_timeout: float = 300.0         # 请求最长等待秒数，超出返回 503
    log_level: str = "info"
```

环境变量前缀 `WHISPER_`，例如 `WHISPER_PORT=9000`。CLI 参数优先级高于环境变量。

---

## 9. 依赖

```toml
[project]
dependencies = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.29",
    "mlx-whisper>=0.4",
    "mlx>=0.16",
    "pydantic>=2.0",
    "python-multipart",
    "click",
    "loguru",
]
```

---

## 10. 启动方式

```bash
# 默认启动
python main.py

# 指定模型路径和端口
python main.py --model-path mlx-community/whisper-large-v3-turbo --port 8000

# 4-bit 量化（适合 8GB 统一内存 Mac）
python main.py --quantize 4

# 环境变量方式
WHISPER_MODEL_PATH=mlx-community/whisper-large-v3-turbo python main.py
```

---

## 11. 不在本期范围内

- 多模型并发加载（进程隔离）
- LLM / Embeddings / 图像生成 handler 实现
- 完整鉴权系统
- 请求队列持久化（重启后恢复）
- Docker 部署配置
