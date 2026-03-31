# mlx-speech-server

[English](README.md)

基于 [MLX](https://github.com/ml-explore/mlx) 在 Apple Silicon 上原生运行的 OpenAI 兼容 Whisper 语音转录 API 服务。

## 功能特性

- **OpenAI API 兼容** — 可直接替换 `POST /v1/audio/transcriptions`
- **全格式输出** — 支持 `json`、`text`、`verbose_json`、`srt`、`vtt`
- **流式输出** — 通过 `stream=true` 启用 SSE 推送（注意：`mlx_whisper` 不支持 token 级流式，实际行为是完整推理完成后按段落依次推送）
- **Apple Silicon 优化** — Metal GPU 加速，定期调用 `mx.clear_cache()` 管理统一内存
- **请求队列** — 可配置队列大小和超时，提供 `GET /v1/queue/stats` 监控接口
- **可扩展架构** — Handler 抽象层支持未来接入其他模型类型（LLM、Embeddings 等）

## 系统要求

- macOS，搭载 Apple Silicon（M1/M2/M3/M4）
- Python 3.11+

## 安装

```bash
git clone https://github.com/harveyzhang814/mlx-speech-server.git
cd mlx-speech-server
```

## 快速开始

### 一键部署（推荐）

使用内置服务管理脚本，自动创建虚拟环境、安装依赖、注册为系统服务（登录自启、崩溃自动重启）：

```bash
./scripts/service.sh install
./scripts/service.sh start
./scripts/service.sh status
```

服务管理命令：

```bash
./scripts/service.sh install    # 安装（自动创建虚拟环境）
./scripts/service.sh uninstall  # 卸载
./scripts/service.sh upgrade    # 更新依赖
./scripts/service.sh start      # 启动
./scripts/service.sh stop       # 停止
./scripts/service.sh restart    # 重启
./scripts/service.sh status     # 查看状态（含健康检查）
./scripts/service.sh logs       # 查看日志
```

虚拟环境默认路径：`~/.local/venvs/mlx-speech-server/`
日志默认路径：`~/.local/logs/mlx-speech-server/`

### 手动启动

```bash
python3 -m venv ~/.local/venvs/mlx-speech-server
source ~/.local/venvs/mlx-speech-server/bin/activate
pip install -e "."

# 默认启动（whisper-large-v3-turbo，端口 8000）
python main.py

# 自定义端口和模型
python main.py --port 9000 --model-path mlx-community/whisper-large-v3-turbo

# 使用量化模型（降低内存占用）
python main.py --model-path mlx-community/whisper-large-v3-turbo-q4
```

模型在首次请求时自动从 HuggingFace 下载。

## 支持的模型

所有 [mlx-community](https://huggingface.co/mlx-community) 下的 Whisper 模型均可直接使用，只需修改 `--model-path`：

| 模型 | 大小 | 最低内存 | 推荐芯片 | 适用场景 |
|---|---|---|---|---|
| `mlx-community/whisper-large-v3-turbo` | ~1.6GB | 8GB | M1 Pro+ | **默认**，速度与质量最佳平衡 |
| `mlx-community/whisper-large-v3-mlx` | ~3GB | 16GB | M1 Pro+ | 最高质量，速度较慢 |
| `mlx-community/whisper-large-v3-mlx-4bit` | ~0.9GB | 8GB | M1+ | 低内存，质量略降 |
| `mlx-community/whisper-large-v3-turbo-8bit` | ~0.8GB | 8GB | M1+ | turbo 量化版，更省内存 |
| `mlx-community/distil-whisper-large-v3` | ~1.5GB | 8GB | M1 Pro+ | 蒸馏版，速度更快 |
| `mlx-community/whisper-medium-mlx` | ~1.5GB | 8GB | M1+ | 中等大小，多语言 |
| `mlx-community/whisper-small-mlx` | ~0.5GB | 8GB | M1+ | 轻量，适合实时场景 |
| `mlx-community/whisper-tiny-mlx` | ~0.15GB | 8GB | M1+ | 最小最快，质量有限 |

> **注意：** 内存需求包含模型权重 + 推理开销（约为模型大小的 2-3 倍）。8GB 设备建议使用量化或小型模型，为系统和其他应用留出余量。Apple Silicon 的统一内存由 CPU 和 GPU 共享，模型和推理缓冲区与系统内存竞争。

其他变体：纯英语版（`.en`）、量化版（2/4/8-bit）、FP32、语言专项微调（德语等）。完整列表见 [mlx-community on HuggingFace](https://huggingface.co/collections/mlx-community/whisper)。

## API 文档

服务启动后，可通过浏览器访问交互式 API 文档：

- **Swagger UI**：`http://localhost:8000/docs`
- **ReDoc**：`http://localhost:8000/redoc`

### 健康检查

```
GET /health
```

```json
{"status": "ok", "model": "whisper-large-v3-turbo"}
```

### 模型列表

```
GET /v1/models
```

```json
{
  "object": "list",
  "data": [
    {"id": "whisper-large-v3-turbo", "object": "model", "owned_by": "local"}
  ]
}
```

### 队列状态

```
GET /v1/queue/stats
```

```json
{"queue_size": 0, "queue_max_size": 10, "active": false}
```

### 语音转录

```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

**请求参数：**

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | file | 是 | 音频文件（mp3、wav、m4a、ogg、flac、aac、webm） |
| `model` | string | 是 | 模型 ID，如 `whisper-large-v3-turbo` |
| `language` | string | 否 | ISO 639-1 语言代码，留空自动检测 |
| `prompt` | string | 否 | 转录上下文提示 |
| `response_format` | string | 否 | `json`（默认）、`text`、`verbose_json`、`srt`、`vtt` |
| `temperature` | float | 否 | 0.0-1.0，默认 0.0 |
| `stream` | bool | 否 | 启用 SSE 流式输出，默认 false |

**请求示例：**

```bash
# JSON（默认）
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-large-v3-turbo

# SRT 字幕
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-large-v3-turbo \
  -F response_format=srt

# 流式输出
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-large-v3-turbo \
  -F stream=true

# 指定语言
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-large-v3-turbo \
  -F language=zh
```

**响应格式示例：**

`json`：
```json
{"text": "你好世界"}
```

`verbose_json`：
```json
{
  "task": "transcribe",
  "language": "zh",
  "duration": 5.52,
  "text": "你好世界",
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.1, "text": "你好世界", "no_speech_prob": 0.01}
  ]
}
```

`srt`：
```
1
00:00:00,000 --> 00:00:02,100
你好世界
```

`vtt`：
```
WEBVTT

00:00:00.000 --> 00:00:02.100
你好世界
```

**流式 SSE：**
```
data: {"text": "你好世界"}

data: [DONE]
```

**错误码：**

| 状态码 | Code | 原因 |
|---|---|---|
| 400 | `model_not_found` | 未知模型 ID |
| 400 | `invalid_response_format` | 不支持的响应格式 |
| 415 | `unsupported_audio_format` | 不支持的音频格式 |
| 503 | `queue_full` | 并发请求过多 |
| 503 | `queue_timeout` | 请求等待超时 |

所有错误均遵循 OpenAI 格式：
```json
{"error": {"message": "...", "type": "...", "code": "..."}}
```

## 配置

通过 CLI 参数或环境变量配置，CLI 参数优先级更高：

| CLI 参数 | 环境变量 | 默认值 | 说明 |
|---|---|---|---|
| `--host` | `WHISPER_HOST` | `0.0.0.0` | 监听地址 |
| `--port` | `WHISPER_PORT` | `8000` | 监听端口 |
| `--model-path` | `WHISPER_MODEL_PATH` | `mlx-community/whisper-large-v3-turbo` | HuggingFace 仓库或本地路径 |
| `--quantize` | `WHISPER_QUANTIZE` | 无 | 量化位数（4/8） |
| `--queue-max-size` | `WHISPER_QUEUE_MAX_SIZE` | `10` | 最大排队请求数，超出返回 503 |
| `--queue-timeout` | `WHISPER_QUEUE_TIMEOUT` | `300` | 队列等待超时（秒） |
| `--memory-cleanup-interval` | `WHISPER_MEMORY_CLEANUP_INTERVAL` | `20` | 每 N 次请求清理一次 Metal 缓存 |
| `--log-level` | `WHISPER_LOG_LEVEL` | `info` | 日志级别（debug/info/warning/error） |

使用 `.env` 文件配置服务部署（放在项目根目录）：

```bash
WHISPER_PORT=8000
WHISPER_MODEL_PATH=mlx-community/whisper-large-v3-turbo
WHISPER_QUEUE_MAX_SIZE=10
```

修改后重新安装服务即可生效：

```bash
./scripts/service.sh install && ./scripts/service.sh restart
```

## 使用 OpenAI SDK 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo",
        file=f,
        response_format="verbose_json",
    )
print(result.text)
```

接口完全兼容 OpenAI `/v1/audio/transcriptions`，任何支持 OpenAI API 的客户端只需修改 `base_url` 即可使用。

## 项目架构

```
main.py（CLI 入口）
  -> app/server.py（FastAPI 工厂、生命周期、Metal 清理中间件）
       -> app/api/audio.py     POST /v1/audio/transcriptions
       -> app/api/models.py    GET  /v1/models
       -> app/api/queue.py     GET  /v1/queue/stats
       -> app/registry.py      model_id → handler 查找
       -> app/worker.py        单线程推理队列
       -> app/handlers/
            base.py            BaseHandler ABC + AudioCapable mixin
            whisper.py         WhisperHandler（mlx_whisper.transcribe）
       -> app/schemas/         Pydantic 模型 + dataclass
       -> app/formatters.py    json/text/verbose_json/srt/vtt 格式转换
       -> app/audio.py         上传文件保存/校验/清理
```

**扩展方式：** 新增模型类型只需实现 `BaseHandler` + 能力 mixin（如 `ChatCapable`），添加 API Router，并在 lifespan 中注册。现有代码无需改动。

## 开发

```bash
# 运行测试
pytest -v

# 开发模式启动（热重载）
uvicorn app.server:create_app --factory --reload

# 代码检查
ruff check .
```

## License

MIT
