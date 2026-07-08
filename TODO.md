# TODO / Backlog

## 🚧 待开发

### 为空闲卸载超时添加 WHISPER_IDLE_UNLOAD_TIMEOUT 环境变量
**优先级**: P2 | **日期**: 2026-07-08

当前 `_IDLE_TIMEOUT = 1800.0` 硬编码在 `app/handlers/whisper.py`。需新增 `WHISPER_IDLE_UNLOAD_TIMEOUT` 环境变量（单位：秒，设为 0 则禁用），通过 `ServerConfig.from_env()` 读取并传入 `WhisperHandler`，与现有 `WHISPER_*` 体系保持一致，方便 launchd 服务通过 `~/.config/mlx-speech-server/config.env` 按机器配置。

---

## ✅ 已完成
