# docs/ 文档索引

## reference/ — 参考文档

稳定的 API 事实、接口定义，供人类与 Agent 查阅。

| 文件 | 用途 |
|------|------|
| [reference/api.md](reference/api.md) | 完整 HTTP API 文档：所有端点、请求参数、响应结构、错误码、语言码与音频格式 |

## adr/ — 架构决策记录

已落地的设计决策，不可修改，仅供回溯。

| 文件 | 用途 |
|------|------|
| [adr/2026-03-31-cli-subcommands-design.md](adr/2026-03-31-cli-subcommands-design.md) | CLI 子命令（install/start/stop 等）的设计方案与选型依据 |
| [adr/2026-03-31-mlx-speech-server-design.md](adr/2026-03-31-mlx-speech-server-design.md) | 服务整体架构设计：请求流、worker 模型、handler 抽象 |
| [adr/2026-04-01-whisper-language-normalization-design.md](adr/2026-04-01-whisper-language-normalization-design.md) | 语言参数规范化设计：BCP 47 → ISO 639-1，400 错误处理 |

## archive/ — 历史归档

已完成的实现计划与临时笔记，只读，不再维护。

| 文件 | 用途 |
|------|------|
| [archive/2026-03-31-cli-subcommands.md](archive/2026-03-31-cli-subcommands.md) | CLI 子命令功能实现任务清单 |
| [archive/2026-03-31-mlx-speech-server.md](archive/2026-03-31-mlx-speech-server.md) | 服务初始实现任务清单 |
| [archive/2026-04-01-service-dual-install-support.md](archive/2026-04-01-service-dual-install-support.md) | 本地 / PyPI 双安装模式实现任务清单 |
| [archive/2026-04-01-whisper-language-normalization.md](archive/2026-04-01-whisper-language-normalization.md) | 语言规范化功能实现任务清单 |
| [archive/chat-example.md](archive/chat-example.md) | 调试期间的查询优化笔记 |
