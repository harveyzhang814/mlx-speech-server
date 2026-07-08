# ADR: Whisper 幻觉防护参数

**日期：** 2026-05-29  
**状态：** 已落地  
**影响文件：** `app/handlers/whisper.py`

---

## 背景

CocoScribe 调用 `POST /v1/audio/transcriptions` 时固定传入 `temperature=0`（单个 float）。上线后发现部分录音产生严重幻觉：

- 重复循环：`好,好,好,好,...`（连续 28 秒录音）
- YouTube 订阅文字：`请不吝点赞 订阅 转发 打赏...`（凭空生成）
- 连锁幻觉：第一段输出"有。"，后续所有段落持续生成"有。有。有。"（55 秒录音）

## 原因分析

### 问题一：temperature=0.0 单值禁用了 fallback 机制

`mlx_whisper.transcribe` 的 `temperature` 参数接受两种形式：

| 形式 | 行为 |
|------|------|
| 单个 float（如 `0.0`） | 只推理一次，无论结果好坏都直接返回 |
| tuple（如 `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`） | 从 0.0 开始，若检测到重复（compression_ratio > 2.4）或置信度过低（logprob < -1.0），自动用下一个温度重试 |

当客户端传入 `temperature=0.0`（单值），服务端原先直接透传给 mlx_whisper，fallback 重试机制失效。`compression_ratio_threshold` 和 `logprob_threshold` 虽然仍会评估，但没有备选温度可用，重复输出无法恢复。

### 问题二：condition_on_previous_text=True 导致连锁幻觉

默认值 `True` 让每个分段的推理都以上一段的输出文本为上下文。一旦某段产生幻觉（如"有。"），后续分段以此为条件继续生成，形成连锁。

## 决策

在 `WhisperHandler.transcribe` 中：

1. **temperature=0.0 时展开为 fallback 序列**：
   ```python
   temperature = (
       (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if params.temperature == 0.0 else params.temperature
   )
   ```
   客户端请求 `temperature=0` 的语义是"确定性/最高质量"，fallback 序列正是 Whisper 实现这一语义的正确机制。非零温度（用户明确要求随机性）则原样透传。

2. **固定传入 `condition_on_previous_text=False`**：
   各分段独立推理，切断连锁幻觉传播路径。代价是丧失跨分段上下文，但对语音转录场景影响极小（每段通常已是语义完整的句子）。

## 结果

修复后新增回归测试 3 条，全部通过：
- `test_temperature_zero_uses_fallback_schedule`
- `test_temperature_nonzero_passed_as_is`
- `test_condition_on_previous_text_always_false`
