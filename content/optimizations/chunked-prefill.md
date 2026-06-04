---
id: chunked-prefill
title: Chunked Prefill
category: optimizations
level: intermediate
status: draft
readingMinutes: 12
tags:
  - Chunked Prefill
  - Scheduler
codeRefs:
  - vllm/v1/core/sched/scheduler.py
  - vllm/config/scheduler.py
heroText: "长 [prefill](term:处理完整 prompt 的阶段，计算密集型。) 请求被拆分为多个 chunk 调度，避免单个长请求阻塞 [decode](term:逐 token 生成的阶段，访存密集型。) 请求。"
---

## 心智模型

想象一条高速公路，有**普通车道**和**快速车道**。长 prefill 请求就像慢速卡车——它们不应该堵住快速行驶的 decode 小汽车。Chunked prefill 把每辆卡车切成小段，让小汽车能在卡车段之间穿插通过。

:::diagram cp-mental-model
```html
<div class="diagram-container">
  <div class="diagram" data-ref="vllm/v1/core/sched/scheduler.py">
    <div class="diagram-title">Chunked Prefill 心智模型</div>
    <div class="sched-flow">
      <div class="sched-phase" data-phase="waiting">
        <div class="sched-phase-title">无 Chunked Prefill</div>
        <div class="sched-phase-steps">
          <div class="sched-step">长 Prefill (100K tokens)</div>
          <div class="sched-step">Decode 1</div>
          <div class="sched-step">Decode 2</div>
          <div class="sched-step">Decode 3</div>
        </div>
        <div class="muted">Decode 请求被阻塞多步</div>
      </div>
      <div class="sched-phase" data-phase="running">
        <div class="sched-phase-title">有 Chunked Prefill</div>
        <div class="sched-phase-steps">
          <div class="sched-step">Prefill Chunk 1</div>
          <div class="sched-step">Decode 1</div>
          <div class="sched-step">Prefill Chunk 2</div>
          <div class="sched-step">Decode 2</div>
          <div class="sched-step">Prefill Chunk 3</div>
          <div class="sched-step">Decode 3</div>
        </div>
        <div class="muted">Decode 请求穿插执行</div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc cp-mental-model
Chunked Prefill 心智模型对比了有无分块时的调度情况：

**无 Chunked Prefill**：一个长 prefill 请求（如 100K tokens）独占整个 token budget，连续执行多步。所有 decode 请求被阻塞，等待 prefill 完成后才能执行。这导致 decode 请求的延迟显著增加。

**有 Chunked Prefill**：长 prefill 被拆分为多个 chunk，每个 chunk 只使用部分 token budget。在 chunk 之间，decode 请求可以穿插执行。这保证了 decode 请求的连续性，降低了延迟。

关键收益：decode 请求不被长 prefill 阻塞、降低 decode 延迟、提高系统响应性。
:::

关键收益：**decode 请求不被长 prefill 阻塞**、**降低 decode 延迟**、**提高系统响应性**。

## 为什么需要 Chunked Prefill

没有 chunked prefill 时，一个长 prompt（如 100K tokens）会独占整个 token budget，阻塞所有 decode 请求：

- **单步阻塞**：假设 token budget = 256，一个 100K token 的 prefill 需要 ~400 步才能完成
- **decode 停滞**：这 400 步内，所有 decode 请求都无法执行，延迟飙升
- **用户体验差**：用户看到生成过程长时间停顿

有了 chunked prefill，长 prompt 被拆成小 chunk，每个 chunk 只用部分 budget：

- **穿插执行**：每步执行一个 prefill chunk + 多个 decode 请求
- **decode 连续**：decode 请求几乎不受影响，保持流畅
- **公平调度**：长 prefill 和 decode 请求都能获得资源

## 分块逻辑

调度器在处理 RUNNING 请求时，会根据配置决定是否对长 prefill 进行分块：

```python
# vllm/v1/core/sched/scheduler.py
# RUNNING 请求的分块逻辑
num_new_tokens = num_tokens_with_spec - num_computed_tokens

if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold

num_new_tokens = min(num_new_tokens, token_budget)
```

- `num_tokens_with_spec`：请求的总 token 数（含投机解码的 draft tokens）
- `num_computed_tokens`：已计算的 token 数
- `long_prefill_token_threshold`：长 prefill 的 chunk 大小阈值
- `token_budget`：本步可用的 token 总预算

如果 `num_new_tokens` 超过阈值，则限制为阈值大小，实现分块。

## 部分 Prefill 追踪

一个 prefill 请求被分块后，会在多个调度步骤中持续执行：

```python
# vllm/v1/core/sched/scheduler.py
is_prefill_chunk = num_computed_tokens < (num_tokens + num_output_placeholders)
```

- `num_computed_tokens`：已计算的 token 数
- `num_tokens`：prompt 的总 token 数
- `num_output_placeholders`：输出占位符（用于投机解码等场景）

当 `is_prefill_chunk = True` 时，请求还在 prefill 阶段，需要继续计算。每步执行后，`num_computed_tokens` 会增加，直到等于 `num_tokens`，prefill 完成。

部分 prefill 请求会留在 `RUNNING` 队列中，下次调度时继续从上次中断的地方执行。

## Mamba 块对齐

对于包含 Mamba 层的混合模型，`num_new_tokens` 必须是 `block_size` 的倍数，以实现块对齐的 KV cache：

```python
# vllm/v1/core/sched/scheduler.py
def _mamba_block_aligned_split(num_new_tokens, block_size):
    remainder = num_new_tokens % block_size
    if remainder == 0:
        return num_new_tokens
    return num_new_tokens - remainder
```

这确保了 Mamba 层的 KV cache 写入是块对齐的，避免跨块边界的复杂处理。

## 调度优先级

Chunked prefill 的调度分为两个阶段：

:::diagram cp-sched-priority
```html
<div class="diagram-container">
  <div class="diagram" data-ref="vllm/v1/core/sched/scheduler.py">
    <div class="diagram-title">调度优先级</div>
    <div class="sched-flow">
      <div class="sched-phase" data-phase="running">
        <div class="sched-phase-title">Phase 1 - 调度 RUNNING 请求</div>
        <div class="sched-phase-steps">
          <div class="sched-step">Decode 请求（优先）</div>
          <div class="sched-step">部分 Prefill 请求（继续执行）</div>
        </div>
      </div>
      <div class="sched-phase" data-phase="waiting">
        <div class="sched-phase-title">Phase 2 - 调度 WAITING 请求</div>
        <div class="sched-phase-steps">
          <div class="sched-step">新 Prefill 请求（仅当无抢占时）</div>
        </div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc cp-sched-priority
调度优先级展示了 chunked prefill 的两阶段调度：

**Phase 1 - 调度 RUNNING 请求**：优先处理已经在运行的请求。Decode 请求优先级最高，确保生成过程流畅。部分 prefill 请求（之前被分块的）继续执行，直到完成。

**Phase 2 - 调度 WAITING 请求**：只有在没有抢占发生时，才会从 WAITING 队列中接纳新的 prefill 请求。这避免了新请求挤占正在运行的 decode 请求的资源。

关键原则：running decode 请求的优先级 > 新 prefill 请求。
:::

关键原则：**running decode 请求的优先级 > 新 prefill 请求**。这确保了 decode 请求不会被新到达的 prefill 请求阻塞。

## 关键配置

| 参数 | 默认值 | 说明 | 源码 |
|------|--------|------|------|
| `enable_chunked_prefill` | True | 是否启用 chunked prefill | scheduler_config.py |
| `long_prefill_token_threshold` | 0（自动） | 长 prefill 的 chunk 大小。0 表示自动 = 4% max_model_len | scheduler_config.py |
| `max_num_partial_prefills` | 1 | 最大同时执行的部分 prefill 请求数 | scheduler_config.py |
| `max_long_partial_prefills` | 1 | 最大同时执行的长部分 prefill 请求数 | scheduler_config.py |
| `scheduler_reserve_full_isl` | True | 是否为完整 ISL（inter-sequence latency）预留资源 | scheduler_config.py |
