---
id: gemma4-spec
title: Gemma4 投机解码
category: decoding
level: intermediate
status: draft
readingMinutes: 10
tags:
  - Gemma4
  - Speculative Decoding
codeRefs:
  - vllm/v1/spec_decode/gemma4.py
  - vllm/model_executor/models/gemma4_mtp.py
heroText: Gemma4 内置 MTP 投机解码，[跨模型 KV 共享](term:Draft 层与 target model 共享 KV cache，避免重复存储。) 使 draft 推理几乎零额外显存开销。
---

## 心智模型

Gemma4 的 draft 模型就像和高级律师共享同一个办公室的初级律师——他们共用同一套参考书（KV cache），不需要复制任何文件。初级律师审阅时直接翻阅高级律师已经整理好的笔记，几乎零额外存储开销。

:::diagram gemma4-mental-model-html
```html
<div class="arch-diagram">
  <div class="arch-proc">
    <div class="arch-proc-title">Target Model</div>
    <div class="arch-box arch-highlight">KV Cache（共享）</div>
    <div class="arch-row">
      <div class="arch-box">K/V</div>
      <div class="arch-box">K/V</div>
      <div class="arch-box">K/V</div>
      <div class="arch-box">K/V</div>
    </div>
  </div>
  <div class="arch-channel">
    <div class="arch-arrow">↕</div>
    <div class="arch-label">共享 KV Cache</div>
  </div>
  <div class="arch-proc">
    <div class="arch-proc-title">Draft MTP Layers</div>
    <div class="arch-box">引用同一 block_table + kv_cache</div>
  </div>
  <div class="arch-channel">
    <div class="arch-box arch-highlight">零额外 KV 显存开销</div>
  </div>
</div>
```
:::

:::diagram-desc gemma4-mental-model-html
Gemma4 的心智模型：Target Model 和 Draft MTP Layers 共享同一份 KV Cache。Draft 层的 attention 直接引用 target model 的 block_table 和 kv_cache 张量，不需要分配额外的 KV 存储空间。这意味着 draft 推理的显存开销几乎为零。
:::

## Gemma4 MTP 特殊设计

Gemma4 的 MTP 实现有几个独特的优化设计，使其与其他 MTP 变体（DeepSeek、Step3.5）区分开来。

### constant_draft_positions

Gemma4 设置 `constant_draft_positions = True`：所有 draft 步预测的 query 都来自**同一个位置**（Q-only attention）。这意味着 draft 层只计算新的 Q，而 K/V 完全复用 target model 已有的 KV cache。

`vllm/v1/spec_decode/gemma4.py`

```python
class Gemma4Proposer(EagleProposer):
    constant_draft_positions = True
```

### 跨模型 KV 共享

`_setup_gemma4_kv_sharing()` 是 Gemma4 的核心优化，将 draft 层的 attention 接线到 target model 的 KV cache：

`vllm/v1/spec_decode/gemma4.py`

```python
def _setup_gemma4_kv_sharing(self, model):
    for draft_layer in model.model.layers:
        draft_attn = draft_layer.self_attn
        target_kv_cache = get_target_kv_cache(draft_attn.layer_name)
        draft_attn.kv_cache = target_kv_cache
        draft_attn.block_table = get_target_block_table()
```

### 多 KV Cache Group

Gemma4 模型有**两种 attention 机制**，对应不同的 KV cache group：

| KV Cache Group | 类型 | Head 维度 | 用途 |
|----------------|------|----------|------|
| Sliding Attention | 滑动窗口 | 较小 | 局部上下文 |
| Full Attention | 全量 | 较大 | 全局上下文 |

Draft MTP 层需要同时与两个 group 的 KV cache 共享，这意味着每个 draft attention 层需要引用两套 block_table 和 kv_cache。

### Centroids Masking

Gemma4 支持 centroids masking（中心点掩码）用于词汇表缩减，配合 CUDA graphs 使用：

`vllm/model_executor/models/gemma4_mtp.py`

```python
if self.config.centroids_config is not None:
    centroids_mask = compute_centroids_mask(logits, self.centroids)
    logits = logits.masked_fill(~centroids_mask, -float('inf'))
```

通过限制候选 token 到一个子集，centroids masking 可以加速采样过程。

## KV 共享机制

Gemma4 的 KV 共享是其最核心的设计优势。Draft attention 层直接引用 target model 的 KV cache 张量，无需任何额外分配。

### 共享原理

```python
draft_attn.kv_cache[0] = target_attn.kv_cache[0]
draft_attn.kv_cache[1] = target_attn.kv_cache[1]
```

### 显存节省分析

| 场景 | 无 KV 共享 | 有 KV 共享 |
|------|-----------|-----------|
| Target KV Cache | N blocks | N blocks |
| Draft KV Cache | N blocks（额外） | 0 blocks（共享） |
| 总显存 | 2N blocks | N blocks |
| 节省 | — | ~50% KV 显存 |

对于长序列（大 N），KV 共享带来的显存节省非常显著。Draft 推理只需少量额外显存用于 query 计算和中间激活值。

### 共享的一致性保证

KV 共享要求 draft 层和 target 层的 KV cache 布局完全一致：

- 相同的 `block_table` 映射
- 相同的 `kv_cache` 张量形状
- 相同的 `block_size` 和 `head_dim`

Gemma4 的 MTP 层在设计时保证了这些约束，使得共享成为可能。

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--speculative-method` | None | 设为 `mtp` |
| `--speculative-model` | None | Gemma4 MTP 模型路径 |
| `--num-speculative-tokens` | None | Draft token 数量（= MTP 层数） |

启动示例：

```bash
vllm serve google/gemma4-27b-it \
  --speculative-method mtp \
  --speculative-model google/gemma4-27b-it-mtp \
  --num-speculative-tokens 4
```
