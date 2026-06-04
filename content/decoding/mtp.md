---
id: mtp
title: MTP 投机解码
category: decoding
level: advanced
status: draft
readingMinutes: 14
tags:
  - MTP
  - Speculative Decoding
  - Multi-Token Prediction
codeRefs:
  - vllm/v1/spec_decode/llm_base_proposer.py
  - vllm/model_executor/models/step3p5_mtp.py
  - vllm/model_executor/models/gemma4_mtp.py
heroText: 在 target model 基础上添加 [MTP 层](term:Multi-Token Prediction 预测层，共享 target 的 embedding 和 LM head，零额外词表开销。) 预测多个未来 token，不同变体各有独特优化（跨模型 KV 共享、多 KV Cache Group、centroids masking）。
---

## 心智模型

MTP（Multi-Token Prediction）就像一个律师在审阅当前文件的同时，也顺手为接下来的几段写草稿。草稿层共享同样的法律知识（embeddings、LM head），不需要额外学习新的词汇表——只是多了一层快速推理的能力。不同 MTP 变体还有各自的独特优化：Gemma4 的 draft 层与 target 共享办公室（KV cache），Step 3.5 的每个草稿层依次传递笔记（hidden states），DeepSeek 则在架构上做了深度优化。

:::diagram mtp-mental-model-html
```html
<div class="arch-diagram">
  <div class="arch-proc">
    <div class="arch-proc-title">Target Model</div>
    <div class="arch-row">
      <div class="arch-box">Embedding</div>
      <div class="arch-box">LM Head</div>
    </div>
    <div class="arch-box arch-highlight">KV Cache（部分变体共享）</div>
  </div>
  <div class="arch-channel">
    <div class="arch-arrow">↓</div>
    <div class="arch-label">hidden_states + embedding</div>
  </div>
  <div class="arch-row">
    <div class="arch-module" data-phase="running">
      <div class="arch-proc-title">MTP Layer 1</div>
      <div class="muted">→ t+1 draft</div>
    </div>
    <div class="arch-module" data-phase="running">
      <div class="arch-proc-title">MTP Layer 2</div>
      <div class="muted">→ t+2 draft</div>
    </div>
    <div class="arch-module" data-phase="running">
      <div class="arch-proc-title">MTP Layer 3</div>
      <div class="muted">→ t+3 draft</div>
    </div>
  </div>
  <div class="muted">共享 Embedding + LM Head（零额外词表开销）</div>
</div>
```
:::

:::diagram-desc mtp-mental-model-html
MTP 的心智模型：Target Model 产出 hidden_states 后，传递给多个 MTP Layer。每个 MTP Layer 预测不同未来位置（t+1, t+2, t+3）的 draft token。所有 MTP Layer 共享 target model 的 Embedding 和 LM Head，不需要额外的词表参数。部分变体（如 Gemma4）还共享 KV Cache，进一步降低显存开销。
:::

## MTP 总体架构

MTP 层的核心架构遵循统一模式：将 embedding 和 hidden_states 融合后，通过投影层和 Transformer block 生成预测。

```
enorm (embedding norm) + hnorm (hidden state norm)
    → eh_proj (project [embedding, hidden_state] → hidden_size)
    → mtp_block (full DecoderLayer)
    → shared_head (norm + LM head)
```

### 数据流

`vllm/model_executor/models/step3p5_mtp.py`

```python
class Step3p5AMultiTokenPredictorLayer(nn.Module):
    def forward(self, input_ids, hidden_states, position_ids, ...):
        e_normed = self.enorm(self.embed(input_ids))
        h_normed = self.hnorm(hidden_states)
        eh_input = torch.cat([e_normed, h_normed], dim=-1)
        eh_proj = self.eh_proj(eh_input)
        mtp_output = self.mtp_block(eh_proj, position_ids, ...)
        logits = self.shared_head(mtp_output)
        return logits, mtp_output
```

### 多个 MTP 层

多个 MTP 层按 step 索引堆叠。第 i 个 MTP 层的输入来自第 i-1 个 MTP 层的输出 hidden_states：

| 层 | 输入 | 输出 |
|----|------|------|
| MTP Layer 1 | target hidden_states + token embedding | t+1 draft logits + hidden_states |
| MTP Layer 2 | MTP Layer 1 hidden_states + token embedding | t+2 draft logits + hidden_states |
| MTP Layer 3 | MTP Layer 2 hidden_states + token embedding | t+3 draft logits + hidden_states |

每个 MTP 层共享同一个 `shared_head`（norm + LM head），确保预测空间一致。

## 支持的 MTP 变体

vLLM 支持多种 MTP 变体，每种针对特定模型架构做了优化：

| 变体 | 模型 | 代码位置 | 特殊优化 |
|------|------|----------|----------|
| `deepseek_mtp` | DeepSeek-V3/R1 | `vllm/model_executor/models/deepseek_mtp.py` | 深度架构优化 |
| `gemma4_mtp` | Gemma 4 | `vllm/model_executor/models/gemma4_mtp.py` | KV 共享、多 KV Cache Group、centroids masking |
| `step3p5_mtp` | Step 3.5 | `vllm/model_executor/models/step3p5_mtp.py` | 标准 MTP 架构 |
| `qwen3_next_mtp` | Qwen3-Next | `vllm/model_executor/models/qwen3_next_mtp.py` | — |
| `mimo_mtp` | MiMo | `vllm/model_executor/models/mimo_mtp.py` | — |
| `mimo_v2_mtp` | MiMo V2 | `vllm/model_executor/models/mimo_v2_mtp.py` | — |
| `glm4_moe_mtp` | GLM4-MoE | `vllm/model_executor/models/glm4_moe_mtp.py` | — |
| `glm4_moe_lite_mtp` | GLM4-MoE Lite | `vllm/model_executor/models/glm4_moe_lite_mtp.py` | — |
| `ernie_mtp` | ERNIE | `vllm/model_executor/models/ernie_mtp.py` | — |
| `nemotron_h_mtp` | Nemotron-H | `vllm/model_executor/models/nemotron_h_mtp.py` | — |
| `exaone_moe_mtp` | EXAONE-MoE | `vllm/model_executor/models/exaone_moe_mtp.py` | — |
| `exaone4_5_mtp` | EXAONE 4.5 | `vllm/model_executor/models/exaone4_5_mtp.py` | — |
| `qwen3_5_mtp` | Qwen3.5 | `vllm/model_executor/models/qwen3_5_mtp.py` | — |
| `longcat_flash_mtp` | LongCat-Flash | `vllm/model_executor/models/longcat_flash_mtp.py` | — |
| `pangu_ultra_moe_mtp` | Pangu Ultra MoE | `vllm/model_executor/models/pangu_ultra_moe_mtp.py` | — |
| `hy_v3_mtp` | HY V3 | `vllm/model_executor/models/hy_v3_mtp.py` | — |

## Step 3.5 MTP

Step 3.5 采用标准 MTP 架构，每个 MTP 层依次接收上一层的 hidden_states，自回归式地预测未来 token。

:::diagram step3p5-flow-html
```html
<div class="engine-step-flow">
  <div class="engine-step">
    <div class="engine-step-num">1</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Target Forward</div>
      <div class="engine-step-desc">产出 hidden_states + next_token</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">2</div>
    <div class="engine-step-content">
      <div class="engine-step-title">MTP Layer 1</div>
      <div class="engine-step-desc">hidden_states + embedding → t+1 draft</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">3</div>
    <div class="engine-step-content">
      <div class="engine-step-title">MTP Layer 2</div>
      <div class="engine-step-desc">Layer 1 hidden_states + embedding → t+2 draft</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">4</div>
    <div class="engine-step-content">
      <div class="engine-step-title">MTP Layer N</div>
      <div class="engine-step-desc">Layer N-1 hidden_states + embedding → t+N draft</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">5</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Return Draft</div>
      <div class="engine-step-desc">[t+1, t+2, ..., t+N] draft tokens</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc step3p5-flow-html
Step 3.5 MTP 的数据流：Target Model 先执行 forward 产出 hidden_states 和 next_token。然后 MTP Layer 1 接收 hidden_states 和 embedding，预测 t+1 位置的 draft token 并输出自己的 hidden_states。后续 MTP Layer 依次接收上一层的 hidden_states，预测更远位置的 draft token。最终返回 N 个 draft tokens。
:::

### Step 3.5 架构细节

`vllm/model_executor/models/step3p5_mtp.py`

```python
class Step3p5AMultiTokenPredictorLayer(nn.Module):
    def __init__(self, config, embedding, shared_head):
        self.enorm = RMSNorm(config.hidden_size)
        self.hnorm = RMSNorm(config.hidden_size)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.mtp_block = DecoderLayer(...)
        self.embed = embedding
        self.shared_head = shared_head
```

## Gemma4 MTP

Gemma4 的 MTP 实现有几个独特的优化设计，使其与其他 MTP 变体区分开来。

### KV 共享心智模型

Gemma4 的 draft 模型就像和高级律师共享同一个办公室的初级律师——他们共用同一套参考书（KV cache），不需要复制任何文件。初级律师审阅时直接翻阅高级律师已经整理好的笔记，几乎零额外存储开销。

:::diagram gemma4-kv-sharing-html
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

:::diagram-desc gemma4-kv-sharing-html
Gemma4 的 KV 共享机制：Target Model 和 Draft MTP Layers 共享同一份 KV Cache。Draft 层的 attention 直接引用 target model 的 block_table 和 kv_cache 张量，不需要分配额外的 KV 存储空间。这意味着 draft 推理的显存开销几乎为零。
:::

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

### 显存节省分析

| 场景 | 无 KV 共享 | 有 KV 共享 |
|------|-----------|-----------|
| Target KV Cache | N blocks | N blocks |
| Draft KV Cache | N blocks（额外） | 0 blocks（共享） |
| 总显存 | 2N blocks | N blocks |
| 节省 | — | ~50% KV 显存 |

对于长序列（大 N），KV 共享带来的显存节省非常显著。Draft 推理只需少量额外显存用于 query 计算和中间激活值。

## vLLM 统一实现

vLLM 将所有 MTP 变体统一在 `method="mtp"` 下，通过 `EagleProposer` 实现。

### 统一接口

`vllm/v1/spec_decode/llm_base_proposer.py`

```python
def use_eagle(speculative_config):
    method = speculative_config.method
    return method in ("eagle", "eagle3", "mtp")
```

所有 MTP 变体都通过 `EagleProposer` 类实现，差异在于模型层的具体实现。

### MTP 与 EAGLE 的关键区别

MTP 的 `model_returns_tuple()` 返回 `False`——MTP 模型只返回 hidden_states（非 tuple），而 EAGLE 可能返回 tuple（hidden_states + 额外信息）。

`vllm/v1/spec_decode/llm_base_proposer.py`

```python
def model_returns_tuple(self):
    if self.speculative_config.method == "mtp":
        return False
    return True
```

### 变体检测

vLLM 通过模型配置自动检测 MTP 变体，加载对应的模型实现：

```python
if "gemma4" in model_config.architectures:
    from vllm.model_executor.models.gemma4_mtp import Gemma4ForCausalLM
elif "step3p5" in model_config.architectures:
    from vllm.model_executor.models.step3p5_mtp import Step3p5ForCausalLM
```

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--speculative-method` | None | 设为 `mtp` |
| `--speculative-model` | None | MTP 模型路径 |
| `--num-speculative-tokens` | None | Draft token 数量（= MTP 层数） |

### Step 3.5 启动示例

```bash
vllm serve step/step-3.5-27b \
  --speculative-method mtp \
  --speculative-model step/step-3.5-27b-mtp \
  --num-speculative-tokens 4
```

### Gemma4 启动示例

```bash
vllm serve google/gemma4-27b-it \
  --speculative-method mtp \
  --speculative-model google/gemma4-27b-it-mtp \
  --num-speculative-tokens 4
```
