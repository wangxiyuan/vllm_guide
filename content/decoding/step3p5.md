---
id: step3p5
title: Step 3.5
category: decoding
level: advanced
status: draft
readingMinutes: 10
tags:
  - Step3.5
  - MTP
  - Speculative Decoding
codeRefs:
  - vllm/v1/spec_decode/llm_base_proposer.py
  - vllm/model_executor/models/step3p5_mtp.py
heroText: 基于 [MTP](term:Multi-Token Prediction，在 target model 基础上添加一个或多个预测层，共享 embedding 和 LM head。) 的多 token 预测，共享 target model 的 embedding 和 LM head。
---

## 心智模型

MTP（Multi-Token Prediction）就像一个律师在审阅当前文件的同时，也顺手为接下来的几段写草稿。草稿层共享同样的法律知识（embeddings、LM head），不需要额外学习新的词汇表——只是多了一层快速推理的能力。

:::diagram step3p5-mental-model-html
```html
<div class="arch-diagram">
  <div class="arch-proc">
    <div class="arch-proc-title">Target Model</div>
    <div class="arch-row">
      <div class="arch-box">Embedding</div>
      <div class="arch-box">LM Head</div>
    </div>
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
  <div class="muted">共享 Embedding + LM Head</div>
</div>
```
:::

:::diagram-desc step3p5-mental-model-html
Step 3.5 MTP 的心智模型：Target Model 产出 hidden_states 后，传递给多个 MTP Layer。每个 MTP Layer 预测不同未来位置（t+1, t+2, t+3）的 draft token。所有 MTP Layer 共享 target model 的 Embedding 和 LM Head，不需要额外的词表参数。
:::

## MTP 架构

Step 3.5 的 MTP 层（`Step3p5AMultiTokenPredictorLayer`）由以下组件构成：

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

## 在 vLLM 中的统一

vLLM 将所有 MTP 变体统一在 `method="mtp"` 下，通过 `EagleProposer` 实现。

### 支持的 MTP 变体

| 变体 | 模型 | 代码位置 |
|------|------|----------|
| `deepseek_mtp` | DeepSeek-V3/R1 | `vllm/model_executor/models/deepseek_mtp.py` |
| `gemma4_mtp` | Gemma 4 | `vllm/model_executor/models/gemma4_mtp.py` |
| `step3p5_mtp` | Step 3.5 | `vllm/model_executor/models/step3p5_mtp.py` |
| `qwen3_next_mtp` | Qwen3-Next | `vllm/model_executor/models/qwen3_next_mtp.py` |

### 统一实现

`vllm/v1/spec_decode/llm_base_proposer.py`

```python
def use_eagle(speculative_config):
    method = speculative_config.method
    return method in ("eagle", "eagle3", "mtp")
```

### MTP 与 EAGLE 的关键区别

MTP 的 `model_returns_tuple()` 返回 `False`——MTP 模型只返回 hidden_states（非 tuple），而 EAGLE 可能返回 tuple（hidden_states + 额外信息）。

`vllm/v1/spec_decode/llm_base_proposer.py`

```python
def model_returns_tuple(self):
    if self.speculative_config.method == "mtp":
        return False
    return True
```

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--speculative-method` | None | 设为 `mtp` |
| `--speculative-model` | None | MTP 模型路径 |
| `--num-speculative-tokens` | None | Draft token 数量（= MTP 层数） |
