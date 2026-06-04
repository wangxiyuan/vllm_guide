---
id: eagle
title: EAGLE
category: decoding
level: advanced
status: draft
readingMinutes: 12
tags:
  - EAGLE
  - Speculative Decoding
codeRefs:
  - vllm/v1/spec_decode/eagle.py
  - vllm/v1/spec_decode/llm_base_proposer.py
heroText: 利用 target model 的 [hidden states](term:Transformer 最后一层的输出向量，包含了丰富的语义信息。) 作为轻量 draft model 的输入，自回归生成候选 token。
---

## 心智模型

EAGLE 就像一个精明的助理，他不仅看高级律师的最终文件，还阅读律师的详细笔记（hidden states）。笔记包含了比最终文本丰富得多的信息——每个词的语义倾向、语法角色、上下文关联。有了这些笔记，助理的草稿准确率远高于只看最终文本。

:::diagram eagle-mental-model-html
```html
<div class="arch-diagram">
  <div class="arch-proc">
    <div class="arch-proc-title">Target Model 输出</div>
    <div class="arch-box arch-highlight">Hidden States（丰富语义）</div>
    <div class="arch-box">Token IDs（离散文本）</div>
  </div>
  <div class="arch-channel">
    <div class="arch-arrow">→</div>
    <div class="arch-label">双重输入</div>
  </div>
  <div class="arch-proc">
    <div class="arch-proc-title">EAGLE Head（轻量）</div>
    <div class="arch-flow-label">自回归生成 k 个 draft token</div>
  </div>
  <div class="arch-channel">
    <div class="arch-arrow">→</div>
    <div class="arch-label">候选 token</div>
  </div>
  <div class="arch-proc">
    <div class="arch-proc-title">Target Model</div>
    <div class="arch-flow-label">一次 forward 验证</div>
  </div>
</div>
```
:::

:::diagram-desc eagle-mental-model-html
EAGLE 的心智模型展示了其双重输入策略：Target Model 的 Hidden States（包含丰富语义信息）和 Token IDs（离散文本）同时作为 EAGLE Head 的输入。EAGLE Head 是一个轻量模型，自回归生成 k 个 draft token，然后提交给 Target Model 一次 forward 验证。Hidden States 提供了比纯文本更丰富的上下文信息，使得 EAGLE 的 draft 准确率显著高于仅使用 token ID 的方法。
:::

## EAGLE 核心思路

EAGLE 的核心创新在于利用 target model 的 **hidden states** 作为 draft model 的输入，而非仅使用 token embedding。Hidden states 包含了比 token ID 丰富得多的语义信息，使得 draft token 的接受率大幅提升。

关键设计：

- `pass_hidden_states_to_model = True`：target model 的 hidden states 被传递给 EAGLE 模型
- V1 中使用 `SpecDecodeBaseProposer.propose()` 作为统一接口
- EAGLE 模型是一个轻量的自回归模型，参数量远小于 target model

`vllm/v1/spec_decode/eagle.py`

```python
class EagleProposer(SpecDecodeBaseProposer):
    def __init__(self, ...):
        self.model = model
        self.pass_hidden_states_to_model = True
```

## Draft 生成流程

EAGLE 通过自回归方式逐个生成 draft token。每一步都利用上一步的 hidden states 更新输入。

:::diagram eagle-draft-flow-html
```html
<div class="engine-step-flow">
  <div class="engine-step">
    <div class="engine-step-num">1</div>
    <div class="engine-step-content">
      <div class="engine-step-title">set_inputs_first_pass</div>
      <div class="engine-step-desc">shift target token_ids +1, 插入 next_token_ids, 复制 hidden_states</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">2</div>
    <div class="engine-step-content">
      <div class="engine-step-title">First Forward</div>
      <div class="engine-step-desc">EAGLE model forward → 第一个 draft token 的 logits</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">3</div>
    <div class="engine-step-content">
      <div class="engine-step-title">First Sample</div>
      <div class="engine-step-desc">采样第一个 draft token</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">4</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Autoregressive Loop</div>
      <div class="engine-step-desc">更新 input_ids/positions/attention → forward → sample → 重复 k-1 次</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">5</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Return Draft</div>
      <div class="engine-step-desc">[batch_size, num_speculative_tokens]</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc eagle-draft-flow-html
EAGLE Draft 生成流程包含 5 个步骤：
1. **set_inputs_first_pass**：将 target token_ids 右移一位，在最后位置插入 next_token_ids，复制 target hidden_states 作为 EAGLE 模型的初始输入。
2. **First Forward**：EAGLE 模型执行前向传播，产出第一个 draft token 位置的 logits。
3. **First Sample**：从 logits 中采样第一个 draft token。
4. **Autoregressive Loop**：将采样的 token 更新到 input_ids、positions 和 attention mask 中，再次 forward 并采样，重复 k-1 次直到生成所有 draft token。
5. **Return Draft**：返回形状为 [batch_size, num_speculative_tokens] 的 draft token 矩阵。
:::

:::steps id=eagle-draft-steps
### 1. 准备首次输入
`set_inputs_first_pass()`：shift target token_ids +1，插入 next_token_ids，复制 hidden_states。
`vllm/v1/spec_decode/eagle.py`

### 2. EAGLE 首次 Forward
运行 EAGLE model forward，获得第一个 draft 位置的 logits。
`vllm/v1/spec_decode/eagle.py`

### 3. 首次采样
从 logits 采样第一个 draft token。
`vllm/v1/spec_decode/eagle.py`

### 4. 自回归循环
更新 input_ids、positions、attention，forward + sample，重复 k-1 次。
`vllm/v1/spec_decode/eagle.py`

### 5. 返回 Draft Tokens
返回 [batch_size, num_speculative_tokens] 的 draft token 矩阵。
`vllm/v1/spec_decode/eagle.py`
:::

## EAGLE3 扩展

EAGLE3 在 EAGLE 基础上引入**辅助 hidden states**——不仅使用最后一层的 hidden states，还利用中间层的输出。

- `combine_hidden_states()`：将多个层的 hidden states 拼接作为 EAGLE3 模型的输入
- 通过 `speculative_config.method == "eagle3"` 检测 EAGLE3 模式
- 中间层的 hidden states 提供了不同抽象级别的语义信息，进一步提升 draft 准确率

`vllm/v1/spec_decode/eagle.py`

```python
def combine_hidden_states(self, hidden_states_list):
    return torch.cat(hidden_states_list, dim=-1)
```

## 概率采样 vs 贪心

EAGLE 的 proposer 采样方法影响 draft 质量和接受率：

| 方法 | 策略 | Draft Logits 存储 | 特点 |
|------|------|-------------------|------|
| `greedy` | argmax | 不存储 | 速度快，但 acceptance rate 略低 |
| `probabilistic` | 随机采样 | 存储 | acceptance rate 更高（用于 rejection sampling 的 p(x)/q(x) 计算） |

`vllm/v1/spec_decode/llm_base_proposer.py`

```python
if self.draft_sample_method == "greedy":
    draft_tokens = logits.argmax(dim=-1)
else:
    draft_tokens = torch.multinomial(probs, num_samples=1)
    self.draft_logits = logits
```

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--speculative-model` | None | EAGLE 模型路径 |
| `--num-speculative-tokens` | None | Draft token 数量 |
| `--speculative-method` | None | 设为 `eagle` 或 `eagle3` |
| `--draft-sample-method` | "greedy" | greedy / probabilistic |
