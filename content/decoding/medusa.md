---
id: medusa
title: Medusa
category: decoding
level: advanced
status: draft
readingMinutes: 10
tags:
  - Medusa
  - Speculative Decoding
  - Multi-Head
codeRefs:
  - vllm/v1/spec_decode/medusa.py
heroText: 在 target model 最后一层添加多个预测头，每个头独立预测不同未来位置的 token，[单次 forward](term:所有 draft token 在一次前向传播中并行生成，无需自回归。) 并行生成所有候选。
---

## 心智模型

Medusa 就像多个预言家同时预测未来：第一个预言家预测下一个词，第二个预测下下个词，第三个预测再下一个……所有预言家同时看同一份上下文，各自独立做出预测。一次 pass 就能得到所有位置的候选词。

:::diagram medusa-mental-model-html
```html
<div class="diagram-container">
  <div class="diagram">
    <div class="diagram-title">Target Hidden States</div>
    <div class="arch-row">
      <div class="arch-box">
        <div class="arch-proc-title">Head 0</div>
        <div class="muted">→ t+1: A</div>
      </div>
      <div class="arch-box">
        <div class="arch-proc-title">Head 1</div>
        <div class="muted">→ t+2: B</div>
      </div>
      <div class="arch-box">
        <div class="arch-proc-title">Head 2</div>
        <div class="muted">→ t+3: C</div>
      </div>
      <div class="arch-box">
        <div class="arch-proc-title">Head 3</div>
        <div class="muted">→ t+4: D</div>
      </div>
    </div>
    <div class="arch-highlight">Draft: A B C D（一次 forward）</div>
  </div>
</div>
```
:::

:::diagram-desc medusa-mental-model-html
Medusa 的心智模型：多个预测头（Head 0-3）同时从 target model 的 hidden states 中读取信息，每个头独立预测不同未来位置（t+1 到 t+4）的 token。所有预测在一次 forward pass 中完成，无需自回归循环。最终 Draft 为 A B C D。
:::

## Medusa 核心思路

Medusa 在 target model 最后一层 hidden states 之上添加**多个预测头**。每个头独立预测不同未来位置的 token，**单次 forward pass** 生成所有 draft token（并行，非自回归）。

与自回归方法（EAGLE、draft model）的关键区别：

| 特性 | Medusa | 自回归方法（EAGLE 等） |
|------|--------|----------------------|
| Draft 生成方式 | 并行，单次 forward | 顺序，k 次 forward |
| 每步延迟 | 低（1 次 draft forward） | 高（k 次 draft forward） |
| 接受率 | 较低（无 token 间依赖） | 较高（后续 token 基于前面 token） |
| 总体加速 | 取决于接受率与延迟的权衡 | 取决于接受率与 draft 开销的权衡 |

Medusa 的核心权衡：**用更低的接受率换取更快的 proposal 速度**。如果 target model 的 forward 开销远大于 draft 开销，Medusa 的单次 proposal 更高效。

## 实现细节

`MedusaProposer.propose()` 的实现简洁明了：

`vllm/v1/spec_decode/medusa.py`

```python
class MedusaProposer(SpecDecodeBaseProposer):
    def propose(self, target_hidden_states, ...):
        hidden_states = self.model(target_hidden_states)
        logits = self.compute_logits(hidden_states.blocks)
        draft_tokens = logits.argmax(dim=-1)
        return draft_tokens
```

### 执行步骤

1. 运行 `self.model(target_hidden_states)`：Medusa 模型处理 target 的 hidden states
2. `compute_logits(blocks)`：计算每个预测头的 logits
3. `argmax` per head：每个头取概率最大的 token
4. Stack 为 `[batch_size, num_heads]`：组装为 draft token 矩阵

### 关键特点

- **始终贪心**：Medusa 使用 argmax（贪心解码），不做随机采样
- **不存储 draft logits**：因为贪心模式不需要 p(x)/q(x) 计算
- **无 KV cache**：Medusa 模型本身不需要 KV cache，因为它不做自回归
- **并行度高**：所有 draft token 一次生成，延迟与 num_speculative_tokens 无关

:::diagram medusa-impl-flow-html
```html
<div class="diagram-container">
  <div class="diagram">
    <div class="engine-step-flow">
      <div class="engine-step">
        <div class="engine-step-num">1</div>
        <div class="engine-step-content">
          <div class="engine-step-title">Medusa Model Forward</div>
          <div class="engine-step-desc">hidden_states = model(target_hidden_states)</div>
        </div>
      </div>
      <div class="engine-step">
        <div class="engine-step-num">2</div>
        <div class="engine-step-content">
          <div class="engine-step-title">Compute Logits</div>
          <div class="engine-step-desc">每个 head 计算各自的 logits</div>
        </div>
      </div>
      <div class="engine-step">
        <div class="engine-step-num">3</div>
        <div class="engine-step-content">
          <div class="engine-step-title">Argmax Per Head</div>
          <div class="engine-step-desc">每个 head 取概率最大的 token</div>
        </div>
      </div>
      <div class="engine-step">
        <div class="engine-step-num">4</div>
        <div class="engine-step-content">
          <div class="engine-step-title">Stack Draft Tokens</div>
          <div class="engine-step-desc">[batch_size, num_heads]</div>
        </div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc medusa-impl-flow-html
Medusa 实现流程包含 4 个步骤：
1. **Medusa Model Forward**：将 target_hidden_states 传入 Medusa 模型，得到处理后的 hidden_states。
2. **Compute Logits**：每个预测头独立计算 logits。
3. **Argmax Per Head**：每个头取概率最大的 token（贪心解码）。
4. **Stack Draft Tokens**：将所有头的预测结果 stack 为 [batch_size, num_heads] 的 draft token 矩阵。
整个过程只有一次 forward，没有自回归循环。
:::

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--speculative-method` | None | 设为 `medusa` |
| `--speculative-model` | None | Medusa 模型路径 |
| `--num-speculative-tokens` | None | Draft token 数量（= Medusa head 数） |
