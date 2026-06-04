---
id: dflash
title: DFlash
category: decoding
level: advanced
status: draft
readingMinutes: 10
tags:
  - DFlash
  - Speculative Decoding
  - In-Fill
codeRefs:
  - vllm/v1/spec_decode/dflash.py
heroText: 并行起草：使用 [非因果注意力](term:query 从 query embedding 来，K/V 从 target hidden states 来，允许 draft token 同时关注所有位置。) 一次性生成所有候选 token，无需自回归循环。
---

## 心智模型

传统的自回归起草像逐词写文档——写完第一个词才能写第二个。DFlash 则像填空题：所有空位同时填写。context（target hidden states）提供参考信息（K/V），mask token 位置被替换为 draft 预测，一次 pass 完成。

:::diagram dflash-mental-model-html
```html
<div class="arch-diagram">
  <div class="arch-row">
    <div class="arch-box">
      <div class="arch-proc-title">Context K/V（来自 Target Hidden States）</div>
      <div class="arch-row">
        <span class="arch-module">K₁</span>
        <span class="arch-module">K₂</span>
        <span class="arch-module">K₃</span>
        <span class="muted">...</span>
      </div>
    </div>
  </div>
  <div class="arch-channel">
    <div class="arch-arrow">↓</div>
    <div class="arch-label">非因果注意力</div>
  </div>
  <div class="arch-row">
    <div class="arch-box">
      <div class="arch-proc-title">Query Embeddings（mask token 位置）</div>
      <div class="arch-row">
        <span class="arch-highlight">[M]₁</span>
        <span class="arch-highlight">[M]₂</span>
        <span class="arch-highlight">[M]₃</span>
        <span class="arch-highlight">[M]₄</span>
      </div>
    </div>
  </div>
  <div class="arch-channel">
    <div class="arch-arrow">↓</div>
    <div class="arch-label">并行预测</div>
  </div>
  <div class="arch-row">
    <div class="arch-box">
      <div class="arch-proc-title">Draft Tokens</div>
      <div class="arch-row">
        <span class="arch-module">A</span>
        <span class="arch-module">B</span>
        <span class="arch-module">C</span>
        <span class="arch-module">D</span>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc dflash-mental-model-html
DFlash 的心智模型：Context 的 K/V 来自 target hidden states（已有序列的语义信息），Query 来自 mask token 位置的 embedding。通过非因果注意力机制，所有 mask token 同时关注 context 的所有位置，一次性并行预测所有 draft token（A B C D）。无需自回归循环。
:::

## DFlash 核心思路

DFlash 采用**并行起草**策略，核心标志是 `parallel_drafting = True`。

### 非因果注意力

DFlash 使用非因果（non-causal）注意力机制：

- **K/V 来源**：target model 的 hidden states（context）
- **Q 来源**：query embeddings（mask token 位置）
- 每个查询位置可以同时关注 context 的所有位置，不受因果掩码限制

### 预计算 Context KV

在 draft forward 之前，`precompute_and_store_context_kv()` 将 target model 的 hidden states 预计算为 K/V 并插入到 draft 模型的 KV cache 中。这样 draft forward 时只需计算 query 部分，无需重新处理 context。

`vllm/v1/spec_decode/dflash.py`

```python
class DFlashProposer(SpecDecodeBaseProposer):
    parallel_drafting = True

    def precompute_and_store_context_kv(self, target_hidden_states):
        context_kv = self.model.compute_kv(target_hidden_states)
        self.model.insert_context_kv(context_kv)
```

### Mask Token

并行起草位置使用 `mask_token`（来自 `dflash_config`）作为占位符。这些位置被 DFlash 模型的 query embedding 替换，通过非因果注意力一次性预测所有 draft token。

`vllm/v1/spec_decode/dflash.py`

```python
mask_token_id = self.dflash_config.mask_token_id
draft_input_ids = torch.full(
    (batch_size, num_spec_tokens), mask_token_id
)
```

## 与 EAGLE 的区别

DFlash 和 EAGLE 是两种截然不同的起草策略：

| 特性 | EAGLE | DFlash |
|------|-------|--------|
| 起草方式 | 自回归（顺序） | 并行（一次 pass） |
| 注意力类型 | 因果注意力 | 非因果注意力 |
| Draft 生成延迟 | k 次 forward | 1 次 forward |
| 接受率 | 较高（token 间有依赖） | 较低（token 间无依赖） |
| Context KV | 不预计算 | 预计算并存储 |
| 适用场景 | 接受率敏感 | 延迟敏感 |

### 接受率 vs 延迟权衡

- EAGLE 的自回归方式使得后续 token 基于前面 token 的预测，接受率更高
- DFlash 的并行方式所有 token 独立预测，接受率较低，但每步 proposal 的延迟更低
- 如果 `k * draft_forward_time >> target_forward_time`，DFlash 更优；反之 EAGLE 更优

:::diagram dflash-vs-eagle-html
```html
<div class="arch-diagram">
  <div class="arch-row">
    <div class="arch-proc">
      <div class="arch-proc-title">EAGLE（自回归）</div>
      <div class="arch-proc-sub">
        <div class="arch-box arch-module">Forward → A</div>
        <div class="arch-box arch-module">Forward → A B</div>
        <div class="arch-box arch-module">Forward → A B C</div>
        <div class="arch-box arch-module">Forward → A B C D</div>
      </div>
      <div class="muted">4 次 forward · 高接受率</div>
    </div>
    <div class="arch-proc">
      <div class="arch-proc-title">DFlash（并行）</div>
      <div class="arch-proc-sub">
        <div class="engine-step">
          <div class="engine-step-content">1 次 Forward → A B C D</div>
        </div>
      </div>
      <div class="muted">1 次 forward · 低延迟</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc dflash-vs-eagle-html
EAGLE 与 DFlash 对比：EAGLE 需要多次 forward 逐步生成 draft token（Forward → A, Forward → A B, ...），接受率高但延迟大。DFlash 只需一次 forward 并行生成所有 draft token（A B C D），延迟低但接受率较低。两种方法的核心权衡是接受率与 proposal 延迟。
:::

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--speculative-method` | None | 设为 `dflash` |
| `--speculative-model` | None | DFlash 模型路径 |
| `--num-speculative-tokens` | None | Draft token 数量 |
