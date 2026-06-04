---
id: mamba-hybrid
title: "Mamba 混合架构"
category: models
level: advanced
status: draft
readingMinutes: 12
tags:
  - Mamba
  - SSM
  - Hybrid
  - Attention
  - Jamba
codeRefs:
  - vllm/model_executor/models/mamba.py
  - vllm/v1/kv_cache_interface.py
heroText: Mamba SSM 与 Attention 混合架构：[SSM 层](term:状态空间模型层，通过线性递推处理序列，无 KV cache 开销。) 处理局部依赖、Attention 处理全局依赖，Jamba/Bamba 等模型在 vLLM 中的混合 KV cache 管理。
---

## 心智模型

想象你在整理一份超长文档。传统 Attention 就像每次都要从头翻阅整份文档（O(N²) 复杂度），文档越长越慢。Mamba SSM 则像边读边记笔记，只保留关键状态，读到哪里处理到哪里（O(N) 复杂度），但可能漏掉远处的关联。

混合架构的思路：**用 SSM 处理局部细节，用 Attention 捕获全局关联**。就像读书时大部分内容快速扫过（SSM），遇到关键章节停下来仔细回顾前后文（Attention）。

:::diagram mamba-hybrid-mental-model-html
```html
<div class="arch-diagram">
<div class="arch-row">
<div class="arch-box">Input</div>
<div class="arch-arrow">→</div>
<div class="arch-proc">
<div class="arch-proc-title">Mamba SSM</div>
<div class="arch-proc-sub">局部依赖 · O(N)</div>
</div>
<div class="arch-arrow">↔</div>
<div class="arch-proc arch-highlight">
<div class="arch-proc-title">Attention</div>
<div class="arch-proc-sub">全局依赖 · O(N²)</div>
</div>
<div class="arch-arrow">→</div>
<div class="arch-box">Output</div>
</div>
<div class="arch-label">Hybrid: SSM 处理长序列局部信息，Attention 捕获全局关联</div>
</div>
```
:::

:::diagram-desc mamba-hybrid-mental-model-html
混合架构心智模型示意图。输入经过两种层的交替处理：

**Mamba SSM 层**：处理局部依赖，复杂度 O(N)，无 KV cache 开销。就像边读边记笔记，只保留固定大小的状态，适合处理长序列的局部信息。

**Attention 层**：捕获全局依赖，复杂度 O(N²)，需要 KV cache。就像停下来仔细回顾前后文，能建立远距离的关联，但开销随序列长度平方增长。

混合架构的核心思路：大部分层用 SSM 快速处理，少量关键层用 Attention 建立全局关联。这样既保持了长序列处理能力，又控制了计算开销。
:::

## Mamba SSM 原理

### 选择性状态空间模型

传统状态空间模型（SSM）的参数是固定的，对所有输入一视同仁。Mamba 的核心创新是**选择性状态空间**：参数 B、C、Δ 依赖于输入。

```python
h_t = A * h_{t-1} + B(x_t) * x_t
y_t = C(x_t) * h_t
```

其中 A 是状态转移矩阵，B(x_t)、C(x_t)、Δ(x_t) 都由当前输入 x_t 决定。这让模型能**选择性地记住或遗忘信息**——重要的内容保留在状态里，不重要的快速衰减。

### 硬件感知算法

Mamba 的训练效率来自两个关键设计：

**并行扫描（Parallel Scan）**：训练时，序列的所有 token 可以并行处理，而不是像 RNN 那样必须串行。这利用了 GPU 的并行能力。

**递归模式（Recurrent Mode）**：推理时，可以像 RNN 一样逐 token 递推，只维护固定大小的状态，不需要存储所有历史。

### 线性复杂度 vs Attention 的平方复杂度

| 机制 | 复杂度 | 内存占用 | 长序列能力 |
|------|--------|----------|------------|
| Attention | O(N²) | O(N) KV cache | 受限于显存 |
| Mamba SSM | O(N) | O(1) 固定状态 | 理论无限长 |

**关键优势**：SSM 的状态大小是固定的，不随序列长度增长。处理 1K token 和 1M token，状态占用一样多。这让 Mamba 天然适合长上下文场景。

### 无 KV Cache 的代价

SSM 层不需要 KV cache，但代价是**无法随机访问历史**。Attention 可以直接跳到任意位置读取 KV，SSM 必须从头递推到当前位置。这就是为什么纯 SSM 模型在需要"回头看"的任务上表现不佳——混合架构正是为了解决这个问题。

## Mamba + Attention 混合层

### 交替排列策略

混合模型的核心设计是**层类型交替**：大部分层是 SSM，少量层是 Attention。典型模式：

:::diagram mamba-layer-pattern-html
```html
<div class="arch-diagram">
<div class="arch-row">
<div class="arch-proc" style="min-width: 60px;">
<div class="arch-proc-title">L0</div>
<div class="arch-proc-sub">SSM</div>
</div>
<div class="arch-arrow">→</div>
<div class="arch-proc" style="min-width: 60px;">
<div class="arch-proc-title">L1</div>
<div class="arch-proc-sub">SSM</div>
</div>
<div class="arch-arrow">→</div>
<div class="arch-proc" style="min-width: 60px;">
<div class="arch-proc-title">L2</div>
<div class="arch-proc-sub">SSM</div>
</div>
<div class="arch-arrow">→</div>
<div class="arch-proc arch-highlight" style="min-width: 60px;">
<div class="arch-proc-title">L3</div>
<div class="arch-proc-sub">Attn</div>
</div>
<div class="arch-arrow">→</div>
<div class="arch-proc" style="min-width: 60px;">
<div class="arch-proc-title">L4</div>
<div class="arch-proc-sub">SSM</div>
</div>
<div class="arch-arrow">→</div>
<div class="arch-proc" style="min-width: 60px;">
<div class="arch-proc-title">L5</div>
<div class="arch-proc-sub">SSM</div>
</div>
<div class="arch-arrow">→</div>
<div class="arch-proc arch-highlight" style="min-width: 60px;">
<div class="arch-proc-title">L6</div>
<div class="arch-proc-sub">Attn</div>
</div>
</div>
<div class="arch-label">典型模式：每 N 层 SSM 后插入 1 层 Attention</div>
</div>
```
:::

:::diagram-desc mamba-layer-pattern-html
混合层排列模式示意图。展示典型的层类型交替策略：

- L0、L1、L2：SSM 层，处理局部依赖
- L3：Attention 层，建立全局关联
- L4、L5：SSM 层
- L6：Attention 层

典型模式是每 N 层 SSM 后插入 1 层 Attention。Jamba 的设计是每 4-5 层 Mamba 后有 1 层 Attention，部分层还结合 MoE。这样大部分计算是高效的 SSM，少量 Attention 层负责"回头看"建立远距离关联。

不同模型的排列比例不同：Jamba 更激进（SSM 占比高），Bamba 更保守（Attention 占比稍高）。
:::

### 为什么这样设计有效？

**局部依赖占主导**：语言模型的大部分计算是局部上下文（相邻词、短语结构），SSM 能高效处理。

**全局关联稀疏但关键**：长距离依赖（指代消解、篇章结构）虽然稀疏，但对理解至关重要。少量 Attention 层足以捕获这些关联。

**显存效率**：假设 32 层模型，只有 4 层是 Attention，KV cache 只需要纯 Attention 模型的 1/8。

## Hybrid KV Cache 管理

### 问题：不同层需要不同的缓存

混合模型面临一个新挑战：SSM 层不需要 KV cache，Attention 层需要。vLLM 通过 **AttentionSpec** 类型系统来区分：

| AttentionSpec 类型 | 对应层 | KV cache 需求 |
|-------------------|--------|---------------|
| `FullAttentionSpec` | 标准 Attention | 完整 KV cache |
| `MambaSpec` | Mamba SSM | 无 KV cache，固定状态 |
| `SlidingWindowAttentionSpec` | 滑动窗口 Attention | 有限 KV cache |

### HybridKVCacheCoordinator

当模型有多种 attention 类型时，vLLM 使用 `HybridKVCacheCoordinator` 管理：

:::diagram mamba-kv-coordinator-html
```html
<div class="arch-diagram">
<div class="arch-row">
<div class="arch-proc" style="min-width: 180px;">
<div class="arch-proc-title">HybridKVCacheCoordinator</div>
<div class="arch-proc-sub">混合缓存协调器</div>
</div>
<div class="arch-arrow">↓</div>
</div>
<div class="arch-row">
<div class="arch-proc arch-highlight" style="min-width: 120px;">
<div class="arch-proc-title">Group 0</div>
<div class="arch-proc-sub">Full Attention<br>block_size=16</div>
</div>
<div class="arch-proc" style="min-width: 120px;">
<div class="arch-proc-title">Group 1</div>
<div class="arch-proc-sub">Mamba SSM<br>无 KV cache</div>
</div>
<div class="arch-proc" style="min-width: 120px;">
<div class="arch-proc-title">Group 2</div>
<div class="arch-proc-sub">Sliding Window<br>block_size=8</div>
</div>
</div>
</div>
```
:::

:::diagram-desc mamba-kv-coordinator-html
HybridKVCacheCoordinator 管理示意图。混合缓存协调器管理多个 KV cache group，每个 group 对应一种 attention 类型：

**Group 0 - Full Attention**：标准全注意力层，需要完整 KV cache，block_size=16。

**Group 1 - Mamba SSM**：SSM 层，不需要 KV cache，只维护固定大小的状态。这个 group 在 KV cache 管理中是"空"的。

**Group 2 - Sliding Window**：滑动窗口注意力层，需要有限 KV cache（只保留最近 N 个 token），block_size=8。

协调器的核心挑战：不同 group 的 block_size 不同，缓存命中长度必须是所有 block_size 的最小公倍数（LCM）的倍数，哈希粒度是最大公约数（GCD）。
:::

### Block Size 的数学约束

不同 attention 类型可能有不同的 block_size。缓存命中必须满足所有 group 的对齐要求：

```python
cache_hit_length = LCM(block_size_group0, block_size_group1, ...)
hash_granularity = GCD(block_size_group0, block_size_group1, ...)
```

例如：Group 0 的 block_size=16，Group 2 的 block_size=8，则缓存命中长度必须是 16 的倍数（LCM(16,8)=16），哈希粒度是 8（GCD(16,8)=8）。

### SSM 层的状态管理

SSM 层虽然不需要 KV cache，但需要维护状态。vLLM 的处理方式：

**Prefill 阶段**：并行计算所有 token 的状态，最后保留最终状态。

**Decode 阶段**：逐 token 递推更新状态，状态大小固定。

状态存储在模型内部，不通过 KV cache 接口管理。这意味着 SSM 层的"缓存"对调度器透明——调度器只关心 Attention 层的 KV cache。

## 主要混合模型

### Jamba（AI21）

Jamba 是首个大规模 Mamba-Attention-MoE 混合模型：

| 特性 | 设计 |
|------|------|
| 总层数 | 32 层 |
| Mamba 层 | 大部分（约 24 层） |
| Attention 层 | 每 4-5 层插入 1 层（约 6-8 层） |
| MoE | 部分层使用 MoE FFN |
| 上下文长度 | 支持 256K token |

**核心优势**：Mamba 层处理长序列，Attention 层建立全局关联，MoE 降低 FFN 计算量。三者结合实现了长上下文 + 高效率。

### Bamba（IBM）

Bamba 是 IBM 的简化混合架构：

| 特性 | 设计 |
|------|------|
| 架构 | Mamba + Attention（无 MoE） |
| 目标 | 高效长上下文推理 |
| 特点 | 更简单的训练和部署 |

Bamba 去掉了 MoE，专注于 SSM + Attention 的混合，适合需要稳定性和可预测性的生产场景。

### GraniteMoeHybrid（IBM）

IBM 的 MoE + Mamba 混合：

| 特性 | 设计 |
|------|------|
| 架构 | MoE FFN + Mamba SSM |
| 特点 | 稀疏专家路由 + SSM 效率 |
| 适用 | 需要大参数量但低推理成本 |

结合了 MoE 的参数效率和 SSM 的序列效率，适合需要大模型能力但资源受限的场景。

## 关键配置

Mamba 混合模型在 vLLM 中**无需特殊 CLI 标志**，模型配置自动检测混合架构：

| 配置项 | 说明 |
|--------|------|
| 模型 config | 自动声明各层的 attention 类型（FullAttention/Mamba/SlidingWindow） |
| KV cache 布局 | 自动根据 attention 类型分组，HybridKVCacheCoordinator 管理 |
| block_size | 各 group 可不同，协调器自动计算 LCM/GCD |

### 调优建议

**长上下文场景**：优先选择 Mamba 占比高的模型（如 Jamba），SSM 层不占用额外显存。

**需要精确回溯**：选择 Attention 占比稍高的模型，确保足够的全局关联能力。

**显存受限**：混合模型比纯 Attention 模型节省大量 KV cache，可支持更长上下文或更大 batch size。

`vllm/v1/kv_cache_interface.py`
`vllm/model_executor/models/mamba.py`
