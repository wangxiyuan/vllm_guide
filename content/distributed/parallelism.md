---
id: parallelism
title: 并行策略总览
category: distributed
level: advanced
status: draft
readingMinutes: 18
tags:
  - TP
  - PP
  - DP
  - EP
  - CP
codeRefs:
  - vllm/distributed/parallel_state.py
  - vllm/model_executor/layers/linear.py
heroText: TP、PP、DP、EP、CP 五种并行策略的统一入口：每种策略切分什么、通信模式、适用场景与组合约束。
---

## 心智模型

想象把一本巨型书籍分配给多个读者阅读。五种并行策略对应不同的分工方式：

**[Tensor Parallel (TP)](term:将权重张量切分到多个 GPU，每个 GPU 计算部分结果后通信合并。)**：每个读者同时阅读不同章节，然后汇总各自的摘要。相当于把矩阵按列/行切分，每个 GPU 计算一部分，通过 all-reduce 合并结果。

**[Pipeline Parallel (PP)](term:将模型按层切分到不同 GPU，形成流水线，阶段间传递中间张量。)**：读者排成流水线，每人处理一个章节后传给下一位。相当于把模型按层切分，每个 GPU 执行若干层，通过 send/recv 传递中间结果。

**[Data Parallel (DP)](term:多个 GPU 各自持有完整模型副本，独立处理不同请求批次。)**：多本相同的书分给不同读者，各自独立阅读。每个 GPU 持有完整模型，独立处理不同请求。

**[Expert Parallel (EP)](term:将 MoE 模型的专家网络切分到不同 GPU，动态路由到对应专家。)**：专业读者各自擅长特定领域，只处理自己领域的内容。MoE 模型的专家网络被切分到不同 GPU，通过路由选择对应专家。

**[Context Parallel (CP)](term:将长序列切分到多个 GPU，每个 GPU 处理一部分上下文。)**：把超长章节拆分给多个读者，每人处理一段。长序列被切分到多个 GPU，通过 all-gather 拼接完整上下文。

:::diagram parallelism-overview
```html
<div class="sched-flow">
  <div class="sched-phase" data-phase="running">
    <div class="sched-phase-title">TP — 权重切分 → all-reduce</div>
    <div class="sched-phase-steps">
      <div class="sched-step">GPU0: W[:, 0:2]</div>
      <div class="sched-step">GPU1: W[:, 2:4]</div>
      <div class="sched-step">GPU2: W[:, 4:6]</div>
      <div class="sched-step">GPU3: W[:, 6:8]</div>
    </div>
  </div>
  <div class="sched-phase" data-phase="waiting">
    <div class="sched-phase-title">PP — 层切分 → send/recv</div>
    <div class="sched-phase-steps">
      <div class="sched-step">GPU0: Layer 0-5</div>
      <div class="sched-step">→ send/recv →</div>
      <div class="sched-step">GPU1: Layer 6-11</div>
      <div class="sched-step">→ send/recv →</div>
      <div class="sched-step">GPU2: Layer 12-17</div>
    </div>
  </div>
  <div class="sched-phase" data-phase="output">
    <div class="sched-phase-title">DP — 独立副本 → 无通信</div>
    <div class="sched-phase-steps">
      <div class="sched-step">GPU0: 完整模型 + Batch0</div>
      <div class="sched-step">GPU1: 完整模型 + Batch1</div>
      <div class="sched-step">GPU2: 完整模型 + Batch2</div>
    </div>
  </div>
  <div class="sched-phase" data-phase="running">
    <div class="sched-phase-title">EP — 专家切分 → all-to-all</div>
    <div class="sched-phase-steps">
      <div class="sched-step">GPU0: Expert 0-3</div>
      <div class="sched-step">GPU1: Expert 4-7</div>
      <div class="sched-step">GPU2: Expert 8-11</div>
    </div>
  </div>
  <div class="sched-phase" data-phase="waiting">
    <div class="sched-phase-title">CP — 序列切分 → all-gather</div>
    <div class="sched-phase-steps">
      <div class="sched-step">GPU0: Seq[0:512]</div>
      <div class="sched-step">GPU1: Seq[512:1024]</div>
      <div class="sched-step">GPU2: Seq[1024:1536]</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc parallelism-overview
五种并行策略的切分方式与通信模式：

**TP (Tensor Parallel)**：权重张量按列或行切分到多个 GPU。每个 GPU 计算部分结果，通过 all-reduce 合并。适用于单层计算密集的场景。

**PP (Pipeline Parallel)**：模型按层切分到不同 GPU，形成流水线。阶段间通过 send/recv 传递中间张量。适用于层数多、单 GPU 内存不足的场景。

**DP (Data Parallel)**：每个 GPU 持有完整模型副本，独立处理不同请求批次。无模型参数通信，仅协调 wave 启停。适用于模型能放入单 GPU 但需要高吞吐的场景。

**EP (Expert Parallel)**：MoE 模型的专家网络被切分到不同 GPU。通过 all-to-all 将 token 路由到对应专家。适用于 MoE 模型。

**CP (Context Parallel)**：长序列被切分到多个 GPU，每个 GPU 处理一部分上下文。通过 all-gather 拼接完整上下文。适用于超长上下文场景。
:::

## Rank 布局

vLLM 使用 5D 张量布局组织所有并行维度：**ExternalDP × DP × PP × PCP × TP**。

:::diagram rank-layout
```html
<div class="kv-lifecycle">
  <div class="kv-lc-state" data-state="used">ExternalDP<br><small>跨节点 DP</small></div>
  <div class="kv-lc-arrow">×</div>
  <div class="kv-lc-state" data-state="cached-used">DP<br><small>节点内 DP</small></div>
  <div class="kv-lc-arrow">×</div>
  <div class="kv-lc-state" data-state="used">PP<br><small>Pipeline 阶段</small></div>
  <div class="kv-lc-arrow">×</div>
  <div class="kv-lc-state" data-state="cached-free">PCP<br><small>Pipeline 内 CP</small></div>
  <div class="kv-lc-arrow">×</div>
  <div class="kv-lc-state" data-state="free">TP<br><small>Tensor Parallel</small></div>
</div>
<div class="cache-flow">
  <div class="cache-step">
    <div class="cache-step-num">1</div>
    <div class="cache-step-content">
      <div class="cache-step-title">reshape(world_size, [edp, dp, pp, pcp, tp])</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">2</div>
    <div class="cache-step-content">
      <div class="cache-step-title">transpose → unbind → 生成各策略的 process groups</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc rank-layout
全局 rank 布局为 5D 张量：ExternalDP × DP × PP × PCP × TP。

**布局构建**：通过 transpose + reshape + unbind 操作，将 1D rank 数组转换为 5D 张量，然后为每种并行策略生成对应的 process group。

**分组逻辑**：
- TP group：同一 PP 阶段、同一 DP rank 内的所有 GPU
- PP group：同一 DP rank 内的不同阶段 GPU
- DP group：同一 PP 阶段、同一 TP rank 的不同 GPU
- EP group：同一 PP 阶段内的 DP×PCP×TP GPU
- CP group：同一 PP 阶段、同一 DP rank 内的 TP GPU 子集

参考：`vllm/distributed/parallel_state.py:1572`
:::

## 五种并行策略概览

| 策略 | 切分对象 | 通信操作 | 典型场景 | 约束 |
|------|---------|---------|---------|------|
| TP | 权重张量（列/行） | all-reduce, all-gather | 单层计算密集、低延迟 | tp_size ≤ 单节点 GPU 数 |
| PP | 模型层 | send/recv | 层数多、内存不足 | pp_size ≤ 层数 |
| DP | 数据批次 | wave 协调 | 模型能放入单 GPU、高吞吐 | dp_size 任意 |
| EP | MoE 专家 | all-to-all | MoE 模型 | ep_size ≤ 专家数 |
| CP | 长序列 | all-gather | 超长上下文 | cp_size ≤ tp_size |

## 策略组合约束

多种并行策略组合时需满足以下约束：

**基础约束**：
```
ExternalDP × DP × PP × PCP × TP = 总 GPU 数
```

**DCP 约束**：DCP（Decode Context Parallel）复用 TP 的 GPU 资源
```
tp_size % dcp_size == 0
```

**EP 约束**：EP 在每个 PP rank 内跨越 DP×PCP×TP
```
ep_size ≤ dp_size × pcp_size × tp_size
```

**SP 约束**：Sequence Parallel 在 TP>1 时自动启用
```
tp_size > 1 → SP 自动启用
```

:::diagram combination-constraints
```html
<div class="comm-layers">
  <div class="comm-layer">
    <div class="comm-layer-title">GPU 总数约束</div>
    <div class="comm-layer-items">
      <div class="comm-item">ExternalDP × DP × PP × PCP × TP = World Size</div>
    </div>
  </div>
  <div class="comm-layer">
    <div class="comm-layer-title">DCP 复用 TP</div>
    <div class="comm-layer-items">
      <div class="comm-item">tp_size % dcp_size == 0</div>
    </div>
    <div class="comm-layer-note">DCP 在 TP group 内切分序列</div>
  </div>
  <div class="comm-layer">
    <div class="comm-layer-title">EP 跨越范围</div>
    <div class="comm-layer-items">
      <div class="comm-item">EP ⊆ DP × PCP × TP (per PP rank)</div>
    </div>
    <div class="comm-layer-note">专家在每个 PP 阶段内分布</div>
  </div>
  <div class="comm-layer">
    <div class="comm-layer-title">SP 自动启用</div>
    <div class="comm-layer-items">
      <div class="comm-item">TP > 1 → SP = True</div>
    </div>
    <div class="comm-layer-note">序列维度随 TP 自动切分</div>
  </div>
</div>
```
:::

:::diagram-desc combination-constraints
并行策略组合的约束关系：

**GPU 总数约束**：所有并行维度的乘积必须等于总 GPU 数。这是基础约束，确保每个 GPU 都被分配到唯一的位置。

**DCP 复用 TP**：DCP（用于长序列的 CP）复用 TP 的 GPU 资源，因此 tp_size 必须能被 dcp_size 整除。例如 tp_size=8 时，dcp_size 可以是 1、2、4、8。

**EP 跨越范围**：EP 在每个 PP rank 内跨越 DP×PCP×TP 的 GPU。专家网络分布在这些 GPU 上，通过 all-to-all 路由 token。

**SP 自动启用**：当 TP>1 时，Sequence Parallel 自动启用。序列维度随 TP 切分，减少每个 GPU 的激活内存。
:::

## 选择策略的决策树

:::steps id=strategy-decision
### 1. 单 GPU 场景
模型能放入单 GPU 内存？
- **是** → 无需并行，直接推理
- **否** → 进入步骤 2

### 2. 模型内存不足
单 GPU 内存不足？
- **是** → 使用 TP 切分权重（优先）
- **否** → 进入步骤 3

### 3. TP 达到上限
TP 已达单节点 GPU 上限，内存仍不足？
- **是** → 添加 PP 切分层
- **否** → 进入步骤 4

### 4. 需要更长上下文
上下文长度超过单 GPU 容量？
- **是** → 添加 CP 切分序列
- **否** → 进入步骤 5

### 5. MoE 模型
模型是 MoE 架构？
- **是** → 添加 EP 切分专家
- **否** → 进入步骤 6

### 6. 提升吞吐
需要更高吞吐？
- **是** → 添加 DP 复制引擎
- **否** → 当前配置已足够
:::

## 关键配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `--tensor-parallel-size` | TP 大小，权重切分 GPU 数 | 1 |
| `--pipeline-parallel-size` | PP 大小，流水线阶段数 | 1 |
| `--data-parallel-size` | DP 大小，引擎副本数 | 1 |
| `--dcp-size` | DCP 大小，序列切分数 | 1 |
| `--enable-eplb` | 启用 EP 负载均衡 | False |
| `--enable-expert-parallel` | 启用 EP | False |
| `VLLM_ALL2ALL_BACKEND` | EP all-to-all 后端 | "naive" |
