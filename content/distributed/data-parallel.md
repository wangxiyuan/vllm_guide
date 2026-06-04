---
id: data-parallel
title: Data Parallel
category: distributed
level: intermediate
status: draft
readingMinutes: 12
tags:
  - Data Parallel
  - DP Attention
  - Load Balance
codeRefs:
  - vllm/v1/engine/coordinator.py
  - vllm/distributed/utils.py
heroText: 多个 EngineCore 独立调度、独立推理，通过 [DPCoordinator](term:协调多个 DP rank 的 wave 启停和负载均衡的进程。) 协调 wave 启停与负载均衡。
---

## 心智模型

想象一个超市有多个独立的收银台。每个收银台（DP rank）独立处理自己的顾客（请求），互不干扰。但需要一个协调员（DPCoordinator）确保所有收银台同时开门/关门（wave 协调），并将顾客分配到最短的队伍（负载均衡）。

:::diagram dp-supermarket
```html
<div class="diagram-container">
  <div class="diagram">
    <div class="diagram-title">DPCoordinator — 协调员：管理 wave 启停、负载均衡</div>
    <div class="sched-mental-model">
      <div class="sched-queue-group">
        <div class="sched-queue" data-queue="running">
          <div class="sched-queue-title">DP Rank 0</div>
          <div class="sched-queue-desc">顾客队列: [A, B, C]</div>
        </div>
        <div class="sched-queue" data-queue="running">
          <div class="sched-queue-title">DP Rank 1</div>
          <div class="sched-queue-desc">顾客队列: [D, E]</div>
        </div>
        <div class="sched-queue" data-queue="running">
          <div class="sched-queue-title">DP Rank 2</div>
          <div class="sched-queue-desc">顾客队列: [F]</div>
        </div>
      </div>
    </div>
    <div class="arch-flow-label">每个收银台独立工作，协调员统一管理启停</div>
  </div>
</div>
```
:::

:::diagram-desc dp-supermarket
Data Parallel 的超市收银台类比：

**独立收银台**：每个 DP rank 是一个独立的 EngineCore，拥有自己的 Scheduler、KV Cache、Worker。它们独立调度和执行请求，互不干扰。

**协调员**：DPCoordinator 进程负责协调所有 DP ranks 的 wave 启停。当所有 ranks 都处理完当前请求时，协调员暂停所有引擎；新请求到达时，协调员广播恢复信号。

**负载均衡**：协调员收集每个 rank 的队列长度，API 服务器根据这些信息将新请求路由到负载最轻的 rank。
:::

## DP 架构

每个 DP rank 拥有完整的推理引擎组件：

:::diagram dp-architecture
```html
<div class="diagram-container">
  <div class="diagram">
    <div class="diagram-title">DP 架构</div>
    <div class="comm-panorama">
      <div class="comm-proc">
        <div class="comm-proc-title">DPCoordinator 进程</div>
        <div class="comm-proc-body">
          <div class="comm-node">XPUB Socket (广播控制消息)</div>
          <div class="comm-node">XSUB Socket (接收状态)</div>
        </div>
      </div>
      <div class="comm-channel-group">
        <div class="comm-channel-item">
          <div class="comm-arrow">↔</div>
          <div class="comm-label">pub-sub</div>
        </div>
      </div>
      <div class="comm-proc">
        <div class="comm-proc-title">DP Rank 0</div>
        <div class="comm-proc-body">
          <div class="comm-node">EngineCore</div>
          <div class="comm-node">Scheduler</div>
          <div class="comm-node">KV Cache Manager</div>
          <div class="comm-node">Worker + ModelRunner</div>
        </div>
      </div>
      <div class="comm-proc">
        <div class="comm-proc-title">DP Rank 1</div>
        <div class="comm-proc-body">
          <div class="comm-node">EngineCore</div>
          <div class="comm-node">Scheduler</div>
          <div class="comm-node">KV Cache Manager</div>
          <div class="comm-node">Worker + ModelRunner</div>
        </div>
      </div>
      <div class="comm-proc">
        <div class="comm-proc-title">DP Rank 2</div>
        <div class="comm-proc-body">
          <div class="comm-node">EngineCore</div>
          <div class="comm-node">Scheduler</div>
          <div class="comm-node">KV Cache Manager</div>
          <div class="comm-node">Worker + ModelRunner</div>
        </div>
      </div>
    </div>
    <div class="arch-flow-label">每个 DP rank 独立调度、独立推理</div>
  </div>
</div>
```
:::

:::diagram-desc dp-architecture
Data Parallel 的架构组成：

**DPCoordinator 进程**：
- 独立进程，管理所有 DP ranks
- 使用 XPUB/XSUB pub-sub 模式广播控制消息
- 收集各 rank 的状态信息

**DP Rank 组件**：
- EngineCore：核心推理引擎
- Scheduler：请求调度器
- KV Cache Manager：KV 缓存管理
- Worker + ModelRunner：模型执行器

**独立性**：每个 DP rank 拥有完整的推理组件，可以独立处理请求。DP ranks 之间不共享状态，仅通过 DPCoordinator 协调。
:::

## Wave 协调

Wave 是 DP 的基本调度单位。所有 DP ranks 必须同步启停：

:::steps id=wave-coordination
### 1. 正常运行
每个 EngineCore 独立运行 step() 循环，处理各自的请求队列。

### 2. 队列清空
某个 DP rank 的请求队列变为空，该 rank 发送 `wave_complete` 消息给 DPCoordinator。

### 3. 全局暂停
DPCoordinator 收到所有 ranks 的 `wave_complete` 后，广播 `PAUSE_DP_WAVE`，所有引擎暂停。

### 4. 新请求到达
API 服务器接收新请求，路由到某个 DP rank。该 rank 通知 DPCoordinator。

### 5. 全局恢复
DPCoordinator 广播 `START_DP_WAVE`，所有 DP ranks 恢复运行，开始新的 wave。
:::

:::diagram wave-flow
```html
<div class="diagram-container">
  <div class="diagram">
    <div class="diagram-title">Wave 协调流程</div>
    <div class="sched-flow">
      <div class="sched-phase" data-phase="running">
        <div class="sched-phase-title">Phase 1: 运行</div>
        <div class="sched-phase-steps">
          <div class="sched-step">Rank 0: 处理中</div>
          <div class="sched-step">Rank 1: 处理中</div>
          <div class="sched-step">Rank 2: 处理中</div>
        </div>
      </div>
      <div class="sched-arrow-down">
        <div class="sched-arrow-icon">↓</div>
      </div>
      <div class="sched-phase" data-phase="waiting">
        <div class="sched-phase-title">Phase 2: 队列清空</div>
        <div class="sched-phase-steps">
          <div class="sched-step">Rank 0: 空 ✓</div>
          <div class="sched-step">Rank 1: 处理中</div>
          <div class="sched-step">Rank 2: 处理中</div>
        </div>
      </div>
      <div class="sched-arrow-down">
        <div class="sched-arrow-icon">↓</div>
      </div>
      <div class="sched-phase" data-phase="output">
        <div class="sched-phase-title">Phase 3: 全部清空</div>
        <div class="sched-phase-steps">
          <div class="sched-step">Rank 0: 空 ✓</div>
          <div class="sched-step">Rank 1: 空 ✓</div>
          <div class="sched-step">Rank 2: 空 ✓</div>
          <div class="sched-step">→ DPCoordinator 广播 PAUSE</div>
        </div>
      </div>
      <div class="sched-arrow-down">
        <div class="sched-arrow-icon">↓</div>
      </div>
      <div class="sched-phase" data-phase="waiting">
        <div class="sched-phase-title">Phase 4: 新请求到达</div>
        <div class="sched-phase-steps">
          <div class="sched-step">Rank 0: 暂停</div>
          <div class="sched-step">Rank 1: 暂停</div>
          <div class="sched-step">Rank 2: 暂停</div>
          <div class="sched-step">→ 新请求 → DPCoordinator 广播 START</div>
        </div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc wave-flow
Wave 协调的完整流程：

**Wave 定义**：一个 wave 是所有 DP ranks 从开始处理请求到全部队列清空的时间段。

**同步机制**：
1. 每个 rank 独立运行，直到队列为空
2. 空队列的 rank 发送完成信号
3. DPCoordinator 等待所有 ranks 完成
4. 全部完成后广播暂停信号
5. 新请求到达时广播恢复信号

**为什么需要同步**：CUDA Graph 要求所有 ranks 执行相同的操作序列。Wave 同步确保所有 ranks 在相同的时机启用/禁用 CUDA Graph。
:::

## 负载均衡

DPCoordinator 收集各 DP engine 的负载信息，API 服务器据此路由请求：

:::diagram load-balance
```html
<div class="diagram-container">
  <div class="diagram">
    <div class="diagram-title">负载均衡</div>
    <div class="cache-flow">
      <div class="cache-step">
        <div class="cache-step-num">1</div>
        <div class="cache-step-content">
          <div class="cache-step-title">DPCoordinator 收集统计</div>
          <div class="cache-step-desc">Rank 0: waiting: 5, running: 10</div>
          <div class="cache-step-desc">Rank 1: waiting: 2, running: 8</div>
          <div class="cache-step-desc">Rank 2: waiting: 0, running: 3</div>
        </div>
      </div>
      <div class="cache-step">
        <div class="cache-step-num">2</div>
        <div class="cache-step-content">
          <div class="cache-step-title">发布到 API 服务器</div>
          <div class="cache-step-desc">通过 ZMQ pub-sub 发布负载统计</div>
        </div>
      </div>
      <div class="cache-step">
        <div class="cache-step-num">3</div>
        <div class="cache-step-content">
          <div class="cache-step-title">API 服务器路由决策</div>
          <div class="cache-step-desc">新请求 → 路由到 Rank 2（负载最轻）</div>
        </div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc load-balance
负载均衡机制：

**统计收集**：每个 DP engine 定期向 DPCoordinator 报告：
- waiting 队列长度（等待调度的请求数）
- running 队列长度（正在执行的请求数）

**信息发布**：DPCoordinator 汇总统计信息，通过 ZMQ pub-sub 发布给所有 API 服务器实例。

**路由决策**：API 服务器收到新请求时，查询最新的负载统计，将请求路由到负载最轻的 DP rank。

**动态调整**：负载统计实时更新，路由决策动态适应各 rank 的负载变化。
:::

## DP 同步

虽然 DP ranks 独立运行，但某些决策需要同步：

:::diagram dp-sync
```html
<div class="diagram-container">
  <div class="diagram">
    <div class="diagram-title">_synchronize_dp_ranks() 同步内容</div>
    <div class="comm-layers">
      <div class="comm-layer">
        <div class="comm-layer-title">同步字段</div>
        <div class="comm-layer-items">
          <div class="comm-item">orig_tokens — 原始 token 数量</div>
          <div class="comm-item">padded_tokens — 填充后的 token 数量</div>
          <div class="comm-item">should_ubatch — 是否启用 microbatch</div>
          <div class="comm-item">cudagraph_mode — CUDA Graph 模式</div>
        </div>
      </div>
      <div class="comm-layer">
        <div class="comm-layer-title">同步方法</div>
        <div class="comm-layer-items">
          <div class="comm-item">1. all-reduce 收集所有 ranks 的值</div>
          <div class="comm-item">2. 取最大值作为统一标准</div>
          <div class="comm-item">3. 所有 ranks padding 到最大值</div>
        </div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc dp-sync
DP ranks 的同步机制：

**同步内容**：
- orig_tokens：每个 rank 实际处理的 token 数
- padded_tokens：padding 后的 token 数
- should_ubatch：是否启用 microbatch（用于 chunked prefill）
- cudagraph_mode：CUDA Graph 模式（none/small/full）

**同步方法**：
1. 使用 all-reduce 收集所有 ranks 的值
2. 取最大值作为统一标准
3. 所有 ranks padding 到最大值

**为什么需要同步**：CUDA Graph 要求所有 ranks 执行相同的操作。如果某个 rank 的 token 数较少，需要 padding 到与其他 ranks 相同，才能使用相同的 CUDA Graph。
:::

## DP 与其他并行的组合

DP 是 rank 布局的最外层维度：

:::diagram dp-combination
```html
<div class="diagram-container">
  <div class="diagram">
    <div class="diagram-title">Rank 布局：ExternalDP × DP × PP × PCP × TP</div>
    <div class="kv-lifecycle">
      <div class="kv-lc-state" data-state="cached-used">ExternalDP<br><small>跨节点 DP</small></div>
      <div class="kv-lc-arrow">×</div>
      <div class="kv-lc-state" data-state="used">DP<br><small>节点内 DP</small></div>
      <div class="kv-lc-arrow">×</div>
      <div class="kv-lc-state" data-state="cached-free">PP<br><small>Pipeline 阶段</small></div>
      <div class="kv-lc-arrow">×</div>
      <div class="kv-lc-state" data-state="free">PCP<br><small>Pipeline 内 CP</small></div>
      <div class="kv-lc-arrow">×</div>
      <div class="kv-lc-state" data-state="free">TP<br><small>Tensor Parallel</small></div>
    </div>
    <div class="arch-flow-label">DP ranks 独立运行，仅通过 DPCoordinator 协调</div>
  </div>
</div>
```
:::

:::diagram-desc dp-combination
DP 与其他并行策略的组合关系：

**布局位置**：DP 是 rank 布局的最外层维度（除了 ExternalDP）。每个 DP rank 内部可以包含完整的 PP×PCP×TP 布局。

**独立性**：DP ranks 之间不共享模型参数、KV Cache 或调度状态。每个 DP rank 是一个完整的推理引擎。

**协调点**：
- Wave 启停：通过 DPCoordinator 同步
- CUDA Graph：所有 DP ranks 使用相同的 CUDA Graph 配置
- 负载均衡：通过 DPCoordinator 收集和发布负载信息

**组合示例**：8 GPUs，dp_size=2, tp_size=4 → 2 个 DP ranks，每个 rank 内 4 个 TP GPUs。
:::

## 关键配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `--data-parallel-size` | DP 大小，引擎副本数 | 1 |
| `--data-parallel-rank-local` | 是否使用本地 rank | False |
| `VLLM_DP_SIZE` | 环境变量设置 DP 大小 | - |
| `VLLM_DP_MASTER_IP` | DPCoordinator IP | 自动检测 |
| `VLLM_DP_MASTER_PORT` | DPCoordinator 端口 | 自动分配 |
