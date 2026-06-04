---
id: pipeline-parallel
title: Pipeline Parallel
category: distributed
level: advanced
status: draft
readingMinutes: 12
tags:
  - Pipeline Parallel
  - Microbatch
codeRefs:
  - vllm/v1/engine/core.py
heroText: 模型按层切分到不同 GPU，阶段间通过 NCCL [send/recv](term:点对点通信，一个 GPU 发送张量、另一个接收。) 传递中间张量，batch queue 消除 pipeline bubble。
---

## 心智模型

想象一个工厂的流水线。每个工位（GPU stage）专门负责一部分工作（若干层）。工位 1 处理第 0-5 层，工位 2 处理第 6-11 层，以此类推。半成品（中间张量）通过传送带（send/recv）从一个工位传到下一个。

:::diagram pp-factory
```html
<div class="diagram-container">
  <div class="diagram">
    <div class="diagram-title">流水线工厂类比</div>
    <div class="arch-diagram">
      <div class="arch-proc">
        <div class="arch-proc-title">Stage 0 (GPU 0)</div>
        <div class="arch-box">处理 Layer 0-5</div>
        <div class="muted">输出：中间张量</div>
      </div>
      <div class="arch-arrow">→ send/recv →</div>
      <div class="arch-proc">
        <div class="arch-proc-title">Stage 1 (GPU 1)</div>
        <div class="arch-box">处理 Layer 6-11</div>
        <div class="muted">输出：中间张量</div>
      </div>
      <div class="arch-arrow">→ send/recv →</div>
      <div class="arch-proc">
        <div class="arch-proc-title">Stage 2 (GPU 2)</div>
        <div class="arch-box">处理 Layer 12-17</div>
        <div class="muted">输出：logits</div>
      </div>
    </div>
    <div class="muted">每个阶段专注自己的工作，通过传送带传递中间结果</div>
  </div>
</div>
```
:::

:::diagram-desc pp-factory
Pipeline Parallel 的工厂流水线类比：

**工位分工**：每个 GPU stage 只加载和执行模型的一部分层。Stage 0 处理前几层，Stage 1 处理中间层，最后一个 stage 输出最终结果。

**传送带**：阶段间通过 NCCL send/recv 传递中间张量。发送方将张量发送到下一个 stage，接收方接收后继续计算。

**流水线并行**：多个 batch 可以同时在流水线中流动。当 Stage 1 处理 Batch 0 的第 6-11 层时，Stage 0 可以开始处理 Batch 1 的第 0-5 层。
:::

## 阶段切分

模型层被均匀分配到各个 PP ranks：

:::diagram stage-partition
```html
<div class="diagram-container">
  <div class="diagram">
    <div class="diagram-title">阶段切分</div>
    <div class="arch-diagram">
      <div class="arch-row">
        <div class="arch-module">总层数: 32</div>
        <div class="arch-module">PP size: 4</div>
        <div class="arch-module">每阶段层数: 8</div>
      </div>
      <div class="arch-row">
        <div class="arch-proc">
          <div class="arch-proc-title">PP Rank 0</div>
          <div class="arch-box">Layer 0-7</div>
          <div class="arch-proc-sub">接收 input embeddings</div>
        </div>
        <div class="arch-proc">
          <div class="arch-proc-title">PP Rank 1</div>
          <div class="arch-box">Layer 8-15</div>
          <div class="arch-proc-sub">中间处理</div>
        </div>
        <div class="arch-proc">
          <div class="arch-proc-title">PP Rank 2</div>
          <div class="arch-box">Layer 16-23</div>
          <div class="arch-proc-sub">中间处理</div>
        </div>
        <div class="arch-proc">
          <div class="arch-proc-title">PP Rank 3</div>
          <div class="arch-highlight">Layer 24-31</div>
          <div class="arch-proc-sub">输出 logits</div>
        </div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc stage-partition
阶段切分规则：

**均匀分配**：模型层尽可能均匀分配到各个 PP ranks。如果层数不能整除，前面的 stages 会多分配一层。

**特殊角色**：
- 第一个 stage（PP rank 0）：接收 input embeddings，处理前几层
- 最后一个 stage（PP rank N-1）：输出 logits，进行采样
- 中间 stages：接收前一个 stage 的输出，处理后发送给下一个 stage

**内存节省**：每个 GPU 只需加载部分层的权重，显著减少内存占用。例如 32 层模型在 4 个 GPUs 上，每个 GPU 只需加载 8 层。
:::

## 阶段间通信

阶段间通过 send/recv 传递中间张量：

:::steps id=send-recv-flow
### 1. 发送元数据
发送方通过 cpu_group（Gloo）发送张量的元数据：形状、数据类型、设备信息。

### 2. 发送 GPU 张量
发送方通过 device_group（NCCL）发送实际的 GPU 张量数据。

### 3. 接收元数据
接收方从 cpu_group 接收元数据，根据元数据分配接收缓冲区。

### 4. 接收张量
接收方从 device_group 接收张量数据到预分配的缓冲区。

### 5. 返回结果
接收方返回包含所有接收张量的字典。
:::

:::diagram send-recv-detail
```html
<div class="diagram-container">
  <div class="diagram">
    <div class="diagram-title">send_tensor_dict / recv_tensor_dict</div>
    <div class="comm-panorama">
      <div class="comm-proc">
        <div class="comm-proc-title">Sender (Stage N)</div>
        <div class="comm-proc-body">
          <div class="comm-node">1. 准备张量字典</div>
          <div class="comm-node">2. 发送元数据 (Gloo)</div>
          <div class="comm-node">3. 发送张量 (NCCL)</div>
        </div>
      </div>
      <div class="comm-channel-group">
        <div class="comm-channel-item">
          <div class="comm-arrow">→ meta →</div>
          <div class="comm-label">Gloo</div>
        </div>
        <div class="comm-channel-item">
          <div class="comm-arrow">→ data →</div>
          <div class="comm-label">NCCL</div>
        </div>
      </div>
      <div class="comm-proc">
        <div class="comm-proc-title">Receiver (Stage N+1)</div>
        <div class="comm-proc-body">
          <div class="comm-node">1. 接收元数据</div>
          <div class="comm-node">2. 分配缓冲区</div>
          <div class="comm-node">3. 接收张量</div>
          <div class="comm-node">4. 返回张量字典</div>
        </div>
      </div>
    </div>
    <div class="muted">参考：vllm/distributed/parallel_state.py</div>
  </div>
</div>
```
:::

:::diagram-desc send-recv-detail
阶段间通信的详细流程：

**两阶段通信**：
1. 元数据通信：通过 Gloo（CPU 通信后端）发送张量的形状、类型等元信息
2. 数据通信：通过 NCCL（GPU 通信后端）发送实际的张量数据

**为什么需要元数据**：接收方需要知道即将接收的张量形状和类型，才能正确分配接收缓冲区。

**异步执行**：send/recv 操作是异步的，允许计算和通信重叠。

**参考实现**：`vllm/distributed/parallel_state.py` 中的 `send_tensor_dict` 和 `recv_tensor_dict` 函数。
:::

## 懒等待

AsyncIntermediateTensors 实现了懒等待机制，允许计算和通信重叠：

:::diagram lazy-wait
```html
<div class="diagram-container">
  <div class="diagram">
    <div class="diagram-title">AsyncIntermediateTensors 懒等待</div>
    <div class="cache-flow">
      <div class="cache-step">
        <div class="cache-step-num">1</div>
        <div class="cache-step-content">
          <div class="cache-step-title">Stage N 发送</div>
          <div class="cache-step-desc">send_tensor_dict() → 返回 AsyncIntermediateTensors</div>
        </div>
      </div>
      <div class="cache-step">
        <div class="cache-step-num">2</div>
        <div class="cache-step-content">
          <div class="cache-step-title">Stage N+1 接收</div>
          <div class="cache-step-desc">recv_tensor_dict() → 返回 AsyncIntermediateTensors</div>
        </div>
      </div>
      <div class="cache-step">
        <div class="cache-step-num">3</div>
        <div class="cache-step-content">
          <div class="cache-step-title">继续其他计算</div>
          <div class="cache-step-desc">可以执行不依赖接收张量的计算</div>
        </div>
      </div>
      <div class="cache-step">
        <div class="cache-step-num">4</div>
        <div class="cache-step-content">
          <div class="cache-step-title">首次访问 .tensors</div>
          <div class="cache-step-desc">__getattribute__ 触发 wait_for_comm()</div>
        </div>
      </div>
      <div class="cache-step">
        <div class="cache-step-num">5</div>
        <div class="cache-step-content">
          <div class="cache-step-title">阻塞等待</div>
          <div class="cache-step-desc">通信完成，返回实际张量</div>
        </div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc lazy-wait
懒等待机制的工作原理：

**AsyncIntermediateTensors**：一个包装类，持有通信的 future 对象，但不立即等待。

**懒触发**：重写 `__getattribute__` 方法，当首次访问 `.tensors` 属性时，才调用 `wait_for_comm()` 阻塞等待通信完成。

**计算通信重叠**：在访问 `.tensors` 之前，可以执行其他不依赖接收张量的计算，实现计算和通信的重叠。

**示例**：
```python
async_tensors = recv_tensor_dict()
do_other_work()
tensors = async_tensors.tensors
```
:::

## Batch Queue 消除 Bubble

Pipeline 的主要问题是 bubble：当某个 stage 在等待输入时，其他 stages 可能空闲。Batch queue 通过让多个 batch 同时在流水线中流动来减少 bubble：

:::diagram batch-queue
```html
<div class="diagram-container">
  <div class="diagram">
    <div class="diagram-title">step_with_batch_queue() 流程</div>
    <div class="arch-diagram">
      <div class="arch-row">
        <div class="arch-module">max_concurrent_batches = 4</div>
        <div class="arch-module">batch_queue maxlen = 4</div>
      </div>
      <div class="sched-flow">
        <div class="sched-phase" data-phase="running">
          <div class="sched-phase-title">Time 0</div>
          <div class="sched-phase-steps">
            <div class="sched-step">Batch 0</div>
            <div class="sched-step">idle</div>
            <div class="sched-step">idle</div>
          </div>
        </div>
        <div class="sched-phase" data-phase="running">
          <div class="sched-phase-title">Time 1</div>
          <div class="sched-phase-steps">
            <div class="sched-step">Batch 1</div>
            <div class="sched-step">Batch 0</div>
            <div class="sched-step">idle</div>
          </div>
        </div>
        <div class="sched-phase" data-phase="running">
          <div class="sched-phase-title">Time 2</div>
          <div class="sched-phase-steps">
            <div class="sched-step">Batch 2</div>
            <div class="sched-step">Batch 1</div>
            <div class="sched-step">Batch 0</div>
          </div>
        </div>
        <div class="sched-phase" data-phase="output">
          <div class="sched-phase-title">Time 3</div>
          <div class="sched-phase-steps">
            <div class="sched-step">Batch 3</div>
            <div class="sched-step">Batch 2</div>
            <div class="sched-step">Batch 1</div>
          </div>
        </div>
      </div>
      <div class="muted">多个 batch 同时流动，减少 idle 时间</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc batch-queue
Batch Queue 的工作机制：

**队列结构**：batch_queue 存储 (future, SchedulerOutput, sample_future) 元组，maxlen = max_concurrent_batches。

**填充优先**：优先填充队列（将新 batch 送入流水线），而不是取出结果。这确保流水线尽可能满载。

**流程**：
1. 检查队列是否已满
2. 未满且有新 batch → 送入流水线，添加到队列
3. 队列已满 → 等待最早的 batch 完成，取出结果

**Bubble 减少**：多个 batch 同时在流水线中，当某个 batch 在等待下一个 stage 时，其他 batch 可以继续流动。
:::

## PP + TP 的 all-gather 优化

PP stage 的张量通常在 TP ranks 间复制。为减少通信量，每个 TP rank 只发送自己的分片，接收方通过 all-gather 重构：

:::diagram pp-tp-optimization
```html
<div class="diagram-container">
  <div class="diagram">
    <div class="diagram-title">PP + TP 通信优化</div>
    <div class="comm-layers">
      <div class="comm-layer">
        <div class="comm-layer-title">优化前：每个 TP rank 发送完整张量</div>
        <div class="comm-layer-items">
          <div class="comm-item">TP Rank 0 → 发送完整张量 [B, S, H]</div>
          <div class="comm-item">TP Rank 1 → 发送完整张量 [B, S, H]</div>
          <div class="comm-item">TP Rank 2 → 发送完整张量 [B, S, H]</div>
        </div>
        <div class="comm-layer-note">通信量: 3 × [B, S, H]</div>
      </div>
      <div class="comm-layer">
        <div class="comm-layer-title">优化后：每个 TP rank 发送分片</div>
        <div class="comm-layer-items">
          <div class="comm-item">TP Rank 0 → 发送分片 [B, S, H/3]</div>
          <div class="comm-item">TP Rank 1 → 发送分片 [B, S, H/3]</div>
          <div class="comm-item">TP Rank 2 → 发送分片 [B, S, H/3]</div>
        </div>
        <div class="comm-layer-note">通信量: [B, S, H] + all-gather</div>
      </div>
    </div>
    <div class="muted">SP-enabled 的 residual 不参与 all-gather</div>
  </div>
</div>
```
:::

:::diagram-desc pp-tp-optimization
PP + TP 组合时的通信优化：

**问题**：PP stage 间传递的中间张量通常在所有 TP ranks 上复制。如果每个 TP rank 都发送完整张量，通信量会很大。

**优化方案**：
1. 每个 TP rank 只发送自己的分片
2. 接收方通过 all-gather 重构完整张量
3. 通信量从 N×[B,S,H] 减少到 [B,S,H] + all-gather

**SP 特殊处理**：当启用 Sequence Parallel 时，residual 张量已经在 TP ranks 间切分，不需要 all-gather。通过 `all_gather_tensors` 字典标记哪些张量需要 all-gather。

**参考**：`vllm/distributed/parallel_state.py` 中的 send/recv 实现。
:::

## 关键配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `--pipeline-parallel-size` | PP 大小，流水线阶段数 | 1 |
| `VLLM_PP_LAYER_PARTITION` | 自定义层分配 | 均匀分配 |
| `VLLM_MAX_CONCURRENT_BATCHES` | 最大并发 batch 数 | 自动计算 |
