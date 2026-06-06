---
id: tensor-parallel
title: Tensor Parallel
category: distributed
level: advanced
status: draft
readingMinutes: 12
tags:
  - Tensor Parallel
  - All Reduce
codeRefs:
  - vllm/model_executor/layers/linear.py
heroText: 权重沿 output/input 维度切分到多个 GPU，forward 中通过 [all-reduce](term:将所有 GPU 上的部分和归约到完整结果。) / [all-gather](term:将所有 GPU 上的分片拼接为完整张量。) 协作完成计算。
---

## 心智模型

想象一个巨大的矩阵乘法 **Y = XA**，其中 A 是权重矩阵。TP 的核心思想是把 A 切分成多块，每个 GPU 计算一部分，然后通过通信合并结果。

**Column Parallel**：把 A 按列切分，每个 GPU 持有若干列。计算后得到部分输出，需要 all-gather 拼接完整输出。

**Row Parallel**：把 A 按行切分，每个 GPU 持有若干行。计算后得到部分和，需要 all-reduce 求和得到完整输出。

:::diagram tp-concept
```html
<div class="arch-diagram">
  <div class="arch-proc">
    <div class="arch-proc-title">Column Parallel</div>
    <div class="arch-proc-sub">按列切分权重</div>
    <div class="arch-row">
      <div class="arch-box">X [B, K]</div>
      <div class="arch-box">×</div>
    </div>
    <div class="arch-row">
      <div class="arch-box arch-highlight">A[:, 0:2]</div>
      <div class="arch-box arch-highlight">A[:, 2:4]</div>
      <div class="arch-box arch-highlight">A[:, 4:6]</div>
    </div>
    <div class="arch-row">
      <div class="arch-box">=</div>
    </div>
    <div class="arch-row">
      <div class="arch-box arch-module">Y[:, 0:2]</div>
      <div class="arch-box arch-module">Y[:, 2:4]</div>
      <div class="arch-box arch-module">Y[:, 4:6]</div>
    </div>
    <div class="arch-flow-label">all-gather → 完整 Y</div>
  </div>
  <div class="arch-proc">
    <div class="arch-proc-title">Row Parallel</div>
    <div class="arch-proc-sub">按行切分权重</div>
    <div class="arch-row">
      <div class="arch-box">X [B, K]</div>
      <div class="arch-box">×</div>
    </div>
    <div class="arch-row">
      <div class="arch-box arch-highlight">A[0:2, :]</div>
      <div class="arch-box arch-highlight">A[2:4, :]</div>
      <div class="arch-box arch-highlight">A[4:6, :]</div>
    </div>
    <div class="arch-row">
      <div class="arch-box">=</div>
    </div>
    <div class="arch-row">
      <div class="arch-box arch-module">Y_partial[0]</div>
      <div class="arch-box arch-module">Y_partial[1]</div>
      <div class="arch-box arch-module">Y_partial[2]</div>
    </div>
    <div class="arch-flow-label">all-reduce → 完整 Y</div>
  </div>
</div>
```
:::

:::diagram-desc tp-concept
Tensor Parallel 的两种切分方式：

**Column Parallel**：权重矩阵按列切分，每个 GPU 持有若干列。输入 X 在所有 GPU 上复制，每个 GPU 计算部分输出列。通过 all-gather 将所有 GPU 的输出列拼接为完整输出。

**Row Parallel**：权重矩阵按行切分，每个 GPU 持有若干行。输入 X 也按对应维度切分，每个 GPU 计算部分和。通过 all-reduce 将所有 GPU 的部分和归约为完整输出。

关键区别：Column Parallel 输出需要拼接（all-gather），Row Parallel 输出需要求和（all-reduce）。
:::

## 线性层切分方式

vLLM 提供四种 TP 线性层，对应不同的切分策略：

| 层类型 | 切分维度 | Forward 通信 | 典型用途 |
|--------|---------|-------------|---------|
| ColumnParallelLinear | output 维度 | gather_output=True 时 all-gather | QKV 投影、up_proj |
| RowParallelLinear | input 维度 | all-reduce | o_proj、down_proj |
| MergedColumnParallelLinear | output 维度 | gather_output=True 时 all-gather | gate_up_proj（合并 gate 和 up） |
| QKVParallelLinear | output 维度 | 无（默认 gather_output=False） | QKV 投影（处理 GQA） |

:::diagram linear-layers
```html
<div class="comm-layers">
  <div class="comm-layer">
    <div class="comm-layer-title">ColumnParallelLinear</div>
    <div class="comm-layer-items">
      <div class="comm-item">W[:, 0:N/tp]</div>
      <div class="comm-item">X @ W_shard → Y_shard</div>
      <div class="comm-item">gather_output=True 时 all-gather</div>
    </div>
  </div>
  <div class="comm-layer">
    <div class="comm-layer-title">RowParallelLinear</div>
    <div class="comm-layer-items">
      <div class="comm-item">W[0:N/tp, :]</div>
      <div class="comm-item">X_shard @ W_shard → Y_partial</div>
      <div class="comm-item">all-reduce</div>
    </div>
  </div>
  <div class="comm-layer">
    <div class="comm-layer-title">MergedColumnParallelLinear</div>
    <div class="comm-layer-items">
      <div class="comm-item">[W_gate, W_up][:, 0:N/tp]</div>
      <div class="comm-item">X @ [W_gate, W_up]_shard</div>
      <div class="comm-item">gather_output=True 时 all-gather</div>
    </div>
  </div>
  <div class="comm-layer">
    <div class="comm-layer-title">QKVParallelLinear</div>
    <div class="comm-layer-items">
      <div class="comm-item">[W_q, W_k, W_v][:, 0:N/tp]</div>
      <div class="comm-item">X @ [W_q, W_k, W_v]_shard</div>
      <div class="comm-item">无通信</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc linear-layers
四种 TP 线性层的切分方式：

**ColumnParallelLinear**：权重按 output 维度切分。每个 GPU 持有部分输出通道。`gather_output=True` 时 all-gather 拼接完整输出，否则输出保持切分状态。

**RowParallelLinear**：权重按 input 维度切分。每个 GPU 持有部分输入通道。`reduce_results=True` 时 all-reduce 合并部分和。

**MergedColumnParallelLinear**：将两个 Column Parallel 层合并（如 gate_proj 和 up_proj）。权重拼接后按 output 维度切分。`gather_output=True` 时 all-gather。

**QKVParallelLinear**：专门处理 QKV 投影，支持 GQA 的 KV head 复制。权重按 output 维度切分，自动处理 head 复制逻辑。默认 `gather_output=False`，无通信。
:::

## 典型 Transformer 层的 TP 通信

一个 Transformer 层包含两次 all-reduce：

:::diagram transformer-tp-flow
```html
<div class="sched-flow">
  <div class="sched-phase" data-phase="running">
    <div class="sched-phase-title">Attention</div>
    <div class="sched-phase-steps">
      <div class="sched-step">Input [B, S, H]</div>
      <div class="sched-step">QKVParallelLinear <span class="muted">（无通信）</span></div>
      <div class="sched-step">Attention <span class="muted">（无 TP 通信）</span></div>
    </div>
  </div>
  <div class="sched-phase" data-phase="output">
    <div class="sched-phase-title">all-reduce #1</div>
    <div class="sched-phase-steps">
      <div class="sched-step">o_proj (RowParallel)</div>
    </div>
  </div>
  <div class="sched-phase" data-phase="waiting">
    <div class="sched-phase-title">Residual + FFN</div>
    <div class="sched-phase-steps">
      <div class="sched-step">residual + norm</div>
      <div class="sched-step">MergedColumnParallel <span class="muted">（无通信）</span></div>
      <div class="sched-step">activation (SiLU/SwiGLU)</div>
    </div>
  </div>
  <div class="sched-phase" data-phase="output">
    <div class="sched-phase-title">all-reduce #2</div>
    <div class="sched-phase-steps">
      <div class="sched-step">down_proj (RowParallel)</div>
    </div>
  </div>
  <div class="sched-phase" data-phase="running">
    <div class="sched-phase-title">Output</div>
    <div class="sched-phase-steps">
      <div class="sched-step">residual + norm</div>
      <div class="sched-step">Output [B, S, H]</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc transformer-tp-flow
典型 Transformer 层的 TP 通信流程：

**Attention 部分**：
1. QKVParallelLinear：无通信，输出已切分
2. Attention 计算：每个 GPU 独立计算部分 head，无 TP 通信
3. o_proj (RowParallel)：all-reduce 合并 attention 输出

**FFN 部分**：
1. MergedColumnParallelLinear：无通信，gate 和 up 投影合并
2. activation：每个 GPU 独立计算
3. down_proj (RowParallel)：all-reduce 合并 FFN 输出

**通信总结**：每个 Transformer 层有 2 次 all-reduce，分别在 attention 和 FFN 的输出位置。
:::

## GQA 的 KV 复制

当 tp_size >= num_kv_heads 时，KV heads 会在 TP ranks 间复制：

:::diagram gqa-kv-replica
```html
<div class="kv-mental-model">
  <div class="kv-data-structures">
    <div class="kv-ds">
      <div class="kv-ds-title">Config</div>
      <div class="kv-ds-items">
        <div class="kv-ds-item">num_q_heads = 32</div>
        <div class="kv-ds-item">num_kv_heads = 8</div>
        <div class="kv-ds-item">tp_size = 4</div>
      </div>
    </div>
  </div>
  <div class="kv-data-structures">
    <div class="kv-ds">
      <div class="kv-ds-title">TP Rank 0</div>
      <div class="kv-ds-items">
        <div class="kv-ds-item">Q: 0-7, KV: 0-1 (replica 0)</div>
      </div>
    </div>
    <div class="kv-ds">
      <div class="kv-ds-title">TP Rank 1</div>
      <div class="kv-ds-items">
        <div class="kv-ds-item">Q: 8-15, KV: 0-1 (replica 1)</div>
      </div>
    </div>
    <div class="kv-ds">
      <div class="kv-ds-title">TP Rank 2</div>
      <div class="kv-ds-items">
        <div class="kv-ds-item">Q: 16-23, KV: 2-3 (replica 0)</div>
      </div>
    </div>
    <div class="kv-ds">
      <div class="kv-ds-title">TP Rank 3</div>
      <div class="kv-ds-items">
        <div class="kv-ds-item">Q: 24-31, KV: 2-3 (replica 1)</div>
      </div>
    </div>
  </div>
  <div class="kv-ds">
    <div class="kv-ds-title">Formula</div>
    <div class="kv-ds-items">
      <div class="kv-ds-item">replica_id = tp_rank // num_kv_head_replicas</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc gqa-kv-replica
GQA (Grouped Query Attention) 的 KV head 复制机制：

**复制原因**：当 tp_size > num_kv_heads 时，每个 KV head 需要在多个 TP ranks 上复制，以匹配 Q heads 的分布。

**复制公式**：
```
num_kv_head_replicas = tp_size // (num_q_heads // num_kv_heads)
replica_id = tp_rank // num_kv_head_replicas
```

**QKVParallelLinear 处理**：自动检测 tp_size 和 num_kv_heads 的关系，正确切分 Q weights 并复制 KV weights。

**示例**：num_q_heads=32, num_kv_heads=8, tp_size=4 时，每个 TP rank 处理 8 个 Q heads，KV heads 被复制到 2 个 ranks。
:::

## all-reduce 调用链

RowParallelLinear 的 all-reduce 调用链：

:::diagram allreduce-chain
```html
<div class="kv-hash-chain">
  <div class="kv-hash-block">
    <div class="kv-hash-block-title">RowParallelLinear.forward()</div>
    <div class="kv-hash-block-detail">vllm/model_executor/layers/linear.py</div>
  </div>
  <div class="kv-hash-arrow">→</div>
  <div class="kv-hash-block">
    <div class="kv-hash-block-title">tensor_model_parallel_all_reduce()</div>
    <div class="kv-hash-block-detail">vllm/distributed/communication_op.py</div>
  </div>
  <div class="kv-hash-arrow">→</div>
  <div class="kv-hash-block">
    <div class="kv-hash-block-title">CudaCommunicator.all_reduce()</div>
    <div class="kv-hash-block-detail">vllm/distributed/device_communicators/cuda_communicator.py</div>
  </div>
  <div class="kv-hash-arrow">→</div>
  <div class="kv-hash-block">
    <div class="kv-hash-block-title">Backend: NCCL / CUSTOM / FLASHINFER / SYMM_MEM / PYNCCL</div>
    <div class="kv-hash-block-detail">根据配置选择</div>
  </div>
</div>
```
:::

:::diagram-desc allreduce-chain
all-reduce 从线性层到底层通信库的完整调用链：

**调用流程**：
1. RowParallelLinear.forward() 检测到需要 all-reduce
2. 调用 tensor_model_parallel_all_reduce() 工具函数（`vllm/distributed/communication_op.py`）
3. 获取 TP group 的 GroupCoordinator
4. 调用注册的 custom op: torch.ops.vllm.all_reduce()
5. CudaCommunicator 根据配置选择后端
6. 执行底层 NCCL 或其他通信库

**后端选择**：CudaCommunicator 根据配置和 GPU 能力自动选择后端，支持 NCCL、Custom All-Reduce（对称内存）、FlashInfer、Symmetric Memory、PyNCCL 等。
:::

## TP 与 CUDA Graph

TP 的集合通信操作需要特殊处理才能支持 CUDA Graph：

:::diagram tp-cudagraph
```html
<div class="engine-step-flow">
  <div class="engine-step">
    <div class="engine-step-num">1</div>
    <div class="engine-step-content">
      <div class="engine-step-title">注册 Custom Op</div>
      <div class="engine-step-desc">all-reduce/all-gather 注册为 torch custom op</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">2</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Graph Capture</div>
      <div class="engine-step-desc">GroupCoordinator.graph_capture() 上下文管理器</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">3</div>
    <div class="engine-step-content">
      <div class="engine-step-title">记录通信模式</div>
      <div class="engine-step-desc">记录每个通信操作的输入输出形状</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">4</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Replay</div>
      <div class="engine-step-desc">CUDA Graph replay 时自动执行通信</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc tp-cudagraph
TP 与 CUDA Graph 的兼容机制：

**问题**：CUDA Graph 需要记录固定的 GPU 操作序列，但 NCCL 集合通信是动态的，无法直接 capture。

**解决方案**：
1. 将 all-reduce/all-gather 注册为 PyTorch custom op
2. 使用 GroupCoordinator.graph_capture() 上下文管理器
3. 在 capture 阶段记录通信操作的元数据
4. Replay 时根据记录的元数据执行通信

**关键代码**：`vllm/distributed/parallel_state.py` 中的 GroupCoordinator 类。
:::

## 关键配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `--tensor-parallel-size` | TP 大小，权重切分 GPU 数 | 1 |
| `VLLM_ALL2ALL_BACKEND` | all-to-all 后端（EP 用） | "naive" |
| `VLLM_USE_V1` | 使用 V1 架构（影响 TP 实现） | 1 |
| `VLLM_ATTENTION_BACKEND` | Attention 后端（影响 TP 通信） | 自动选择 |
