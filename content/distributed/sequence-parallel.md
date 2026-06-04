---
id: sequence-parallel
title: Sequence Parallel
category: distributed
level: advanced
status: draft
readingMinutes: 12
tags:
  - Sequence Parallel
  - All-Reduce
  - All-Gather
codeRefs:
  - vllm/compilation/passes/fusion/sequence_parallelism.py
  - vllm/model_executor/layers/linear.py
heroText: 将 TP 的 [all-reduce](term:将所有 GPU 上的部分和归约到完整结果。) 分解为 reduce_scatter + all_gather，使激活值在序列维度保持切分，为通信-计算融合创造条件。
---

## 心智模型

在标准 TP 中，RowParallelLinear 后的 all-reduce 会在每个 GPU 上产生完整的（replicated）激活张量。SP（Sequence Parallel）的核心思想：让这个张量在序列维度上保持切分状态，每个 GPU 只持有 1/tp_size 的 token，减少显存占用并为通信-计算融合创造条件。

:::diagram sp-mental-model-html
```html
<div class="diagram-container">
  <div class="diagram">
    <div class="diagram-title">SP 核心变换</div>
    <div class="comm-panorama">
      <div class="comm-proc">
        <div class="comm-proc-title">标准 TP（无 SP）</div>
        <div class="comm-proc-body">
          <div class="comm-node">RowParallelLinear</div>
          <div class="comm-node-sub">→ AllReduce →</div>
          <div class="comm-node">RMSNorm</div>
        </div>
        <div class="comm-node-sub">每个 GPU 持有完整激活（replicated）</div>
      </div>
      <div class="comm-proc">
        <div class="comm-proc-title">SP 变换后</div>
        <div class="comm-proc-body">
          <div class="comm-node">RowParallelLinear</div>
          <div class="comm-node-sub">→ ReduceScatter →</div>
          <div class="comm-node">RMSNorm</div>
          <div class="comm-node-sub">→ AllGather →</div>
        </div>
        <div class="comm-label">激活在序列维度保持切分（split）</div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc sp-mental-model-html
SP 的核心变换：将标准 TP 中的 AllReduce 分解为 ReduceScatter + AllGather。

**标准 TP（无 SP）**：RowParallelLinear → AllReduce → RMSNorm。AllReduce 后每个 GPU 持有完整激活张量（replicated），显存占用高。

**SP 变换后**：RowParallelLinear → ReduceScatter → RMSNorm → AllGather。ReduceScatter 将激活沿序列维度切分，每个 GPU 只持有 1/tp_size 的 token。RMSNorm 在切分状态上执行，AllGather 在下一层 ColumnParallelLinear 前拼接完整激活。
:::

## 形态一：编译期 SP

vLLM 通过 `torch.compile` 的 FX graph pattern matching 实现编译期 SP。`SequenceParallelismPass` 在 FX graph 中识别 all-reduce + RMSNorm 模式，替换为 reduce_scatter + RMSNorm + all_gather。

支持三种 pattern：

| Pattern | 原始模式 | SP 变换 |
|---------|---------|--------|
| FirstAllReduceRMSNorm | all_reduce(input) → rms_norm | reduce_scatter(input) → rms_norm → all_gather |
| MiddleAllReduceRMSNorm | all_reduce(mm_out) → fused_add_rms_norm | reduce_scatter(mm_out) → fused_add_rms_norm → all_gather |
| 量化变体 | StaticFP8 / NVFP4 的相同模式 | 对应的量化 reduce_scatter + all_gather |

`vllm/compilation/passes/fusion/sequence_parallelism.py`

## 变换细节

### FirstAllReduceRMSNorm

第一层 Transformer 的 residual 分支：

```
原始模式：
  Input → AllReduce → RMSNorm → Output

SP 变换后：
  Input → ReduceScatter → RMSNorm → AllGather → Output
```

### MiddleAllReduceRMSNorm

中间层的 residual 分支，包含残差加法：

```
原始模式：
  mm_out → AllReduce → FusedAddRmsNorm(residual) → Output

SP 变换后：
  mm_out → ReduceScatter → FusedAddRmsNorm(residual) → AllGather → Output
```

残差 `residual` 在 SP 启用时已经是切分状态（来自上一层的 AllGather 输出被下一层 ReduceScatter 切分），FusedAddRmsNorm 直接在切分状态上执行。

`vllm/compilation/passes/fusion/sequence_parallelism.py`

## SP 在 forward 中的位置

SP 启用后，激活值在 Transformer 层间保持序列维度切分状态。只有在进入 ColumnParallelLinear 前才 all_gather 拼接完整激活，在 RowParallelLinear 后立即 reduce_scatter 切分。

```
Embedding (完整)
  │
  ├─ [Layer 0]
  │    ColumnParallelLinear (gather_output=False, 无通信)
  │    Attention (本地 head)
  │    o_proj = RowParallelLinear
  │         └─ ReduceScatter → RMSNorm → AllGather  ← SP 变换
  │    残差连接（切分状态）
  │    MergedColumnParallelLinear (gather_output=False)
  │    激活函数
  │    down_proj = RowParallelLinear
  │         └─ ReduceScatter → Add+RMSNorm → AllGather  ← SP 变换
  │
  ├─ [Layer 1..N-1] (重复相同模式)
  │
  └─ Output (完整)
```

每层 Transformer 有 2 次 SP 通信对（ReduceScatter + AllGather），分别发生在 o_proj 和 down_proj 之后。与标准 TP 的 2 次 AllReduce 相比，通信量相同但激活显存占用降低为 1/tp_size。

`vllm/model_executor/layers/linear.py`

## 形态二：MoE Sequence Parallel

在 MoE 模型中（DeepSeek-V2、Qwen3-Next 等），`sequence_parallel_chunk` 将 hidden_states 沿 token 维度（dim=0）按 `tp_rank` 切分，每个 rank 处理 `seq_len / tp_size` 个 token。

```python
def sequence_parallel_chunk(hidden_states: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    seq_len = hidden_states.shape[0]
    chunk_size = seq_len // tp_size
    start = tp_rank * chunk_size
    end = start + chunk_size
    return hidden_states[start:end]
```

在 MoE Runner 中，SP 切分与 EP 的 all-to-all 配合使用：先按 TP rank 切分 token，再通过 EP all-to-all 将 token 路由到对应专家所在的 GPU。

`vllm/model_executor/layers/fused_moe/layer.py`

## SP 对 PP 通信的影响

SP 启用时，residual 张量沿序列维度分散在各 TP rank 上（而非 replicated）。PP send/recv 时需通过 `all_gather_tensors` 字典标记 residual 不应使用 all-gather 优化。

```python
all_gather_tensors = {
    "residual": False,
    "hidden_states": True,
}
```

若错误地对 residual 做 all-gather，会将各 TP rank 的 residual 切片合并为完整张量，破坏 SP 的切分语义，导致后续层计算错误。

`vllm/v1/worker/utils.py`

## 启用条件与阈值

SP 不是无条件启用的。`get_sequence_parallelism_threshold()` 根据模型规模和 GPU 架构判断是否启用：

| 条件 | 阈值 | 说明 |
|------|------|------|
| SM90+ (Hopper) | hidden_size ≥ 8192 | H100/H800 等架构 |
| SM100+ (Blackwell) | hidden_size ≥ 8192 | B100/B200 等架构 |
| 其他架构 | hidden_size ≥ 8192 | 较旧架构需更大模型 |

同时计算 `min_token_num`（基于 `min_per_gpu_size_mb` 配置），只有当序列长度足够大时才启用 SP。小模型或短序列场景下，SP 的通信开销可能超过显存节省收益。

禁用方式：设置环境变量 `VLLM_DISABLE_SP=1`。

`vllm/distributed/parallel_state.py`

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| SP 启用 | 自动 | TP>1 且模型足够大时自动启用 |
| `VLLM_DISABLE_SP` | 未设置 | 设为 1 强制禁用 SP |
| `min_per_gpu_size_mb` | 256 | SP 启用的最小每 GPU 激活大小阈值 |
