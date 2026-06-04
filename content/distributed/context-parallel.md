---
id: context-parallel
title: Context Parallel
category: distributed
level: advanced
status: draft
readingMinutes: 14
tags:
  - Context Parallel
  - DCP
  - PCP
  - All-to-All
codeRefs:
  - vllm/v1/attention/ops/dcp_alltoall.py
  - vllm/v1/worker/gpu_model_runner.py
heroText: DCP 复用 TP GPU 扩展 decode 序列长度，PCP 独立维度加速 MoE prefill，[LSE 校正](term:各 rank 只看到部分 KV，softmax 分母不同，需用 log-sum-exp 值做数值校正后合并。) 合并各 rank 的 attention 输出。
---

## 心智模型

想象你在阅读一份很长的文档。一个人读不完，于是把文档拆成几段，每人读一段。但回答关于整篇文档的问题时，每个人都需要知道其他人读了什么——这就是 Context Parallel 的核心矛盾。

DCP（Decode Context Parallel）把 KV cache（"已读内容的记忆"）切分到多个 GPU 上，每个 DCP rank 只存一部分上下文，但 query 在所有 rank 上完整，通过 LSE 校正合并各 rank 的 attention 输出。PCP（Prefill Context Parallel）则把输入 token 切分到多个 GPU，加速 MoE 模型的 prefill 阶段。

:::diagram cp-mental-model-html
```html
<div class="comm-panorama">
  <div class="comm-proc">
    <div class="comm-proc-title">DCP：KV Cache 切分</div>
    <div class="comm-proc-body">
      <div class="comm-node" data-rank="0">DCP Rank 0：Q (完整) + KV[0..N/2]</div>
      <div class="comm-node" data-rank="1">DCP Rank 1：Q (完整) + KV[N/2..N]</div>
      <div class="comm-node-sub">LSE 校正 → 合并部分 attention 输出</div>
    </div>
  </div>
  <div class="comm-channel-group">
    <div class="comm-channel-item">
      <div class="comm-arrow">▶</div>
      <div class="comm-label">DCP 复用 TP GPU，tp_size % dcp_size == 0</div>
    </div>
  </div>
  <div class="comm-proc">
    <div class="comm-proc-title">PCP：Prefill Token 切分</div>
    <div class="comm-proc-body">
      <div class="comm-node" data-rank="0">PCP Rank 0：Token[0..T/2]</div>
      <div class="comm-node" data-rank="1">PCP Rank 1：Token[T/2..T]</div>
      <div class="comm-node-sub">AllGather → MoE → ReduceScatter</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc cp-mental-model-html
Context Parallel 分为两种模式：

**DCP（Decode Context Parallel）**：将 KV cache 切分到多个 GPU。每个 DCP rank 持有完整的 query 但只有部分 KV cache（如 Rank 0 持有 KV[0..N/2]，Rank 1 持有 KV[N/2..N]）。各 rank 独立计算部分 attention 输出，通过 LSE 校正合并为完整结果。

**PCP（Prefill Context Parallel）**：将 prefill 输入 token 切分到多个 GPU。每个 PCP rank 处理部分 token（如 Rank 0 处理 Token[0..T/2]，Rank 1 处理 Token[T/2..T]）。在 MoE 前通过 AllGather 汇聚 token，MoE 后通过 ReduceScatter 重新切分。
:::

## DCP 通信模式

DCP 复用 TP 组的 GPU，将一个 TP 组拆分为 `tp_size / dcp_size` 个 DCP 组。约束条件：`tp_size % dcp_size == 0`。每个 DCP rank 持有 KV cache 的不同序列段，但 query 在所有 DCP rank 上完整（通过 all-gather 获得）。

:::diagram dcp-comm-flow-html
```html
<div class="engine-step-flow">
  <div class="engine-step" data-step="1">
    <div class="engine-step-num">1</div>
    <div class="engine-step-content">
      <div class="engine-step-title">AllGather Query</div>
      <div class="engine-step-desc">沿 head 维度 all-gather Q，每个 DCP rank 获得完整 query</div>
    </div>
  </div>
  <div class="engine-step" data-step="2">
    <div class="engine-step-num">2</div>
    <div class="engine-step-content">
      <div class="engine-step-title">本地 Attention</div>
      <div class="engine-step-desc">完整 Q 对本地 KV cache 做 attention，产出 output + LSE</div>
    </div>
  </div>
  <div class="engine-step" data-step="3">
    <div class="engine-step-num">3</div>
    <div class="engine-step-content">
      <div class="engine-step-title">LSE 校正合并</div>
      <div class="engine-step-desc">用 LSE 值做数值校正，合并各 rank 的部分 attention 输出</div>
    </div>
  </div>
  <div class="engine-step" data-step="4">
    <div class="engine-step-num">4</div>
    <div class="engine-step-content">
      <div class="engine-step-title">ReduceScatter 输出</div>
      <div class="engine-step-desc">将校正后的输出按 head 维度切分回各 DCP rank</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc dcp-comm-flow-html
DCP 通信模式包含四个步骤：
1. AllGather Query — 沿 head 维度 all-gather Q，使每个 DCP rank 获得完整 query。
2. 本地 Attention — 用完整 Q 对本地 KV cache 做 attention，产出部分 output 和 LSE（log-sum-exp）。
3. LSE 校正合并 — 用 LSE 值做数值校正，合并各 rank 的部分 attention 输出为完整结果。
4. ReduceScatter 输出 — 将校正后的输出按 head 维度切分回各 DCP rank。
:::

:::steps id=dcp-flash-attn-steps
### 1. AllGather Query
`get_dcp_group().all_gather(query, dim=1)` 沿 head 维度 all-gather Q，使每个 DCP rank 拥有完整的 query，可以对本地 KV cache 做 attention。
`vllm/v1/attention/backends/flash_attn.py`

### 2. 本地 Attention
用完整 Q 对本地 KV cache 做 attention，产出 context_attn_out 和 LSE（log-sum-exp）。同时用本地 Q 对新 token 做 query attention。
`vllm/v1/attention/backends/flash_attn.py`

### 3. LSE 校正合并
各 DCP rank 的 attention 输出需要用 LSE 做数值校正（因为每个 rank 只看到部分 KV，softmax 分母不同）。校正后合并 context 和 query attention 输出。
`vllm/v1/attention/backends/flash_attn.py`

### 4. ReduceScatter 校正输出
将 LSE 校正后的完整输出沿 head 维度 reduce_scatter，每个 DCP rank 取回属于自己的 head 切片。
`vllm/v1/attention/backends/flash_attn.py`
:::

## LSE 校正原理

每个 DCP rank 只看到部分 KV cache，独立计算 attention 时 softmax 分母不同，直接相加会引入数值误差。LSE（Log-Sum-Exp）校正解决了这个问题。

每个 rank 计算部分 attention 后，同时产出 LSE 值（即 `log(sum(exp(attn_logits)))`）。校正过程：

```python
lse_max = max(lse_i for i in range(dcp_world_size))
global_lse = log(sum(exp(lse_i - lse_max) for i in range(dcp_world_size))) + lse_max
result = sum(output_i * exp(lse_i - global_lse) for i in range(dcp_world_size))
```

直觉理解：`exp(lse_i - global_lse)` 是每个 rank 的 attention 输出在全局 softmax 中的权重。LSE 校正等价于将各 rank 的部分 softmax 结果按正确的比例加权合并。

`vllm/v1/attention/backends/flash_attn.py`

## DCP 合并后端

LSE 校正合并有两种后端实现，核心区别在于通信次数：

| 后端 | NCCL 通信次数 | 流程 | 适用场景 |
|------|-------------|------|---------|
| AG+RS | 3 次 | all_gather(LSE) + all_gather(output) → Triton LSE 校正 → reduce_scatter(corrected) | 通用 |
| A2A | 1 次 | Triton 打包 output+LSE → all_to_all_single → Triton 解包+LSE 加权合并 | 低延迟优先 |

AG+RS 后端分三步通信：先 all_gather LSE 值用于计算全局校正系数，再 all_gather 各 rank 的 attention 输出，Triton kernel 做 LSE 校正计算，最后 reduce_scatter 将结果切分回各 rank。

A2A 后端将 output 和 LSE 打包到同一 buffer，用单次 all_to_all_single 通信完成数据交换，然后 Triton kernel 解包并直接做 LSE 加权合并。通信次数从 3 降为 1，显著降低延迟。

`vllm/v1/attention/ops/dcp_alltoall.py`

## A2A 实现

A2A 后端的核心函数 `dcp_a2a_lse_reduce()` 将 output 和 LSE 打包到同一 buffer，通过单次 all_to_all 完成交换，再解包并做 LSE 加权合并。

:::steps id=dcp-a2a-impl
### 1. 打包 output + LSE
Triton kernel `_dcp_a2a_pack_send` 将每个 rank 的 attention output 和 LSE 值打包到连续 buffer。FP16 输出时每对 rank 间打包 2 个 LSE slot（FP16 占 2 字节，LSE 为 FP32 占 4 字节，2 个 LSE 刚好对齐）；FP32 输出时打包 1 个 LSE slot。
`vllm/v1/attention/ops/dcp_alltoall.py`

### 2. All-to-All 通信
`torch.distributed.all_to_all_single` 单次集合通信，每个 rank 向所有其他 rank 发送不同的 output+LSE 数据块，同时接收其他 rank 的数据块。
`vllm/v1/attention/ops/dcp_alltoall.py`

### 3. 解包 + LSE 加权合并
Triton kernel `_dcp_a2a_unpack_combine` 解包接收到的 output+LSE 数据，用 LSE 值计算加权系数 `exp(lse_i - global_lse)`，对各 rank 的 output 做加权求和，得到校正后的完整 attention 输出。
`vllm/v1/attention/ops/dcp_alltoall.py`
:::

## PCP 通信模式

PCP 是独立于 TP 的并行维度，主要用于 MoE 模型的 prefill 加速。与 DCP 复用 TP GPU 不同，PCP 增加总 GPU 数量。GPU 进程布局为 `reshape(ExternalDP, DP, PP, PCP, TP)`。

:::steps id=pcp-moe-steps
### 1. MoE Dispatch 前：AllGather
`get_pcp_group().all_gather(hidden_states, dim=0)` 沿 token 维度 all-gather，将分散在各 PCP rank 的 token 汇聚，使 MoE router 能看到完整 token 集合。
`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`

### 2. MoE 本地计算
每个 PCP rank 对汇聚后的完整 token 集合执行 MoE dispatch → expert computation → MoE combine。
`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`

### 3. MoE Combine 后：ReduceScatter
`get_pcp_group().reduce_scatter(hidden_states, dim=0)` 沿 token 维度 reduce-scatter，将 MoE 计算结果重新切分到各 PCP rank。
`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
:::

## CP 与 TP/SP 的组合

| 并行维度 | 切分对象 | 通信操作 | 与 TP 的关系 |
|---------|---------|---------|-------------|
| TP | 权重（output/input 维度） | all_reduce / all_gather | — |
| SP | 激活值（序列/token 维度） | reduce_scatter + all_gather | TP 的附带优化，tp>1 时启用 |
| DCP | KV cache（上下文序列） | all_gather(Q) + LSE 校正 + reduce_scatter / all_to_all | 复用 TP GPU，tp_size % dcp_size == 0 |
| PCP | Prefill token | all_gather + reduce_scatter | 独立维度，增加 GPU 数 |

组合约束：`total_cp = pcp_world_size × dcp_world_size`，`total_cp_rank = pcp_rank × dcp_world_size + dcp_rank`。KV cache 按 `total_cp_rank` 交错存储，确保 DCP 和 PCP 的 KV 切分互不冲突。

`vllm/distributed/parallel_state.py`

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--dcp-size` | 1 | DCP 组大小，需满足 tp_size % dcp_size == 0 |
| `--dcp-comm-backend` | "ag_rs" | DCP 合并后端：ag_rs（AllGather+ReduceScatter）或 a2a（All-to-All） |
| `--prefill-context-parallel-size` | 1 | PCP 组大小，独立于 TP 增加 GPU 数 |
| `--context-parallel-size` | 1 | 总 CP 大小（= pcp_size × dcp_size） |
