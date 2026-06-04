---
id: expert-parallel
title: Expert Parallel
category: distributed
level: advanced
status: draft
readingMinutes: 12
tags:
  - Expert Parallel
  - MoE
  - All To All
codeRefs:
  - vllm/model_executor/models/deepseek_v2.py
  - vllm/model_executor/layers/fused_moe/layer.py
heroText: MoE 专家分布到多个 GPU，路由器决定每个 token 去哪个专家，[all-to-all](term:每个 GPU 向所有其他 GPU 发送不同数据并接收不同数据的集合通信。) 完成专家调度与结果回收。
---

## 心智模型

想象一家大公司有很多专业部门（专家）。每个办公室（GPU）驻扎几个部门。当请求进来时，路由器决定哪些部门应该处理它。请求被转发（dispatch）到对应的办公室，处理后结果汇总（combine）回来——这就是 Expert Parallel 的核心流程。

:::diagram ep-mental-model-html
```html
<div class="comm-panorama">
  <div class="comm-proc">
    <div class="comm-proc-title">输入 Token</div>
    <div class="comm-proc-body">
      <div class="comm-node" data-expert="0">T0→E0</div>
      <div class="comm-node" data-expert="2">T1→E2</div>
      <div class="comm-node" data-expert="1">T2→E1</div>
      <div class="comm-node" data-expert="0">T3→E0</div>
    </div>
  </div>
  <div class="comm-channel-group">
    <div class="comm-channel-item">
      <div class="comm-arrow">→</div>
      <div class="comm-label">Dispatch (All2All)</div>
    </div>
  </div>
  <div class="comm-proc">
    <div class="comm-proc-title">专家计算</div>
    <div class="comm-proc-body">
      <div class="comm-node">
        GPU 0
        <div class="comm-node-sub">E0, E1</div>
      </div>
      <div class="comm-node">
        GPU 1
        <div class="comm-node-sub">E2, E3</div>
      </div>
    </div>
  </div>
  <div class="comm-channel-group">
    <div class="comm-channel-item">
      <div class="comm-arrow">→</div>
      <div class="comm-label">Combine (All2All)</div>
    </div>
  </div>
  <div class="comm-proc">
    <div class="comm-proc-title">输出 Token</div>
    <div class="comm-proc-body">
      <div class="comm-node">T0'</div>
      <div class="comm-node">T1'</div>
      <div class="comm-node">T2'</div>
      <div class="comm-node">T3'</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc ep-mental-model-html
Expert Parallel 的核心流程：

**输入 Token**：每个 token 被路由器分配到特定专家（如 T0→E0, T1→E2, T2→E1, T3→E0）。

**Dispatch (All2All)**：通过 all-to-all 通信，将 token 发送到其目标专家所在的 GPU。例如 E0 在 GPU 0，则 T0 和 T3 被发送到 GPU 0；E2 在 GPU 1，则 T1 被发送到 GPU 1。

**专家计算**：每个 GPU 对接收到的 token 执行本地专家计算。GPU 0 执行 E0 和 E1，GPU 1 执行 E2 和 E3。

**Combine (All2All)**：通过 all-to-all 通信（反向），将专家输出结果发送回原始 token 所在的 GPU，按路由器权重加权求和。
:::

## EP 组构造

EP 组跨越 DP×PCP×TP 维度，在每个 PP rank 内部构造。构造逻辑：

```python
all_ranks = torch.arange(world_size).reshape(ExternalDP, DP, PP, PCP, TP)
ep_ranks = all_ranks.transpose(1, 2).reshape(-1, DP * PCP * TP).unbind(0)
```

每个 EP rank 持有 `n_routed_experts / ep_size` 个专家。例如 8 个路由专家、4 个 EP rank，则每个 rank 持有 2 个专家。

`vllm/distributed/parallel_state.py`

## Expert Map Manager

`FusedMoE` 层管理专家映射关系，核心数据结构：

| 数据结构 | 说明 |
|---------|------|
| `expert_map` | 全局专家 ID → 本地 ID（-1 表示不在本 rank） |
| `routing_tables` | global_to_physical, physical_to_global, local_to_global 映射 |
| `local_num_experts` | 本 rank 持有的专家数量 |

```python
expert_map = torch.full((num_experts,), -1, dtype=torch.int32)
for local_id, global_id in enumerate(local_expert_ids):
    expert_map[global_id] = local_id
```

路由器输出的专家 ID 是全局 ID，通过 `expert_map` 转换为本地 ID。若映射为 -1，表示该专家不在本 rank，需要通过 all-to-all 发送到其他 rank。

`vllm/model_executor/layers/fused_moe/layer.py`

## Dispatch-Combine 流程

:::steps id=ep-dispatch-combine
### 1. Router 计算 top-k 专家
Router 对每个 token 计算 expert scores，选择 top-k 个专家及其权重。输出 `topk_ids`（专家全局 ID）和 `topk_weights`（路由权重）。
`vllm/model_executor/layers/fused_moe/layer.py`

### 2. Dispatch：All2All 发送 token
根据 `topk_ids`，将 token 发送到目标专家所在的 EP rank。all-to-all 通信后，每个 rank 收到需要由本地专家处理的 token。
`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`

### 3. 本地专家计算
每个 EP rank 对接收到的 token 执行本地专家的 forward pass。专家计算通常是 GEMM（gate_proj, up_proj, down_proj）。
`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`

### 4. Combine：All2All 收集结果
反向 all-to-all 通信，将专家输出结果发送回原始 token 所在的 EP rank。
`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`

### 5. 加权求和
按 `topk_weights` 对各专家的输出做加权求和，得到最终 MoE 输出。
`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
:::

## All2All 后端

vLLM 支持多种 All2All 后端实现，针对不同硬件和场景优化：

| 后端 | 特点 | 适用场景 |
|------|------|---------|
| DeepEP HT | High-Throughput，高吞吐量 | 大 batch、吞吐优先 |
| DeepEP LL | Low-Latency，FP8 dispatch | 低延迟、FP8 量化 |
| MoRI | I/O based，绕过 NVLink | 跨节点、网络传输 |
| FlashInfer NVLink 2-sided | FlashInfer 实现，双向 NVLink | 单节点、NVLink |
| FlashInfer NVLink 1-sided | FlashInfer 实现，单向 NVLink | 单节点、优化延迟 |
| NIXL EP | NVIDIA NIXL 库 | NVIDIA GPU |
| Naive | AllGather + ReduceScatter 回退 | 无专用 all2all kernel |
| None | 无 EP（所有专家在单 GPU） | ep_size=1 |

`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`

## Naive DP+EP 回退

当没有专用 all-to-all kernel 时，使用 Naive 后端回退到 AllGather + ReduceScatter：

**Dispatch 阶段**：
1. AllGather：将所有 EP rank 的 token 汇聚到每个 rank
2. 每个 rank 根据本地专家 ID 筛选需要处理的 token

**Combine 阶段**：
1. 每个 rank 计算本地专家输出
2. ReduceScatter：将输出按 token 维度切分回各 rank

Naive 后端通信量更高（AllGather + ReduceScatter 各一次），但实现简单，无需专用 kernel。

`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`

## EPLB 负载均衡

Expert Parallelism Load Balancing（EPLB）通过冗余专家实现负载均衡。启用后，部分专家会被复制到多个 rank，根据负载动态调整专家分布。

| 配置项 | 说明 |
|--------|------|
| `enable_eplb` | 启用 EPLB |
| `num_redundant_experts` | 冗余专家数量 |
| `ep_rebalance_interval` | 重平衡间隔（步数） |

EPLB 统计各专家的负载（处理的 token 数），定期重新分配专家到不同 rank，使各 rank 负载更均衡。

`vllm/model_executor/layers/fused_moe/layer.py`

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--expert-parallel-size` | 1 | EP 组大小，决定专家分布到多少 GPU |
| `--enable-eplb` | False | 启用 EPLB 负载均衡 |
| `--num-redundant-experts` | 0 | 冗余专家数量（用于 EPLB） |
| `--ep-revise-role` | - | EP 角色修订（调试用） |
| `--moe-all2all-backend` | auto | All2All 后端选择 |
