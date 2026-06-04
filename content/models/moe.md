---
id: moe
title: "MoE 架构"
category: models
level: advanced
status: draft
readingMinutes: 14
tags:
  - MoE
  - Expert
  - Routing
  - FusedMoE
  - EPLB
codeRefs:
  - vllm/model_executor/layers/fused_moe/layer.py
  - vllm/v1/worker/gpu/eplb_utils.py
heroText: MoE 模型的路由与专家并行：[grouped top-k](term:将 expert 分组后在组内选择 top-k，平衡负载与精度。) 路由、[EP 负载均衡](term:动态调整 expert 在 GPU 间的分布，使每个 GPU 的计算量尽量均等。)、fused MoE kernel 与 expert 并行的协作。
---

## 心智模型

:::diagram moe-mental-model-html
```html
<div class="arch-diagram">
<div class="arch-row">
<div class="arch-box">Input Token</div>
<div class="arch-arrow">→</div>
<div class="arch-proc">
<div class="arch-proc-title">Router</div>
<div class="arch-proc-sub">top-k selection</div>
</div>
<div class="arch-arrow">→</div>
<div class="arch-row">
<div class="arch-module arch-highlight">Expert 1</div>
<div class="arch-module arch-highlight">Expert 2</div>
<div class="arch-module">Expert N</div>
</div>
</div>
<div class="arch-label">MoE: 稀疏激活，只激活部分专家处理每个 token</div>
</div>
```
:::

:::diagram-desc moe-mental-model-html
MoE 架构的核心是稀疏激活：每个 token 不经过所有专家，而是由 Router 选择 top-k 个专家处理。图中 Input Token 经过 Router 后，被路由到 Expert 1 和 Expert 2（高亮显示），其他专家不参与计算。这种设计在保持模型容量的同时大幅降低计算量。
:::

## MoE 路由机制

vLLM 的 Router 工厂按优先级选择路由实现：

1. **RoutingSimulatorRouter** — 环境变量启用时使用
2. **ZeroExpertRouter** — zero_expert_type 不为 None 时使用
3. **GroupedTopKRouter** — use_grouped_topk=True 时使用（DeepSeek-V2/V3）
4. **CustomRoutingRouter** — 自定义路由函数
5. **FusedTopKBiasRouter** — e_score_correction_bias 存在时使用（DeepSeek-V3/V4）
6. **AiterSharedRoutedFusedMoERouter** — ROCm 平台共享专家路由
7. **FusedTopKRouter** — 默认标准 Top-K 路由

### 标准 Top-K 路由（FusedTopKRouter）

使用融合 CUDA kernel（`ops.topk_softmax` / `ops.topk_sigmoid`）实现：

- 预分配输出张量，避免动态内存分配
- 支持 softmax 和 sigmoid 两种评分函数
- 单次 kernel 调用完成评分、top-k 选择和权重归一化

### Grouped Top-K 路由（DeepSeek-V2/V3）

Grouped Top-K 将专家分组，在组内选择 top-k，实现负载均衡与精度的平衡：

:::steps id=grouped-topk-steps
### 1. 计算评分
对 gating output 应用 softmax 或 sigmoid，得到每个专家的原始评分。

### 2. 偏置校正（仅用于选择，不影响权重）
如果有 e_score_correction_bias，将其加到评分上，但仅用于专家选择，不改变最终权重。

### 3. 组评分计算
对每个专家组计算组评分：有偏置时取 top-2 求和，无偏置时取最大值。

### 4. 选择 top-topk_group 个组
按组评分选择得分最高的 topk_group 个组。

### 5. 组内选择 top-k 专家
在选中的组内，按原始评分选择 top-k 个专家。

### 6. 权重重归一化
对选中的专家权重重新归一化，确保权重和为 1。
:::

### Top-K with Bias（DeepSeek-V3/V4, MiniMax）

支持多种评分函数：

| 评分函数 | 公式 | 使用模型 |
|---------|------|---------|
| softmax | softmax(gating_output) | Mixtral, Qwen3MoE |
| sigmoid | sigmoid(gating_output) | DeepSeek-V2/V3 |
| sqrtsoftplus | sqrt(softplus(gating_output)) | DeepSeek-V4 |

DeepSeek-V4 还支持 **Hash MoE**，通过 `hash_indices_table` 实现基于哈希的专家选择，进一步优化负载均衡。

## FusedMoE Kernel

FusedMoE kernel 是 MoE 推理性能的关键优化，采用 Blocked GEMM 配合专家感知调度：

### 核心优化技术

**Memory Coalescing**：按专家 ID 分组排序 token，使同一专家的 token 连续访问，提升 L2 cache 命中率。

**Token Sorting + Padding**：调用 `moe_align_block_size` 将 token 按专家排序，并填充到 `BLOCK_SIZE_M` 的倍数，确保 tensor core 高效利用。

**Router Weight 内联**：将路由权重乘法集成到 kernel 内部，避免单独的 kernel 启动开销。

**Tensor Core 利用**：通过 `tl.dot()` 使用 tensor core，支持 FP8 和 INT8 量化。

### Auto-tuned 配置

配置从 JSON 文件加载，根据 batch size 自动选择最优配置：

- 小 batch：优先低延迟配置
- 大 batch：优先高吞吐配置
- 默认配置自适应调整

## Expert Parallel 与 All-to-All

Expert Parallel（EP）将专家分布到多个 GPU，每个 GPU 只持有部分专家：

### EP 组结构

EP 组跨越 DP × PCP × TP，在每个 PP rank 内：

- 每个 EP rank 持有 `n_routed_experts / ep_size` 个专家
- 专家放置策略：linear（连续放置）、round_robin（交错放置）

### Dispatch-Combine 流程

:::diagram ep-dispatch-combine-html
```html
<div class="arch-diagram">
<div class="arch-proc arch-proc-api">
<div class="arch-proc-title">GPU 0</div>
<div class="arch-box">Experts 0-7</div>
</div>
<div class="arch-channel">
<div class="arch-arrow">Dispatch</div>
<div class="arch-label">All2All</div>
</div>
<div class="arch-proc arch-proc-core">
<div class="arch-proc-title">GPU 1</div>
<div class="arch-box">Experts 8-15</div>
</div>
<div class="arch-channel">
<div class="arch-arrow">Compute</div>
<div class="arch-label">Expert</div>
</div>
<div class="arch-proc arch-proc-worker">
<div class="arch-proc-title">GPU 2</div>
<div class="arch-box">Experts 16-23</div>
</div>
<div class="arch-channel">
<div class="arch-arrow">Combine</div>
<div class="arch-label">All2All</div>
</div>
<div class="arch-proc arch-proc-api">
<div class="arch-proc-title">Result</div>
<div class="arch-box">Output</div>
</div>
</div>
```
:::

:::diagram-desc ep-dispatch-combine-html
EP 的 Dispatch-Combine 流程：1) Dispatch 阶段通过 All2All 将 token 发送到持有对应专家的 GPU；2) Compute 阶段各 GPU 并行执行专家计算；3) Combine 阶段通过 All2All 将结果收集回原 GPU。All2All 通信是 EP 的主要开销。
:::

### All2All 后端

| 后端 | 特点 | 适用场景 |
|-----|------|---------|
| deepep_high_throughput | 高吞吐优化 | 大 batch 推理 |
| deepep_low_latency | 低延迟优化 | 在线服务 |
| flashinfer | FlashInfer 集成 | 通用场景 |
| nixl_ep | NVIDIA 优化 | H100/A100 |
| allgather_reducescatter | 标准实现 | 兼容性优先 |

## EPLB 负载均衡

Expert Parallel Load Balancing（EPLB）动态调整专家分布，解决专家负载不均衡问题：

### 冗余专家机制

热门专家创建多个副本分布到不同 GPU：

- 256 个逻辑专家 + 32 个冗余专家 = 288 个物理专家
- 32 个 EP rank → 每个 GPU 9 个物理专家
- 冗余专家分担热门专家的计算压力

### 负载追踪

使用滑动窗口追踪每个专家的负载：

- 记录每个专家处理的 token 数
- 计算均衡度指标：`avg_tokens / max_tokens`（1.0 = 完美均衡）
- 周期性触发重排

### 重排流程

:::steps id=eplb-rearrange-steps
### 1. 收集负载
各 GPU 统计本地专家的负载，映射到逻辑专家 ID。

### 2. All-Reduce 汇总
通过 all-reduce 操作汇总全局负载信息。

### 3. 计算新映射
根据负载计算新的专家分布，热门专家分配更多冗余副本。

### 4. 异步权重迁移
后台线程异步传输专家权重，不阻塞推理。
:::

### 异步 EPLB

权重迁移在后台线程执行，避免阻塞推理：

- 迁移期间使用旧映射继续推理
- 迁移完成后原子切换到新映射
- 支持增量迁移，减少内存峰值

## 主要 MoE 模型变体

| 模型 | 路由策略 | 评分函数 | Grouped | 共享专家 | EPLB 支持 |
|-----|---------|---------|---------|---------|----------|
| Mixtral | Top-K | Softmax | 否 | 否 | 是 |
| DeepSeek-V2 | Grouped Top-K | Sigmoid | 是 | 是 | 是 |
| DeepSeek-V3 | Grouped Top-K + Bias | Sigmoid | 是 | 是 | 是 |
| DeepSeek-V4 | Top-K + Bias + Hash | sqrtsoftplus | 否 | 是 | 是 |
| Qwen3MoE | Top-K | Softmax | 否 | 否 | 是 |
| GLM4MoE | Top-K | Softmax | 否 | 否 | 是 |
| MiniMax | Top-K + Bias | Sigmoid | 否 | 否 | 是 |
| Llama4 | Top-1 | Sigmoid | 否 | 否 | 是 |

### 共享专家

部分模型（DeepSeek 系列）引入共享专家机制：

- 共享专家对所有 token 都激活
- 路由专家通过 top-k 选择
- 输出 = 共享专家输出 + 路由专家加权和

## 关键配置

| 配置项 | 默认值 | 说明 |
|-------|-------|------|
| `--expert-parallel-size` | 1 | EP 组大小，决定专家分布的 GPU 数量 |
| `--enable-eplb` | False | 启用 EPLB 动态负载均衡 |
| `--num-redundant-experts` | 0 | 冗余专家数量，用于 EPLB |
| `--moe-all2all-backend` | allgather_reducescatter | All2All 通信后端 |
| `--moe-block-size` | 64 | MoE kernel 的 block size |

### 调优建议

**小 batch（< 64 tokens）**：
- EP size = 1，避免 All2All 开销
- 使用低延迟 All2All 后端

**大 batch（> 512 tokens）**：
- 启用 EP，将专家分布到多 GPU
- 启用 EPLB，均衡专家负载
- 使用高吞吐 All2All 后端

**热门专家场景**：
- 配置冗余专家（`--num-redundant-experts`）
- 启用 EPLB 自动重排
