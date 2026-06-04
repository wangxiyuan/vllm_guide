---
id: cuda-graph
title: CUDA Graph
category: execution
level: advanced
status: draft
readingMinutes: 16
tags:
  - CUDA Graph
  - Capture
  - Replay
  - Piecewise
codeRefs:
  - vllm/v1/worker/gpu/cudagraph_utils.py
  - vllm/v1/cudagraph_dispatcher.py
heroText: "decode 阶段的核心加速机制：[CUDA Graph](term:将 GPU 操作序列记录为固定图结构，replay 时无需重新调度内核，消除 CPU launch overhead。) 消除 CPU 内核调度开销，piecewise 模式支持 attention 与前向计算分离捕获。"
---

## 心智模型

CUDA Graph 的核心思想：**将 decode 阶段固定的 GPU 操作序列"录下来"，之后每步只需 replay 整张图，跳过 CPU 逐个 launch 内核的开销。** piecewise 模式将 attention 和前向计算分别捕获，以适应动态 shape。

:::diagram cuda-graph-html
```html
<div class="engine-step-flow">
  <div class="engine-step">
    <div class="engine-step-num">1</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Warmup</div>
      <div class="engine-step-desc">dummy 输入执行 forward，GPU 达到稳定状态</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">2</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Capture</div>
      <div class="engine-step-desc">记录固定 batch size 下的 CUDA 操作序列为图</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">3</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Replay</div>
      <div class="engine-step-desc">每步 replay 图，CPU 只需一次 launch 调用</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc cuda-graph-html
CUDA Graph 三阶段流程：
1. Warmup：用 dummy 输入执行 forward，让 GPU 达到稳定状态（CUDA 上下文初始化、cache 分配完成）
2. Capture：在固定 batch size 下，将 GPU 操作序列记录为图结构
3. Replay：每步只需 replay 整张图，CPU 端只需一次 launch 调用，消除逐个内核调度的开销
:::

### CPU Launch Overhead 的来源

GPU 内核执行需要 CPU 端发起 launch 调用。每次 launch 涉及驱动层状态检查、命令队列写入、同步开销。典型 Transformer 层包含数十个内核（embedding、layer norm、linear、attention、activation 等），decode 阶段每次 forward 的 GPU 计算量很小（batch 中每个请求只生成 1 个 token），但 launch 次数不变，导致 CPU 端开销占比极高。

### 为什么 Decode 收益最大而 Prefill 不适用

Decode 阶段的操作序列几乎固定：相同的模型结构、相同的内核调用顺序、可预测的 tensor shape（uniform decode batch）。Prefill 阶段 token 数量变化大、attention shape 动态，无法捕获为固定图。

## CUDA Graph 基本原理

将多个 GPU kernel launch 合并为一次图提交，消除 CPU 端逐个调度开销。

```python
# 捕获阶段
with torch.cuda.graph(cudagraph, pool=graph_pool, stream=current_stream):
    output = model(*args, **kwargs)

# Replay 阶段
cudagraph.replay()
```

关键约束：**replay 时输入 tensor 的地址必须和捕获时一致**，因此 vLLM 使用预分配的持久化 GPU buffer。

## FULL vs PIECEWISE 模式

vLLM 定义五种 [CUDAGraphMode](term:控制 CUDA Graph 捕获行为的枚举类型，决定哪些操作被捕获为图、哪些走 eager 路径。)：

| 模式 | Decode 批次 | Mixed/Prefill 批次 | 显存 | 要求 |
|------|------------|-------------------|------|------|
| `NONE` | Eager | Eager | 最低 | 无 |
| `PIECEWISE` | Piecewise CG | Piecewise CG | 中等 | 需要 piecewise 编译 |
| `FULL` | Full CG | Full CG | 中等 | Attention 后端 CG 支持 |
| `FULL_DECODE_ONLY` | Full CG | Eager | 中低 | Decode 端 CG 支持 |
| `FULL_AND_PIECEWISE` | Full CG | Piecewise CG | 最高 | 两者都需支持 |

:::diagram cg-modes-html
```html
<div class="engine-step-flow">
  <div class="engine-step">
    <div class="engine-step-num">F</div>
    <div class="engine-step-content">
      <div class="engine-step-title">FULL 模式</div>
      <div class="engine-step-desc">整个 forward 捕获为一张图，包括 attention</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">P</div>
    <div class="engine-step-content">
      <div class="engine-step-title">PIECEWISE 模式</div>
      <div class="engine-step-desc">attention 间分段捕获，attention 本身 eager 执行</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">F+P</div>
    <div class="engine-step-content">
      <div class="engine-step-title">FULL_AND_PIECEWISE</div>
      <div class="engine-step-desc">decode 用 FULL，mixed 用 PIECEWISE</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc cg-modes-html
三种主要 CUDA Graph 模式对比：

FULL 模式将整个 forward pass（包括 attention）捕获为一张图，要求 attention 后端支持 CUDA Graph。

PIECEWISE 模式将模型在 attention 操作处断开，分别捕获 attention 之间的子图，attention 本身以 eager 模式执行。适用于 attention shape 动态变化的场景。

FULL_AND_PIECEWISE 是双模式：uniform decode 批次使用 FULL 模式（性能最优），mixed prefill-decode 批次使用 PIECEWISE 模式（适应动态 shape）。
:::

### Attention 后端 CG 支持等级

| 后端 | CG 支持 |
|------|---------|
| FlashAttention v3 | `ALWAYS` |
| Triton Attention | `ALWAYS` |
| FlashAttention v2 | `UNIFORM_BATCH` |
| FlashMLA | `UNIFORM_BATCH` |
| FlashInfer | `UNIFORM_SINGLE_TOKEN_DECODE` |
| Mamba | `UNIFORM_SINGLE_TOKEN_DECODE` |

当后端报告 `NEVER` 时，FULL 模式自动降级为 PIECEWISE 或 NONE。

`vllm/v1/attention/backend.py`

## Warmup 与 Capture

捕获前必须 warmup 确保 CUDA 上下文初始化完成、cache 分配稳定。

:::steps id=cg-capture-steps
### 1. Warmup
用 dummy 输入执行 forward，使 GPU 达到稳定状态。warmup 次数由 `cudagraph_num_of_warmups` 控制（默认 0）。
`vllm/v1/worker/gpu_model_runner.py`

### 2. Capture
在 `graph_capture()` 上下文中，对每个 batch size 区间调用 `_warmup_and_capture()`。capture 时使用 `torch.cuda.graph()` 记录操作序列。

### 3. 全局内存池
所有 CUDA Graph 共享一个全局内存池（`get_global_graph_pool()`），防止 OOM 并复用内存。

### 4. 捕获顺序
PIECEWISE 先于 FULL 捕获——PIECEWISE 的激活值更大，FULL 可以复用已分配的池内存。
:::

### Batch Size 区间（Bucket）策略

默认捕获尺寸：

```python
max_graph_size = min(max_num_seqs * 2, 512)
cudagraph_capture_sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_graph_size + 1, 16))
```

交互式性能模式使用 `1, 2, 3, ..., 32`（每个整数），减少 padding 开销。

运行时通过查表 `_bs_to_padded_graph_size` 将实际 batch size 映射到最近的已捕获尺寸：

| 实际大小 | 映射到 |
|---------|--------|
| 1 | 1 |
| 2 | 2 |
| 3-4 | 4 |
| 5-8 | 8 |
| 9-16 | 16 |

`vllm/config/vllm.py`

## CudagraphDispatcher

运行时调度器，根据当前 batch size 选择对应的已捕获图并 replay。

:::diagram cg-dispatch-html
```html
<div class="engine-step-flow">
  <div class="engine-step">
    <div class="engine-step-num">1</div>
    <div class="engine-step-content">
      <div class="engine-step-title">查询 keys</div>
      <div class="engine-step-desc">num_tokens → 创建 padded BatchDescriptor</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">2</div>
    <div class="engine-step-content">
      <div class="engine-step-title">尝试 FULL</div>
      <div class="engine-step-desc">batch_desc ∈ cudagraph_keys[FULL]?</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">3</div>
    <div class="engine-step-content">
      <div class="engine-step-title">尝试 PIECEWISE</div>
      <div class="engine-step-desc">放松约束（num_reqs=None, uniform=False）</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">4</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Fallback NONE</div>
      <div class="engine-step-desc">无匹配图 → eager 模式</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc cg-dispatch-html
CudagraphDispatcher 的调度逻辑：
1. 根据当前 num_tokens 创建 padded BatchDescriptor
2. 优先尝试 FULL 模式：检查 BatchDescriptor 是否在 cudagraph_keys[FULL] 中
3. FULL 未命中则尝试 PIECEWISE：放松约束（num_reqs=None, uniform=False）后查表
4. 都未命中则 fallback 到 NONE（eager 模式）

PIECEWISE 的约束更宽松，因为不需要精确匹配请求数（num_reqs=None）和 uniform 标志。
:::

### 输入张量 Copy-In

vLLM 使用预分配的持久化 GPU buffer，模型始终从相同的 GPU 内存地址读取：

```python
self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
self.positions = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=self.device)
self.query_start_loc = self._make_buffer(self.max_num_reqs + 1, dtype=torch.int32)
self.seq_lens = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
```

每步将真实数据拷贝到这些 buffer 中，保证 replay 时输入地址不变。

### Fallback 条件

| 条件 | 说明 |
|------|------|
| `num_tokens > max_cudagraph_capture_size` | 超出捕获范围 |
| `num_tokens == 0` | 空批次 |
| `cudagraph_mode == NONE` | 全局禁用 |
| Cascade attention 激活 | 不支持 FULL |
| Encoder-decoder 有 encoder 输入 | 跳过编译路径 |
| `calculate_kv_scales=True` | FP8 KV cache 动态计算 |
| `force_eager=True` | 显式覆盖 |

`vllm/v1/cudagraph_dispatcher.py`

## LoRA 与 CUDA Graph

LoRA 动态权重合并与固定图结构存在冲突，需特殊处理。

### 专用图捕获

当 `cudagraph_specialize_lora=True`（默认）时，为有无 LoRA 的场景分别捕获图：

```python
# LoRA 计数特化：2 的幂次 + max_loras+1
# 例如 max_loras=8 → 捕获 {1, 2, 4, 9} 个活跃 LoRA 的图
```

BatchDescriptor 包含 `has_lora` 和 `num_active_loras` 字段，调度时根据当前活跃 LoRA 数量选择对应图。

### 配置项

| 场景 | 行为 |
|------|------|
| `specialize_lora=True` | 分别捕获有/无 LoRA 的图，无 LoRA 时无额外开销 |
| `specialize_lora=False` | 统一使用 `max_loras+1` 图，有冗余开销 |

`vllm/lora/utils.py`

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--enforce-eager` | False | 禁用 CUDA Graph，强制 eager 模式 |
| `cudagraph_mode` | FULL_AND_PIECEWISE | 捕获模式 |
| `cudagraph_capture_sizes` | 自动 | 指定捕获的 batch size 列表 |
| `max_cudagraph_capture_size` | min(max_num_seqs×2, 512) | 最大捕获尺寸 |
| `cudagraph_num_of_warmups` | 0 | 捕获前 warmup 次数 |
| `cudagraph_copy_inputs` | False | 是否拷贝输入到内部 buffer（PIECEWISE） |
| `cudagraph_specialize_lora` | True | 为 LoRA 分开捕获图 |
| `cudagraph_mm_encoder` | False | 启用多模态 encoder CG |

`vllm/config/compilation.py`
