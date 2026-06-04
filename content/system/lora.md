---
id: lora
title: LoRA
category: system
level: intermediate
status: draft
readingMinutes: 14
tags:
  - LoRA
  - Punica
  - SGMV
  - Adapter
codeRefs:
  - vllm/lora/lora_model.py
  - vllm/lora/punica_wrapper/punica_gpu.py
  - vllm/v1/worker/lora_model_runner_mixin.py
heroText: "多 [LoRA](term:Low-Rank Adaptation，在冻结的 base model 权重旁添加低秩可训练矩阵，实现参数高效的微调。) 并发推理：Punica [SGMV](term:Segmented Matrix-Vector Multiplication，分段矩阵-向量乘法，将不同 adapter 的计算打包到单次 GPU kernel launch 中并行执行。) kernel 实现单 batch 内多 adapter 混合执行，LRU 管理 adapter 加载与卸载。"
---

## 心智模型

想象一家翻译公司，每位客户（请求）需要不同风格（adapter）的翻译。朴素做法是：先翻译完所有"A 风格"的文件，再翻译"B 风格"的——每次切换风格都要重新准备。Punica SGMV 的做法是：**把所有文件混在一起，翻译官在同一轮工作中根据每页标注的"风格号"切换对应规则**，一轮完成所有风格。而 GPU 显存有限，不可能同时备齐所有风格的规则手册，于是用 LRU 策略：最久没用的风格手册先下架，需要时再从仓库（磁盘）取回。

:::diagram lora-overview-html
```html
<div class="arch-diagram">
<div class="arch-row">
  <div class="arch-box">Request Batch<br><small>R1(adapter-A) R2(adapter-B) R3(adapter-A)</small></div>
  <div class="arch-arrow">→</div>
  <div class="arch-proc">
    <div class="arch-proc-title">LoRA Router</div>
    <div class="arch-proc-sub">按 adapter_id 分组</div>
  </div>
  <div class="arch-arrow">→</div>
  <div class="arch-highlight">
    <div class="arch-module">Punica SGMV</div>
    <div class="arch-module">多 adapter 并行</div>
  </div>
</div>
</div>
```
:::

:::diagram-desc lora-overview-html
LoRA 推理概览流程：请求批次（包含不同 adapter 的请求）进入 LoRA Router，按 adapter_id 分组后，由 Punica SGMV kernel 在单次 GPU kernel launch 中并行处理多个 adapter 的计算，避免串行执行。

核心挑战：如何在单次 forward 中高效处理多个不同的 adapter，避免串行执行带来的性能损失。
:::

LoRA 推理的核心挑战：**如何在单次 forward 中高效处理多个不同的 adapter，避免串行执行带来的性能损失**。

## LoRA 层实现（Punica SGMV）

### LoRA 的数学形式

LoRA 在 base model 的每个线性层旁添加两个低秩矩阵 A 和 B。原始权重 W 不变，前向计算变为：

```
y = W·x + B·A·x
```

其中 A 的形状为 `(rank, in_features)`，B 的形状为 `(out_features, rank)`。rank 通常很小（4-64），所以 BA 的参数量远小于 W。

### 朴素方法的问题

如果 batch 中有 N 个不同的 adapter，朴素做法是：

1. 按 adapter_id 分组请求
2. 对每组分别执行 `W·x + B_i·A_i·x`
3. 拼接结果

这意味着 N 次 kernel launch，每次只处理一部分 token，GPU 利用率低。当 adapter 数量增多时，延迟线性增长。

### SGMV：分段矩阵-向量乘法

Punica SGMV 的核心思想：**将所有 token 的 LoRA 计算打包到一次 kernel launch 中**。kernel 内部根据每个 token 所属的 adapter_id，路由到对应的权重矩阵执行计算。

:::diagram lora-sgmv-html
```html
<div class="cache-flow">
  <div class="cache-step">
    <div class="cache-step-num">1</div>
    <div>
      <div class="cache-step-title">输入：batch 中混合了不同 adapter 的 token</div>
      <div class="cache-step-desc">每个 token 携带自己的 adapter_id（如 0=base, 1=adapter-A, 2=adapter-B）</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">2</div>
    <div>
      <div class="cache-step-title">Segment 分组：按 adapter_id 将 token 分段</div>
      <div class="cache-step-desc">adapter_id=0 的 token 连续排列，adapter_id=1 的连续排列……形成 segments</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">3</div>
    <div>
      <div class="cache-step-title">SGMV Kernel：每个 segment 独立计算 B_i·A_i·x_i</div>
      <div class="cache-step-desc">单次 kernel launch 中，GPU 并行处理所有 segment，每个 segment 使用对应 adapter 的权重</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">4</div>
    <div>
      <div class="cache-step-title">输出：所有 token 的 LoRA 增量合并到 base model 输出</div>
      <div class="cache-step-desc">y = W·x + lora_output，其中 lora_output 是 SGMV 的结果</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc lora-sgmv-html
Punica SGMV 执行流程：
1. 输入 batch 中混合了不同 adapter 的 token，每个 token 携带自己的 adapter_id（0=base, 1=adapter-A, 2=adapter-B）。
2. 按 adapter_id 将 token 分段（segment），相同 adapter_id 的 token 连续排列。
3. SGMV Kernel 在单次 kernel launch 中，每个 segment 独立计算 B_i·A_i·x_i，使用对应 adapter 的权重。
4. 输出时，所有 token 的 LoRA 增量合并到 base model 输出：y = W·x + lora_output。

关键优势：N 个 adapter 只需 1 次 kernel launch，而非 N 次。
:::

关键优势：**N 个 adapter 只需 1 次 kernel launch，而非 N 次**。这消除了 kernel launch overhead 和 GPU 空闲，使多 LoRA 并发推理的吞吐接近单 adapter 推理。

`vllm/lora/punica_wrapper/punica_gpu.py`

### LoRA 层的 forward 路径

在 vLLM 中，每个线性层被替换为带 LoRA 支持的版本（如 `LinearWithLoRA`）。forward 时：

1. 先执行 base model 的线性计算 `base_output = W·x`
2. 识别哪些 token 需要 LoRA（adapter_id != 0）
3. 调用 Punica SGMV kernel 计算 `lora_output = B·A·x`（仅对有 LoRA 的 token）
4. 将 lora_output 加到 base_output 对应位置

adapter_id=0 表示不使用任何 LoRA（纯 base model 推理），这些 token 不参与 SGMV 计算。

`vllm/lora/lora_model.py`

## 多 LoRA 并发调度

### Scheduler 的 LoRA 感知

调度器在调度请求时需要考虑 LoRA adapter 的可用性。每个请求携带 `lora_name` 和 `lora_int_id`，调度器据此决定：

- 该请求的 adapter 是否已在 GPU 上
- 是否需要先加载 adapter（触发 LRU 淘汰）
- 当前并发 adapter 数是否已达 `max_loras` 上限

### 请求分组

在 model runner 执行 forward 前，请求按 adapter_id 分组。这不是串行执行的分组，而是为了让 SGMV kernel 知道哪些 token 属于哪个 adapter：

```python
lora_requests = [req for req in scheduled_reqs if req.lora_int_id > 0]
active_adapters = set(req.lora_int_id for req in lora_requests)
```

### 调度约束

| 约束 | 说明 |
|------|------|
| `max_loras` | 同一时刻 GPU 上最多驻留的 adapter 数量 |
| adapter 加载延迟 | 新 adapter 首次使用时需从磁盘加载权重 |
| CUDA Graph 兼容 | 活跃 adapter 数量影响 CUDA Graph 选择 |

当请求需要的 adapter 不在 GPU 上且已达 `max_loras` 上限时，该请求会被延迟调度（等待有 adapter 被卸载），而非抢占运行中的请求。

## Adapter LRU 管理

### 为什么需要 LRU？

GPU 显存有限，无法同时驻留所有 adapter 的权重。假设一个 adapter 的 rank=16，对一个 4096×4096 的线性层，LoRA 权重大小为 `2 × 4096 × 16 × 2 bytes = 256 KB`。一个模型有几十个线性层，一个 adapter 就需要数 MB 到数十 MB。当 adapter 数量达到数百个时，不可能全部常驻 GPU。

### LRU 策略

vLLM 使用 LRU（Least Recently Used）策略管理 adapter 的加载和卸载：

:::diagram lora-lru-html
```html
<div class="cache-flow">
  <div class="cache-step">
    <div class="cache-step-num">1</div>
    <div>
      <div class="cache-step-title">请求到达，需要 adapter-C</div>
      <div class="cache-step-desc">检查 adapter-C 是否在 GPU 上</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">2</div>
    <div>
      <div class="cache-step-title">adapter-C 不在 GPU，且已达 max_loras</div>
      <div class="cache-step-desc">选择最久未使用的 adapter（如 adapter-A），卸载其权重</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">3</div>
    <div>
      <div class="cache-step-title">从磁盘加载 adapter-C 的权重到 GPU</div>
      <div class="cache-step-desc">safetensors 格式的权重文件通过 mmap 加载，然后拷贝到 GPU</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">4</div>
    <div>
      <div class="cache-step-title">更新 LRU 记录，adapter-C 标记为最近使用</div>
      <div class="cache-step-desc">后续请求使用 adapter-C 时不会触发卸载</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc lora-lru-html
Adapter LRU 管理流程：
1. 请求到达，需要 adapter-C，检查 adapter-C 是否在 GPU 上。
2. 如果 adapter-C 不在 GPU 且已达 max_loras 上限，选择最久未使用的 adapter（如 adapter-A）卸载其权重。
3. 从磁盘加载 adapter-C 的权重到 GPU（safetensors 格式，通过 mmap 加载后拷贝到 GPU）。
4. 更新 LRU 记录，adapter-C 标记为最近使用。

关键：adapter 的加载/卸载是显存管理操作，不影响 base model 权重。base model 权重始终驻留 GPU。
:::

关键点：**adapter 的加载/卸载只影响 LoRA 权重，不影响 base model 权重**。base model 权重始终驻留 GPU。

### LRU 数据结构

adapter 的 LRU 信息维护在 `LRUCache` 中，核心操作：

- **touch(adapter_id)**：将 adapter 标记为最近使用，移到 LRU 队列尾部
- **remove(adapter_id)**：从 LRU 队列中移除 adapter
- **evict()**：弹出 LRU 队列头部的 adapter_id（最久未用）

当 adapter 被卸载时，其 GPU 上的权重张量被释放，但磁盘上的权重文件保留，下次使用时重新加载。

## LoRA-aware CUDA Graph

### 问题：CUDA Graph 与动态 LoRA 的冲突

[CUDA Graph](term:将 GPU 操作序列记录为固定图结构，replay 时无需重新调度内核，消除 CPU launch overhead。) 要求操作序列和内存地址在捕获时固定。但 LoRA 引入了两个动态因素：

1. **是否使用 LoRA**：有些请求用 LoRA，有些不用，forward 路径不同
2. **活跃 adapter 数量**：每步可能有不同数量的 adapter 参与 SGMV 计算

### cudagraph_specialize_lora

vLLM 通过 `cudagraph_specialize_lora` 配置（默认 True）解决这个问题：**为不同的 LoRA 状态分别捕获 CUDA Graph**。

BatchDescriptor 包含两个字段：

- `has_lora`：是否有任何 token 使用 LoRA
- `num_active_loras`：当前活跃的 adapter 数量

当 `cudagraph_specialize_lora=True` 时，系统为 `has_lora=True` 和 `has_lora=False` 分别捕获图。对于 `has_lora=True` 的图，再按 `num_active_loras` 的不同值细分。

### LoRA 数量特化策略

为每个可能的 `num_active_loras` 都捕获一张图太浪费内存。vLLM 采用**2 的幂次 + max_loras+1**的策略：

:::diagram lora-cg-specialize-html
```html
<div class="cache-flow">
  <div class="cache-step">
    <div class="cache-step-num">1</div>
    <div>
      <div class="cache-step-title">确定捕获点：2 的幂次 + max_loras+1</div>
      <div class="cache-step-desc">max_loras=8 时，捕获 num_active_loras ∈ {1, 2, 4, 9} 的图</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">2</div>
    <div>
      <div class="cache-step-title">运行时：二分查找最小的捕获点 ≥ 当前 num_active_loras</div>
      <div class="cache-step-desc">当前有 3 个活跃 adapter → 选择 num_active_loras=4 的图</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">3</div>
    <div>
      <div class="cache-step-title">Replay 选中的图</div>
      <div class="cache-step-desc">多分配的 adapter 槽位用零权重填充，不影响计算结果</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc lora-cg-specialize-html
LoRA CUDA Graph 特化策略：
1. 确定捕获点：取 2 的幂次加上 max_loras+1。例如 max_loras=8 时，捕获 num_active_loras 为 {1, 2, 4, 9} 的图。
2. 运行时，对当前 num_active_loras 二分查找，选择最小的捕获点大于等于当前值。例如当前有 3 个活跃 adapter，选择 num_active_loras=4 的图。
3. Replay 选中的图。多分配的 adapter 槽位用零权重填充，因为 0 矩阵的乘法结果为 0，不影响 base model 的输出。

优势：用少量捕获点覆盖所有可能的 adapter 数量，避免内存爆炸。
:::

优势：用少量捕获点覆盖所有可能的 adapter 数量，避免内存爆炸。多分配的槽位用零权重填充——零矩阵的乘法结果为零，不影响 base model 的输出。

### 计算示例

以 `max_loras=4` 为例：

| 捕获点 | num_active_loras | 说明 |
|--------|-----------------|------|
| — | 0 | has_lora=False 的图（无 LoRA） |
| 1 | 1 | 2^0 |
| 2 | 2 | 2^1 |
| 3 | 5 | max_loras+1 = 4+1 |

运行时，若当前有 3 个活跃 adapter，二分查找找到 5 ≥ 3，replay num_active_loras=5 的图，多余 2 个槽位填零。

## LoRA 权重加载

### 权重格式与存储

LoRA 权重以 [safetensors](term:一种安全的张量序列化格式，支持零拷贝 mmap 加载，避免 pickle 的安全风险。) 格式存储在磁盘上，每个 adapter 对应一个目录，包含该 adapter 所有层的 A 和 B 矩阵。

### LoRA 请求结构

每个 LoRA 请求包含三个关键字段：

```python
@dataclass
class LoRARequest:
    lora_name: str
    lora_int_id: int
    lora_path: str
```

`lora_int_id` 是 SGMV kernel 用来路由 token 的标识符。0 保留给 base model（不使用 LoRA），正整数分配给各 adapter。

### 加载流程

:::diagram lora-load-html
```html
<div class="cache-flow">
  <div class="cache-step">
    <div class="cache-step-num">1</div>
    <div>
      <div class="cache-step-title">接收 LoRARequest，检查 adapter 是否已加载</div>
      <div class="cache-step-desc">通过 lora_int_id 查找已加载的 adapter 缓存</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">2</div>
    <div>
      <div class="cache-step-title">若未加载：触发 LRU 淘汰（如需要），释放 GPU 显存</div>
      <div class="cache-step-desc">卸载最久未用的 adapter 权重张量</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">3</div>
    <div>
      <div class="cache-step-title">从磁盘读取 safetensors 权重</div>
      <div class="cache-step-desc">mmap 方式加载，按 lora_dtype 转换精度</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">4</div>
    <div>
      <div class="cache-step-title">拷贝到 GPU，注册到 LoRA 层</div>
      <div class="cache-step-desc">每个 LinearWithLoRA 层获得对应的 A、B 张量引用</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc lora-load-html
LoRA 权重加载流程：
1. 接收 LoRARequest，通过 lora_int_id 查找已加载的 adapter 缓存，检查 adapter 是否已在 GPU 上。
2. 若未加载且需要淘汰，触发 LRU 策略卸载最久未用的 adapter 权重张量，释放 GPU 显存。
3. 从磁盘读取 safetensors 权重文件，使用 mmap 方式加载，按 lora_dtype 配置转换精度。
4. 将权重拷贝到 GPU，注册到每个 LinearWithLoRA 层，各层获得对应的 A、B 张量引用。

权重在 forward 时动态合并：y = W·x + B·A·x，不需要修改 base model 权重。
:::

### 权重精度

`lora_dtype` 配置控制 LoRA 权重的数据类型：

| lora_dtype | 行为 |
|------------|------|
| `"auto"` | 使用与 base model 相同的精度 |
| `"float16"` | 强制 fp16 |
| `"bfloat16"` | 强制 bf16 |
| `"float32"` | 强制 fp32（精度最高，显存开销大） |

使用比 base model 更低的精度可以减少 adapter 的显存占用，但可能影响推理质量。

### 运行时合并

LoRA 权重**不在加载时合并到 base model**，而是在每次 forward 时动态计算增量：

```
output = base_layer(input) + lora_A * lora_B * input
```

这样 base model 权重始终保持不变，多个 adapter 可以安全地共享同一个 base model，切换 adapter 只需更改 SGMV kernel 的路由表。

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `max_loras` | 1 | 单 GPU 上最大并发 LoRA adapter 数量 |
| `max_lora_rank` | 16 | 最大 LoRA 秩（rank），影响显存预分配 |
| `lora_dtype` | `"auto"` | LoRA 权重数据类型，auto 表示与 base model 一致 |
| `lora_extra_vocab_size` | 256 | LoRA 额外词汇表大小（adapter 扩展词表时使用） |
| `lora_max_long_lora_rank` | 0 | 长上下文 LoRA 的最大秩（0 表示禁用） |
| `cudagraph_specialize_lora` | True | 是否为 LoRA/非 LoRA 分别捕获 CUDA Graph |
| `enable_lora` | False | 是否启用 LoRA 支持（需显式开启） |
| `fully_sharded_loras` | False | 是否在 TP 间分片 LoRA 权重 |
| `max_cpu_loras` | None | CPU 上缓存的最大 adapter 数（None = 不限） |

`max_loras` 是最关键的配置：它决定了 GPU 上同时驻留的 adapter 数量上限，直接影响显存占用和并发能力。增大 `max_loras` 可以减少 adapter 换入换出的频率，但会占用更多显存。
