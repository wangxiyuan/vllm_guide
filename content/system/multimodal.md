---
id: multimodal
title: "多模态"
category: system
level: intermediate
status: draft
readingMinutes: 14
tags:
  - Multimodal
  - Vision Encoder
  - Placeholder
  - Encoder Cache
codeRefs:
  - vllm/multimodal/registry.py
  - vllm/multimodal/processing/base_processor.py
  - vllm/v1/core/encoder_cache_manager.py
heroText: "图像、音频、视频等多模态输入的完整处理路径：[encoder](term:多模态 encoder 模型（如 ViT、Whisper），将原始媒体编码为特征向量。) 编码 → placeholder 替换 → [encoder cache](term:缓存多模态 encoder 的输出特征向量，避免重复计算。) 管理 → 与 language model 的融合。"
---

## 心智模型

:::diagram mm-mental-model-html
```html
<div class="arch-diagram">
<div class="arch-row">
  <div class="arch-box">Image/Audio/Video</div>
  <div class="arch-arrow">→</div>
  <div class="arch-proc">
    <div class="arch-proc-title">Processor</div>
    <div class="arch-proc-sub">预处理 + Placeholder</div>
  </div>
  <div class="arch-arrow">→</div>
  <div class="arch-highlight">
    <div class="arch-module">Encoder</div>
    <div class="arch-module">ViT/Whisper</div>
  </div>
  <div class="arch-arrow">→</div>
  <div class="arch-box">Encoder Cache</div>
</div>
</div>
```
:::

:::diagram-desc mm-mental-model-html
多模态推理的关键路径：原始媒体输入（图像/音频/视频）→ Processor（预处理和 Placeholder 插入）→ Encoder（ViT/Whisper 等编码器，将媒体编码为特征向量）→ Encoder Cache（缓存 encoder 输出，避免重复计算）。最终特征向量在 LLM forward 时替换 placeholder 位置。
:::

多模态推理的关键：将媒体特征无缝注入 language model 的 token 序列，同时管理 encoder 计算开销。

## 多模态 Registry

[MultiModalRegistry](term:全局单例注册表，管理模型与多模态处理管线的映射关系。) 是核心调度机制，通过 `register_processor` 装饰器将模型类与三组件绑定：

| 组件 | 职责 |
|------|------|
| `ProcessingInfo` | 模型特定知识：支持的模态、最大 token 数 |
| `MultiModalProcessor` | 处理管线：HF Processor → placeholder → 特征提取 |
| `DummyInputsBuilder` | 生成最坏情况输入，用于显存 profiling |

```python
@MULTIMODAL_REGISTRY.register_processor(
    info=Qwen2VLProcessingInfo,
    processor=Qwen2VLMultiModalProcessor,
    dummy_inputs=Qwen2VLDummyInputsBuilder,
)
class Qwen2VLForConditionalGeneration(nn.Module, SupportsMultiModal):
    ...
```

构建链：`InputProcessingContext` → `BaseProcessingInfo` → `BaseDummyInputsBuilder` → `BaseMultiModalProcessor`

`vllm/multimodal/registry.py`

## Processor Pipeline

三步处理流程：

:::steps id=mm-pipeline-steps
### 1. HF Processor
对 prompt 文本 + 多模态数据调用 HuggingFace Processor，生成 token IDs 和处理后的张量。

### 2. Placeholder 替换
在 token IDs 中找到占位符（如 `<image>`），替换为与特征尺寸匹配的 placeholder token 数量。

### 3. Placeholder 信息提取
从处理后的 token IDs 中提取 placeholder 位置信息（offset、length），用于后续特征注入。
:::

### 增量处理与缓存

为避免重复处理相同的媒体输入，Processor 支持增量处理：

1. 计算所有媒体项的 hash
2. 检查缓存，只处理**未命中**的项
3. 合并缓存结果和新处理结果

缓存分两级：
- **P0（API 进程）**：`MultiModalProcessorOnlyCache`，单进程 LRU
- **P1（Core 进程）**：`MultiModalReceiverCache`，IPC 镜像 LRU

`vllm/multimodal/processing/processor.py`

## Placeholder 机制

[PlaceholderRange](term:记录多模态特征在 token 序列中位置的数据结构，包含 offset、length 和 is_embed 掩码。) 是核心数据结构：

```python
@dataclass(frozen=True)
class PlaceholderRange:
    offset: int                    # 在 prompt 中的起始位置
    length: int                    # placeholder token 数量
    is_embed: torch.Tensor | None  # 布尔掩码，标记哪些位置是 embedding
```

### 两种替换模式

| 模式 | 说明 | 示例 |
|------|------|------|
| `PromptReplacement` | 替换 `<image>` 为 N 个 placeholder | `<image>` → `<image_pad>` × N |
| `PromptInsertion` | 在指定位置插入 token | 在 prompt 开头插入特殊 token |

### 部分嵌入控制

某些模型（如 Qwen2-VL）的 placeholder 包含**文本 token** 和 **特征 token** 的混合：

```python
# Qwen2-VL: <image_bos> + <image_pad>×N + <image_eos>
PromptUpdateDetails(
    full=[image_bos_id] + [image_pad_id] * N + [image_eos_id],
    is_embed=select_token_id(full, image_pad_id),  # 只有 image_pad 是 embedding
)
```

`is_embed` 掩码精确控制哪些位置接收 encoder 输出的特征，哪些保留原始 token embedding。

`vllm/multimodal/inputs.py`

## Encoder Cache 管理

调度器级的 [EncoderCacheManager](term:管理多模态 encoder 输出的 GPU 显存分配与回收，通过 mm_hash 索引缓存条目。) 控制缓存空间：

### 缓存生命周期

| 操作 | 说明 |
|------|------|
| `check_and_update_cache()` | 检查 mm_hash 是否已缓存，命中则增加引用计数 |
| `can_allocate()` | 检查是否有足够空间，不够则 FIFO 驱逐最旧未引用条目 |
| `free_encoder_input()` | 释放引用，标记为可驱逐（但数据仍在显存中） |

关键设计：释放时数据**不立即清除**，只在显存紧张时驱逐。新请求仍可命中刚释放的缓存。

```python
# 驱逐逻辑：FIFO 淘汰最旧的无引用条目
while num_embeds > self.num_free_slots:
    mm_hash, num_free = self.freeable.popitem(last=False)
    del self.cached[mm_hash]
    self.freed.append(mm_hash)
```

`vllm/v1/core/encoder_cache_manager.py`

## Encoder Compute Budget

控制 encoder 计算资源，防止阻塞 LLM 推理：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `encoder_compute_budget` | = max_num_batched_tokens | 单步最大 encoder token 数 |
| `encoder_cache_size` | = max_num_batched_tokens | encoder 输出缓存容量 |

[MultiModalBudget](term:计算多模态 encoder 的计算预算和缓存容量，确保单个媒体项能被完整处理。) 根据模型配置计算每种模态的最大项数，约束条件包括：

- encoder 预算
- decoder 预算
- 用户限制
- 模型最大序列长度

`vllm/multimodal/encoder_budget.py`

## CUDA Graph 兼容

多模态输入对 CUDA Graph 捕获的影响与处理策略：

| 挑战 | 解决方案 |
|------|----------|
| 动态 token 数量 | Encoder 在 CG **外部**运行，仅 decoder 被 CG 捕获 |
| Placeholder 动态 shape | 特征注入在 CG replay 之前完成 |
| 部分调度 MM 输入 | `disable_chunked_mm_input` 禁止拆分 MM 项 |
| 显存 profiling | DummyInputsBuilder 使用最坏情况 MM 输入 |

`disable_chunked_mm_input=True` 时，文本 token 先调度，完整 MM 项在下一步调度，避免部分 encoder 计算。

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `limit_mm_per_prompt` | `{}` | 每种模态每 prompt 最大项数 |
| `mm_processor_kwargs` | `None` | 传递给 HF Processor 的额外参数 |
| `mm_processor_cache_gb` | 4 | Processor 缓存大小（GiB） |
| `mm_processor_cache_type` | `"lru"` | 缓存类型：`lru` 或 `shm` |
| `disable_chunked_mm_input` | False | 禁止拆分 MM 项跨调度步骤 |
| `mm_encoder_tp_mode` | `"weights"` | Encoder TP 策略 |
| `skip_mm_profiling` | False | 跳过 MM 显存 profiling 加速启动 |

`vllm/config/multimodal.py`
