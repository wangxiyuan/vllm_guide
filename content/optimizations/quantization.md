---
id: quantization
title: 量化
category: optimizations
level: advanced
status: draft
readingMinutes: 14
tags:
  - Quantization
  - FP8
  - INT4
codeRefs:
  - vllm/model_executor/layers/quantization/fp8.py
  - vllm/model_executor/layers/quantization/__init__.py
heroText: "用低精度数值表示权重和/或激活值，[FP8](term:8-bit 浮点格式，E4M3 或 E5M2，权重和 KV cache 可用 FP8 存储以节省显存。) / INT4 / INT8 等方案减少显存占用和加速推理。"
---

## 心智模型

想象你在压缩一张高分辨率照片。你会丢失一些细节（精度），但文件变小了（显存减少），加载也更快了（带宽减少）。不同的压缩格式（量化方法）在画质和压缩率之间做不同的权衡。

:::diagram quant-mental-model
```html
<div class="diagram-container">
  <div class="diagram" data-ref="vllm/model_executor/layers/quantization/__init__.py">
    <div class="diagram-title">量化：精度与显存的权衡</div>
    <div class="comm-layers">
      <div class="comm-layer">
        <div class="comm-layer-title">高精度 → 低精度</div>
        <div class="comm-layer-items">
          <div class="comm-item">FP32 — 32 bit — 无损</div>
          <div class="comm-item">FP16/BF16 — 16 bit — 几乎无损</div>
          <div class="comm-item">FP8 — 8 bit — 轻微损失</div>
          <div class="comm-item">INT8 — 8 bit — 中等损失</div>
          <div class="comm-item">INT4 — 4 bit — 较大损失</div>
        </div>
        <div class="comm-layer-note">精度下降 → 显存减少 → 推理加速</div>
      </div>
      <div class="comm-layer">
        <div class="comm-layer-title">量化收益</div>
        <div class="comm-layer-items">
          <div class="comm-item">显存占用 ↓</div>
          <div class="comm-item">推理速度 ↑</div>
          <div class="comm-item">部署成本 ↓</div>
        </div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc quant-mental-model
量化精度与显存的权衡图展示了不同量化方案的特性：

**FP32**（32 bit）：无损精度，但显存占用最大。
**FP16/BF16**（16 bit）：几乎无损，是模型训练和推理的默认格式。
**FP8**（8 bit）：轻微精度损失，显存减半，是目前推理量化的主流选择。
**INT8**（8 bit）：中等精度损失，适合权重和激活量化。
**INT4**（4 bit）：较大精度损失，但显存压缩比最高（4x）。

核心权衡：精度下降换来显存减少和推理加速。选择取决于模型类型、硬件支持和精度容忍度。
:::

## 量化架构总览

vLLM 的量化系统采用三层架构，从配置到执行层层委托：

:::diagram quant-arch
```html
<div class="diagram-container">
  <div class="diagram" data-ref="vllm/model_executor/layers/quantization/__init__.py">
    <div class="diagram-title">量化架构总览</div>
    <div class="engine-step-flow">
      <div class="engine-step">
        <div class="engine-step-num">1</div>
        <div class="engine-step-content">
          <div class="engine-step-title">QuantizationConfig</div>
          <div class="engine-step-desc">从 checkpoint 解析量化配置</div>
        </div>
      </div>
      <div class="engine-step">
        <div class="engine-step-num">2</div>
        <div class="engine-step-content">
          <div class="engine-step-title">QuantizeMethodBase</div>
          <div class="engine-step-desc">量化策略的抽象基类</div>
        </div>
      </div>
      <div class="engine-step">
        <div class="engine-step-num">3</div>
        <div class="engine-step-content">
          <div class="engine-step-title">三条实现路径</div>
          <div class="engine-step-desc">LinearMethodBase (Linear 层) / FusedMoEMethodBase (MoE 层) / BaseKVCacheMethod (KV Cache)</div>
        </div>
      </div>
    </div>
    <div class="kv-lifecycle">
      <div class="kv-lc-state" data-state="used">create_weights()</div>
      <div class="kv-lc-arrow">→</div>
      <div class="kv-lc-state" data-state="cached-used">apply()</div>
      <div class="kv-lc-arrow">→</div>
      <div class="kv-lc-state" data-state="cached-free">process_weights_after_loading()</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc quant-arch
量化架构总览展示了 vLLM 量化系统的分层设计：

**第 1 层 - QuantizationConfig**：从模型 checkpoint 中解析量化配置，包含量化方法名称、参数等信息。

**第 2 层 - QuantizeMethodBase**：量化策略的抽象基类，定义了 create_weights()、apply()、process_weights_after_loading() 等接口。

**第 3 层 - 三条实现路径**：
- **LinearMethodBase**：处理 Linear 层的权重和激活量化
- **FusedMoEMethodBase**：处理 MoE（混合专家）层的量化
- **BaseKVCacheMethod**：处理 KV cache 的量化

**生命周期**：create_weights()（创建量化参数）→ apply()（执行量化计算）→ process_weights_after_loading()（加载后处理，如重打包权重）。
:::

量化方法的执行生命周期：`create_weights()` → `apply()` → `process_weights_after_loading()`。其中 `process_weights_after_loading()` 在模型权重加载完成后调用，用于权重重打包等后处理。

## 支持的量化方法

vLLM 支持多种量化方法，涵盖权重、激活和 KV cache 量化：

| 方法 | 关键特性 | 权重位数 |
|------|---------|---------|
| **fp8** | FP8 权重，支持 static/dynamic 激活量化 | 8 |
| **awq** | 4-bit 仅权重量化，分组量化 | 4 |
| **gptq / auto_gptq** | GPTQ 算法，Marlin kernel 加速 | 4 |
| **awq_marlin** | AWQ 重打包为 Marlin 格式 | 4 |
| **modelopt** | NVIDIA ModelOptimizer FP8 | 8 |
| **modelopt_fp4** | NVIDIA FP4 | 4 |
| **gguf** | GGUF 格式，支持多种子格式 | 2-8 |
| **compressed-tensors** | Neural Magic 压缩格式 | 2-8 |
| **bitsandbytes** | bitsandbytes 量化 | 4/8 |
| **experts_int8** | MoE 专家 INT8 量化 | 8 |
| **quark** | AMD 量化方案 | 4/8 |
| **moe_wna16** | MoE 权重量化，激活 FP16 | 4/8 |
| **torchao** | PyTorch 原生量化 | 4/8 |
| **inc** | Intel / AutoRound 量化 | 4/8 |
| **mxfp4** | MX 格式 FP4 | 4 |
| **deepseek_v3_fp8** | DeepSeek V3 专用 FP8 | 8 |
| **online** | 运行时在线量化 | 可变 |

## FP8 详解

FP8 是目前 vLLM 推理量化的主流选择，使用 8-bit 浮点格式（E4M3 或 E5M2）存储权重和 KV cache。

### 配置

```python
# vllm/model_executor/layers/quantization/fp8.py
@dataclass
class Fp8Config:
    activation_scheme: str          # "static" 或 "dynamic"
    weight_block_size: Optional[List[int]]  # 块级量化大小
    is_checkpoint_fp8_serialized: bool      # checkpoint 是否已 FP8 序列化
```

- **activation_scheme**：`static` 使用预计算的缩放因子，`dynamic` 在运行时计算
- **weight_block_size**：块级量化时每组权重的大小，None 表示逐张量量化
- **is_checkpoint_fp8_serialized**：如果为 True，直接加载 FP8 权重，无需在线量化

### 计算内核

`Fp8LinearMethod` 根据硬件和配置选择不同的内核：

| 内核 | 条件 | 特点 |
|------|------|------|
| `torch._scaled_mm` (Cutlass) | 默认 | PyTorch 原生 FP8 GEMM |
| DeepGEMM | SM90+，块级量化 | 优化的块级 FP8 GEMM |
| Marlin | 权重重打包 | 高吞吐量 FP8 推理 |

支持**逐张量**和**块级**两种量化粒度。块级量化精度更高，但需要额外的缩放因子存储。

## GPTQ/AWQ 详解

GPTQ 和 AWQ 是两种主流的 4-bit 仅权重量化方案，都使用 Marlin kernel 进行高效的 GPU 推理。

### 核心特点

- **仅权重量化**：权重 4-bit，激活保持 FP16/BF16
- **分组量化**：每 `group_size` 个权重共享一组缩放因子（常见 group_size=128）
- **Marlin kernel**：重打包权重后使用 Marlin kernel，比通用 GEMM 快数倍
- **动态覆盖**：支持逐层覆盖量化配置

```python
# vllm/model_executor/layers/quantization/awq.py
# AWQ 量化权重结构
# qweight: [num_groups, group_size // 8, intermediate_size]
# scales:  [num_groups, 1, intermediate_size]
# qzeros: [num_groups, 1, intermediate_size // pack_factor]
```

AWQ Marlin 是 AWQ 的优化变体，将 AWQ 权重重打包为 Marlin 格式，获得更好的 kernel 性能。

## KV Cache 量化

KV cache 量化独立于权重量化，可以使用更低的精度存储 KV 向量：

| 配置 | 格式 | 说明 |
|------|------|------|
| `kv_cache_dtype="fp8_e4m3"` | E4M3 | 更高精度，适合大部分场景 |
| `kv_cache_dtype="fp8_e5m2"` | E5M2 | 更大动态范围，适合极端值 |

### 量化流程

```python
# vllm/_custom_ops.py
reshape_and_cache_flash(
    key, value,
    key_cache, value_cache,
    slot_mapping,
    kv_cache_dtype,       # "fp8_e4m3" 或 "fp8_e5m2"
    k_scale, v_scale,     # 逐 token-head 缩放因子
)
```

写入 KV cache 时，`reshape_and_cache_flash` kernel 会在写入的同时完成 FP16→FP8 的量化转换。支持**逐 token-head 量化**，每个 token 的每个 head 有独立的缩放因子。

KV cache 量化可将 KV cache 显存占用**减半**（FP16 → FP8），对于长上下文场景收益显著。

## 量化方法注册

vLLM 支持自定义量化方法的注册：

```python
# vllm/model_executor/layers/quantization/__init__.py
def register_quantization_config(name: str, config_cls: type):
    _CUSTOM_QUANTIZATION_CONFIG_REGISTRY[name] = config_cls
```

注册后可通过 `--quantization <name>` 使用。内置方法通过延迟导入（lazy import）加载，避免不必要的依赖。

### 在线量化快捷方式

无需提前量化模型，运行时自动量化：

| 快捷方式 | 等效配置 |
|---------|---------|
| `fp8_per_tensor` | FP8，逐张量量化 |
| `fp8_per_block` | FP8，块级量化 |
| `int8_per_channel_weight_only` | INT8，逐通道仅权重 |
| `mxfp8` | MX 格式 FP8 |

## 关键配置

| 参数 | 默认值 | 说明 | 源码 |
|------|--------|------|------|
| `--quantization` | None | 量化方法名称（如 fp8, awq, gptq） | quantization_config.py |
| `--kv-cache-dtype` | auto | KV cache 数据类型（auto/fp8_e4m3/fp8_e5m2） | cache_config.py |
| `--gpu-memory-utilization` | 0.92 | GPU 显存利用率 | cache_config.py |
| `--quantization-param-path` | None | 外部量化参数文件路径 | quantization_config.py |
