---
id: model-runner
title: Model Runner
category: execution
level: advanced
status: draft
readingMinutes: 16
tags:
  - ModelRunner
  - Forward Pass
  - Input Preparation
  - Attention Metadata
  - CUDA Graph
codeRefs:
  - vllm/v1/worker/gpu_model_runner.py
  - vllm/v1/worker/gpu/model_runner.py
heroText: "调度层与 GPU 执行之间的桥梁：[GPUModelRunner](term:负责准备输入张量、构建 attention metadata、执行 forward pass、采样与 draft proposal 的核心执行组件。) 管理 input batch、forward pass、采样与 draft proposal 的完整执行流程。"
---

## 心智模型

GPUModelRunner 是调度层与 GPU 之间的**唯一执行入口**。核心职责：接收 `SchedulerOutput`，将调度决策转化为 GPU 可执行的张量与元数据，驱动 forward → sample → draft proposal 流程。

:::diagram model-runner-html
```html
<div class="engine-step-flow">
  <div class="engine-step">
    <div class="engine-step-num">1</div>
    <div class="engine-step-content">
      <div class="engine-step-title">_prepare_inputs</div>
      <div class="engine-step-desc">SchedulerOutput → input_ids、positions、block tables</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">2</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Attention Metadata</div>
      <div class="engine-step-desc">构建 slot mapping、seq_lens、query_start_loc</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">3</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Forward + Sample</div>
      <div class="engine-step-desc">模型前向计算 → 采样 → draft proposal</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc model-runner-html
GPUModelRunner 的三步核心流程：
1. _prepare_inputs — 将 SchedulerOutput 翻译为 GPU 张量：input_ids、positions、block tables。
2. Attention Metadata — 构建 PagedAttention 所需的 slot mapping、seq_lens、query_start_loc 等元数据。
3. Forward + Sample — 执行模型前向计算，采样生成 token，投机解码模式下还需生成 draft proposal。
:::

## ModelRunner 生命周期

从初始化到稳态运行的完整生命周期。

:::diagram mr-lifecycle-html
```html
<div class="cache-flow">
  <div class="cache-step">
    <div class="cache-step-num">1</div>
    <div class="cache-step-content">
      <div class="cache-step-title">__init__</div>
      <div class="cache-step-desc">存储配置，创建 Sampler、speculative drafter、InputBatch、持久 GPU 缓冲区、CudagraphDispatcher</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">2</div>
    <div class="cache-step-content">
      <div class="cache-step-title">load_model</div>
      <div class="cache-step-desc">model_loader.load_model() 加载主模型，speculator.load_model() 加载投机模型</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">3</div>
    <div class="cache-step-content">
      <div class="cache-step-title">warmup</div>
      <div class="cache-step-desc">prefill step + decode step 的 dummy 数据执行，确定最大 batch size</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">4</div>
    <div class="cache-step-content">
      <div class="cache-step-title">capture</div>
      <div class="cache-step-desc">CUDA Graph 捕获：为所有 batch size 区间录制固定操作图</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">5</div>
    <div class="cache-step-content">
      <div class="cache-step-title">稳态 execute_model</div>
      <div class="cache-step-desc">反复调用 execute_model，每步完成 input 准备 → forward → sample → draft</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc mr-lifecycle-html
ModelRunner 生命周期分五个阶段：
1. __init__ — 存储配置，创建 Sampler、speculative drafter、InputBatch、持久 GPU 缓冲区、CudagraphDispatcher。
2. load_model — 通过 model_loader.load_model() 加载主模型，speculator.load_model() 加载投机模型。
3. warmup — 用 dummy 数据执行一次 prefill step 和一次 decode step，确定最大 batch size。
4. capture — CUDA Graph 捕获，为所有 batch size 区间录制固定操作图。
5. 稳态 — 反复调用 execute_model，每步完成 input 准备 → forward → sample → draft。
:::

### Drafter 类型

投机解码的 drafter 支持多种实现：

| Drafter | 说明 |
|---------|------|
| `ngram` | 基于 n-gram 的轻量 draft，无需额外模型 |
| `eagle` | EAGLE 投机解码，基于最后一层 hidden states |
| `eagle3` | EAGLE v3，改进的 draft 策略 |
| `medusa` | Medusa 多头 draft，每个 head 独立预测 |
| `draft_model` | 独立小模型作为 drafter |
| `dflash` | Draft Flash，优化的 draft 方案 |
| `suffix` | Suffix decoding，基于后缀匹配 |
| `gemma4_mtp` | Gemma4 多 token 预测 |

`vllm/v1/worker/gpu_model_runner.py`

## Input Preparation（_prepareInputs）

将 `SchedulerOutput` 翻译为 GPU 张量，是 ModelRunner 中最复杂的预处理步骤。

:::diagram mr-prepare-inputs-html
```html
<div class="cache-flow">
  <div class="cache-step">
    <div class="cache-step-num">1</div>
    <div class="cache-step-content">
      <div class="cache-step-title">Block table commit</div>
      <div class="cache-step-desc">异步拷贝 block table 到 GPU（cudaMemcpyAsync）</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">2</div>
    <div class="cache-step-content">
      <div class="cache-step-title">Request indices</div>
      <div class="cache-step-desc">np.repeat 按每个请求的 num_scheduled_tokens 展开请求索引</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">3</div>
    <div class="cache-step-content">
      <div class="cache-step-title">Positions</div>
      <div class="cache-step-desc">num_computed_tokens[req_indices] + query_pos，得到每个 token 的绝对位置</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">4</div>
    <div class="cache-step-content">
      <div class="cache-step-title">Input IDs</div>
      <div class="cache-step-desc">torch.index_select 从 InputBatch 中按索引提取 token IDs</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">5</div>
    <div class="cache-step-content">
      <div class="cache-step-title">query_start_loc / seq_lens</div>
      <div class="cache-step-desc">query_start_loc = cumsum(num_scheduled_tokens)；seq_lens = num_computed + num_scheduled</div>
    </div>
  </div>
  <div class="cache-step">
    <div class="cache-step-num">6</div>
    <div class="cache-step-content">
      <div class="cache-step-title">slot_mapping</div>
      <div class="cache-step-desc">block_table.compute_slot_mapping() 计算每个 token 在 KV cache 中的写入位置</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc mr-prepare-inputs-html
_prepareInputs 的六步流程：
1. Block table commit — 异步拷贝 block table 到 GPU。
2. Request indices — 用 np.repeat 按每个请求的 num_scheduled_tokens 展开请求索引。
3. Positions — 计算 num_computed_tokens[req_indices] + query_pos，得到每个 token 的绝对位置。
4. Input IDs — 用 torch.index_select 从 InputBatch 中按索引提取 token IDs。
5. query_start_loc / seq_lens — query_start_loc 是 num_scheduled_tokens 的累积和；seq_lens = num_computed + num_scheduled。
6. slot_mapping — 通过 block_table.compute_slot_mapping() 计算每个 token 在 KV cache 中的写入位置。
:::

### logits_indices

采样时只需要每个请求**最后一个 token** 的 logits（decode 阶段），或投机解码位置对应的 logits。`logits_indices` 就是这些位置在展平张量中的索引。

`vllm/v1/worker/gpu_model_runner.py`

## Attention Metadata 构建

PagedAttention 正确执行的关键元数据，由 `_build_attention_metadata()` 构建。

### CommonAttentionMetadata 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `query_start_loc` | Tensor | 每个 query 的起始位置，cumsum(num_scheduled_tokens) |
| `seq_lens` | Tensor | 每个请求的完整序列长度 = num_computed + num_scheduled |
| `num_reqs` | int | 当前 batch 中的请求数 |
| `num_actual_tokens` | int | 实际 token 总数 |
| `max_query_len` | int | 最大 query 长度（prefill >1，decode =1） |
| `max_seq_len` | int | 最大完整序列长度 |
| `block_table_tensor` | Tensor | 请求 → KV block 的映射表 |
| `slot_mapping` | Tensor | 每个 token 在 KV cache 中的物理写入位置 |

### Prefill vs Decode 差异

| 属性 | Prefill | Decode |
|------|---------|--------|
| `max_query_len` | > 1（可能很长） | = 1（每请求只生成 1 token） |
| `query_start_loc` | 间隔大（每个请求多个 token） | 相邻（每个请求 1 token） |
| `is_prefilling` | True | False |

:::diagram mr-attn-meta-html
```html
<div class="pa-metadata-pipeline">
  <div class="pa-mp-step">
    <div class="pa-mp-box pa-mp-input">num_scheduled_tokens</div>
    <div class="pa-mp-desc">每个请求本步要处理的 token 数</div>
  </div>
  <div class="pa-mp-arrow">→</div>
  <div class="pa-mp-step">
    <div class="pa-mp-box pa-mp-builder">cumsum</div>
    <div class="pa-mp-desc">累积和得到 query_start_loc</div>
  </div>
  <div class="pa-mp-arrow">→</div>
  <div class="pa-mp-step">
    <div class="pa-mp-box pa-mp-builder">+ num_computed</div>
    <div class="pa-mp-desc">加上已计算 token 数得到 seq_lens</div>
  </div>
  <div class="pa-mp-arrow">→</div>
  <div class="pa-mp-step">
    <div class="pa-mp-box pa-mp-output">CommonAttentionMetadata</div>
    <div class="pa-mp-desc">完整的 attention 元数据对象</div>
  </div>
</div>
```
:::

:::diagram-desc mr-attn-meta-html
Attention Metadata 构建流水线：从每个请求的 num_scheduled_tokens 出发，通过 cumsum 得到 query_start_loc，加上 num_computed_tokens 得到 seq_lens，最终组装为 CommonAttentionMetadata 对象。
:::

`vllm/v1/worker/gpu_model_runner.py`

## Forward Pass

核心前向计算：输入送入模型，获取 logits。`execute_model` 的完整流程如下：

:::diagram mr-forward-html
```html
<div class="engine-step-flow">
  <div class="engine-step">
    <div class="engine-step-num">1</div>
    <div class="engine-step-content">
      <div class="engine-step-title">_update_states</div>
      <div class="engine-step-desc">更新请求状态，处理上一步的输出与新调度信息</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">2</div>
    <div class="engine-step-content">
      <div class="engine-step-title">MM Encoder</div>
      <div class="engine-step-desc">多模态请求执行 vision encoder，缓存结果</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">3</div>
    <div class="engine-step-content">
      <div class="engine-step-title">_prepare_inputs</div>
      <div class="engine-step-desc">构建 input_ids、positions、block tables 等张量</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">4</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Batch 执行决策</div>
      <div class="engine-step-desc">确定 cudagraph_mode、batch_desc，选择 FULL/PIECEWISE/NONE</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">5</div>
    <div class="engine-step-content">
      <div class="engine-step-title">_build_attention_metadata</div>
      <div class="engine-step-desc">构建 PagedAttention 所需的完整元数据</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">6</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Preprocess</div>
      <div class="engine-step-desc">input_ids → inputs_embeds，positions 编码</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">7</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Model Forward</div>
      <div class="engine-step-desc">set_forward_context() 下执行模型前向，输出 hidden states</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">8</div>
    <div class="engine-step-content">
      <div class="engine-step-title">compute_logits</div>
      <div class="engine-step-desc">从 sample_hidden_states 计算 logits</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc mr-forward-html
Forward Pass 的八步流程：
1. _update_states — 更新请求状态，处理上一步的输出与新调度信息。
2. MM Encoder — 多模态请求执行 vision encoder，缓存结果。
3. _prepare_inputs — 构建 input_ids、positions、block tables 等张量。
4. Batch 执行决策 — 确定 cudagraph_mode、batch_desc，选择 FULL/PIECEWISE/NONE。
5. _build_attention_metadata — 构建 PagedAttention 所需的完整元数据。
6. Preprocess — input_ids 转为 inputs_embeds，positions 编码。
7. Model Forward — 在 set_forward_context() 上下文下执行模型前向，输出 hidden states。
8. compute_logits — 从 sample_hidden_states 计算 logits。
:::

### set_forward_context

模型前向执行时，通过 `set_forward_context()` 将 attention metadata、KV cache 等信息注入全局上下文，使模型各层能访问到这些数据而无需逐层传递。

`vllm/v1/worker/gpu_model_runner.py`

## 采样与 Draft Proposal

Forward 后立即采样生成 token；投机解码模式下还需生成 draft proposal。

### 无投机解码

标准路径：`sampler(logits, sampling_metadata)` → 直接采样输出 token。

### 有投机解码

投机路径：先由 drafter 生成 draft tokens，再由 rejection sampler 验证：

```python
draft_probs = drafter.propose()
rejection_sampler(spec_decode_metadata, draft_probs, logits, sampling_metadata)
```

:::diagram mr-sample-draft-html
```html
<div class="pa-metadata-pipeline">
  <div class="pa-mp-step">
    <div class="pa-mp-box pa-mp-input">logits</div>
    <div class="pa-mp-desc">模型前向输出</div>
  </div>
  <div class="pa-mp-arrow">→</div>
  <div class="pa-mp-step">
    <div class="pa-mp-box pa-mp-builder">Sampler</div>
    <div class="pa-mp-desc">标准采样生成 verify token</div>
  </div>
  <div class="pa-mp-arrow">→</div>
  <div class="pa-mp-step">
    <div class="pa-mp-box pa-mp-output">ModelRunnerOutput</div>
    <div class="pa-mp-desc">采样结果封装</div>
  </div>
</div>
<div class="pa-metadata-pipeline" style="margin-top: 12px;">
  <div class="pa-mp-step">
    <div class="pa-mp-box pa-mp-input">drafter.propose()</div>
    <div class="pa-mp-desc">生成 draft tokens + draft_probs</div>
  </div>
  <div class="pa-mp-arrow">→</div>
  <div class="pa-mp-step">
    <div class="pa-mp-box pa-mp-builder">Rejection Sampler</div>
    <div class="pa-mp-desc">按概率比验证 draft token</div>
  </div>
  <div class="pa-mp-arrow">→</div>
  <div class="pa-mp-step">
    <div class="pa-mp-box pa-mp-output">accepted / rejected</div>
    <div class="pa-mp-desc">接受或拒绝的 draft token</div>
  </div>
</div>
```
:::

:::diagram-desc mr-sample-draft-html
采样与 Draft Proposal 的两条路径：

标准路径：logits → Sampler → ModelRunnerOutput，直接采样生成 token。

投机解码路径：drafter.propose() 生成 draft tokens 和 draft_probs → Rejection Sampler 按概率比验证 draft token → 输出 accepted/rejected 结果。
:::

### 采样结果封装

采样结果封装为 `ModelRunnerOutput`，包含：

| 字段 | 说明 |
|------|------|
| `sampled_token_ids` | 每个请求采样的 token IDs |
| `logprobs` | 采样 token 的对数概率 |
| `prompt_logprobs` | prompt token 的对数概率 |
| `spec_token_ids` | 投机解码接受的 draft token IDs |
| `num_spec_tokens_accepted` | 每个请求接受的 draft token 数 |

`vllm/v1/worker/gpu_model_runner.py`

## CUDA Graph Dispatch

ModelRunner 内部触发 CUDA Graph replay，根据 batch size 选择已捕获图。

### CudagraphDispatcher

`CudagraphDispatcher.dispatch()` 按优先级尝试三种模式：

| 模式 | 条件 | 说明 |
|------|------|------|
| FULL | batch size 匹配已捕获图 | 整个 forward 作为一张图 replay |
| PIECEWISE | attention 与非 attention 分离捕获 | 应对 attention 的动态 shape |
| NONE | 无匹配图或 prefill 阶段 | 退回 eager 执行 |

### 执行决策

`_determine_batch_execution_and_padding` 根据当前 batch 的特征决定使用哪种模式：

- **Prefill 阶段**：始终 NONE（eager），因为 query 长度不固定
- **Decode 阶段**：优先 FULL，fallback PIECEWISE，最终 NONE
- **V2 架构**：使用 `ModelCudaGraphManager` 管理

### Replay 时的数据搬运

Replay 时只需将输入张量 copy 到捕获时固定的输入缓冲区，然后一次 `graph.replay()` 执行整张图，最后从输出缓冲区读取结果。CPU 端只需一次 launch 调用，消除了逐个 kernel 调度的开销。

`vllm/v1/worker/gpu_model_runner.py`

## 多模态 Encoder 执行

多模态请求需在 LLM forward 前先执行 vision encoder。

### 执行时机

MM encoder 在 `_preprocess` 阶段执行，**早于 LLM forward**。这保证了 LLM forward 时所有多模态 embedding 已经就绪。

:::diagram mr-mm-encoder-html
```html
<div class="engine-step-flow">
  <div class="engine-step">
    <div class="engine-step-num">1</div>
    <div class="engine-step-content">
      <div class="engine-step-title">_execute_mm_encoder</div>
      <div class="engine-step-desc">批量收集 MM inputs，执行 encoder，输出 GPU tensor</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">2</div>
    <div class="engine-step-content">
      <div class="engine-step-title">EncoderCache</div>
      <div class="engine-step-desc">mm_hash → GPU tensor，内容寻址缓存</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">3</div>
    <div class="engine-step-content">
      <div class="engine-step-title">LLM Forward</div>
      <div class="engine-step-desc">从 cache 取出 embedding，替换对应 token 的 inputs_embeds</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc mr-mm-encoder-html
多模态 Encoder 执行流程：
1. _execute_mm_encoder — 批量收集多模态输入，执行 encoder，输出 GPU tensor。
2. EncoderCache — 以 mm_hash 为键缓存 GPU tensor，实现内容寻址（相同输入直接复用）。
3. LLM Forward — 从 cache 取出 embedding，替换对应 token 的 inputs_embeds。
:::

### EncoderCache

EncoderCache 使用**内容寻址**策略：以 `mm_hash`（多模态输入的哈希值）为键，GPU tensor 为值。相同的多模态输入（如同一张图片）只需计算一次，后续请求直接从 cache 取用。

- **缓存命中**：跳过 encoder 计算，直接返回已缓存的 tensor
- **缓存未命中**：执行 encoder，将结果存入 cache
- **缓存释放**：请求完成后，通过 `free_encoder_mm_hashes` 释放不再需要的缓存项

`vllm/v1/worker/gpu_model_runner.py`

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--enforce-eager` | False | 禁用 CUDA Graph，强制 eager 模式 |
| `--max-num-seqs` | 128 | 最大并发序列数 |
| `--max-model-len` | 模型默认 | 最大序列长度 |
| `--gpu-memory-utilization` | 0.9 | GPU 显存利用率上限 |

`vllm/v1/worker/gpu_model_runner.py`
