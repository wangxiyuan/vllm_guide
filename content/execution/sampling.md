---
id: sampling
title: 采样与 Logits 处理
category: execution
level: intermediate
status: draft
readingMinutes: 14
tags:
  - Sampling
  - Temperature
  - Top-k
  - Top-p
  - Min-p
  - Penalties
  - Rejection Sampler
codeRefs:
  - vllm/v1/sample/sampler.py
  - vllm/v1/sample/ops/topk_topp_sampler.py
  - vllm/v1/sample/logits_processor/builtin.py
heroText: "从 logits 到 token 的完整路径：[temperature](term:控制采样概率分布的平坦程度，温度越低越倾向于高概率 token。) / top-k / top-p / min-p 采样策略、重复惩罚、bad words 过滤与 [rejection sampler](term:投机解码中验证 draft token 的核心模块，按概率比决定接受或拒绝。) 的协作。"
---

## 心智模型

采样的本质：**logits 经一系列变换后归一化为概率分布，再采样一个 token。** 变换顺序至关重要——先惩罚、再屏蔽、再采样策略、最后采样。投机解码的 rejection sampler 在标准采样之外并行验证 draft token。

:::diagram sampling-html
```html
<div class="engine-step-flow">
  <div class="engine-step">
    <div class="engine-step-num">1</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Logits 处理</div>
      <div class="engine-step-desc">penalties / bad words 屏蔽 / grammar bitmask</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">2</div>
    <div class="engine-step-content">
      <div class="engine-step-title">采样策略</div>
      <div class="engine-step-desc">temperature → top-k → top-p → min-p 过滤</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">3</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Token 选择</div>
      <div class="engine-step-desc">greedy / multinomial 采样，rejection sampler 验证</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc sampling-html
采样流程分三个阶段：
1. Logits 处理 — 应用 penalties（频率/重复/presence 惩罚）、bad words 屏蔽、grammar bitmask。
2. 采样策略 — 依次应用 temperature 缩放、top-k 过滤、top-p 过滤、min-p 过滤。
3. Token 选择 — 执行 greedy 或 multinomial 采样，投机解码场景下 rejection sampler 验证 draft token。
:::

## 采样流程总览

`Sampler.__call__` 实现从 logits 到 token 的完整流水线，严格按规范顺序执行各阶段变换。

### 规范执行顺序

```python
# vllm/v1/sample/sampler.py
def __call__(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata):
    # 1. Logprobs 捕获（可选）
    if sampling_params.logprobs:
        raw_logprobs = logits.log_softmax(dim=-1)
    
    # 2. 转换为 float32 保证数值精度
    logits = logits.to(torch.float32)
    
    # 3. Allowed token ids 白名单屏蔽
    if allowed_token_ids is not None:
        logits = apply_token_ids_mask(logits, allowed_token_ids)
    
    # 4. Bad words 排除
    if bad_words_ids is not None:
        logits = apply_bad_words_mask(logits, bad_words_ids)
    
    # 5. Non-argmax-invariant logits processors
    logits = apply_logits_processors(logits, non_argmax_invariant_processors)
    
    # 6. Penalties（repetition / frequency / presence）
    logits = apply_penalties(logits, output_token_ids, prompt_token_ids)
    
    # 7. 采样
    if not all_random:
        # 7a. Greedy 采样
        sampled_tokens = logits.argmax(dim=-1)
    else:
        # 7b. Temperature 缩放
        logits = logits / temperature
        # 7c. Argmax-invariant logits processors（min_p）
        logits = apply_logits_processors(logits, argmax_invariant_processors)
        # 7d. Top-k 和 Top-p
        logits = apply_top_k_top_p(logits, top_k, top_p)
        # 7e. 从概率分布采样
        probs = logits.softmax(dim=-1)
        sampled_tokens = multinomial(probs)
        # 7f. 根据 temperature 选择 greedy 或 random
        sampled_tokens = select_greedy_or_random(sampled_tokens, temperature)
    
    # 8. 收集 logprobs
    if sampling_params.logprobs:
        logprobs = gather_logprobs(raw_logprobs, sampled_tokens)
    
    # 9. 返回 SamplerOutput
    return SamplerOutput(sampled_tokens, logprobs)
```

### 各阶段数据形状

| 阶段 | 输入形状 | 输出形状 | 说明 |
|------|----------|----------|------|
| 原始 logits | `[num_seqs, vocab_size]` | 同上 | 模型输出，float16/bfloat16 |
| float32 转换 | `[num_seqs, vocab_size]` | 同上 | 保证 softmax 数值稳定 |
| Penalties 后 | `[num_seqs, vocab_size]` | 同上 | 部分位置 logits 被修改 |
| Top-k/p 后 | `[num_seqs, vocab_size]` | 同上 | 部分位置设为 -inf |
| 采样结果 | `[num_seqs]` | `[num_seqs]` | 每个 seq 一个 token id |

`vllm/v1/sample/sampler.py`

## Temperature 与 Top-k/Top-p

Temperature 控制概率分布的"平坦程度"，Top-k 和 Top-p 进一步过滤候选 token。

### Temperature 缩放

Temperature 通过缩放 logits 影响概率分布：

```
logits_scaled = logits / temperature
probs = softmax(logits_scaled)
```

- **temperature < 1**：放大差异，高概率 token 更突出，趋向 greedy
- **temperature = 1**：不改变分布
- **temperature > 1**：平滑分布，低概率 token 机会增加，趋向均匀

当 `temperature < 1e-5` 时，直接使用 greedy 采样，跳过所有随机采样逻辑。最小 temperature 被钳制到 0.01 以保证数值安全。

```python
# vllm/v1/sample/sampler.py
if temperature < 1e-5:
    # Greedy 采样
    return logits.argmax(dim=-1)
temperature = max(temperature, 0.01)  # 数值安全下界
```

### Top-k 采样

Top-k 只保留概率最高的 k 个 token，其余设为 -inf：

```python
# vllm/v1/sample/ops/topk_topp_sampler.py
def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    top_k_logits, _ = logits.topk(k, dim=-1)
    threshold = top_k_logits[:, -1:]  # 第 k 大的值
    mask = logits < threshold
    logits[mask] = float('-inf')
    return logits
```

- `top_k = -1`：不启用 top-k 过滤
- `top_k = 1`：等价于 greedy
- 典型值：40、50、100

### Top-p (Nucleus) 采样

Top-p 保留累积概率达到 p 的最小 token 集合：

```python
# vllm/v1/sample/ops/topk_topp_sampler.py
def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = logits.sort(descending=True, dim=-1)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    
    # 找到累积概率超过 p 的位置
    sorted_mask = cumulative_probs > p
    # 保证至少保留一个 token
    sorted_mask[:, -1] = False
    
    # 将被过滤的位置设为 -inf
    sorted_logits[sorted_mask] = float('-inf')
    # 恢复原始顺序
    return scatter_sorted_logits(sorted_logits, sorted_indices)
```

- `top_p = 1.0`：不启用 top-p 过滤
- 典型值：0.9、0.95、0.99
- `sorted_mask[:, -1] = False` 保证至少一个 token 存活

### Top-k 与 Top-p 协作

**Top-k 先执行，Top-p 后执行**。Top-k 先粗筛 k 个候选，Top-p 再在剩余候选中按累积概率过滤：

```
logits → top_k(k=50) → top_p(p=0.9) → softmax → sample
```

这种顺序避免 Top-p 在整个词表上排序，提高效率。

### Triton 实现

当 batch size >= 8 时，vLLM 使用 Triton kernel 加速 top-k/top-p：

```python
# vllm/v1/sample/ops/topk_topp_sampler.py
if batch_size >= 8:
    # 使用 Qrita 算法的 Triton 实现
    return topk_topp_sampler_triton(logits, k, p)
else:
    # PyTorch 实现
    return topk_topp_sampler_pytorch(logits, k, p)
```

FlashInfer 后端使用 rejection sampling 方法实现 top-k/top-p，避免排序操作。

`vllm/v1/sample/ops/topk_topp_sampler.py`

## Min-p 采样

Min-p 是动态阈值策略，根据模型置信度自适应调整过滤强度。

### 核心原理

Min-p 过滤掉概率低于 `min_p × P(max_token)` 的 token：

```
threshold = min_p × P(token_max)
保留 P(token) >= threshold 的 token
```

- **高置信场景**：`P(max_token)` 很大（如 0.8），threshold 高，激进过滤
- **低置信场景**：`P(max_token)` 较小（如 0.3），threshold 低，保留更多候选

### 与 Top-p 的区别

| 特性 | Top-p | Min-p |
|------|-------|-------|
| 阈值类型 | 累积概率阈值 | 相对概率阈值 |
| 候选数量 | 动态变化，保证累积概率 | 动态变化，保证相对概率 |
| 高置信时 | 可能保留过多低概率 token | 激进过滤，只保留头部 |
| 低置信时 | 可能过早截断 | 保留更多候选 |

### Argmax-invariant 特性

Min-p 是 **argmax-invariant** 的：应用 min-p 后，argmax 结果不变。因此 min-p 在 temperature 缩放**之后**、top-k/top-p **之前**执行：

```
logits → penalties → temperature → min_p → top_k → top_p → sample
```

### Triton Kernel 实现

Min-p 的 Triton kernel 在 log-space 计算，避免 softmax：

```python
# vllm/v1/sample/logits_processor/builtin.py
def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    # 在 log-space 计算阈值
    max_logit = logits.max(dim=-1, keepdim=True)
    threshold_logit = max_logit + math.log(min_p)
    
    # 过滤低于阈值的 token
    mask = logits < threshold_logit
    logits[mask] = float('-inf')
    return logits
```

典型值：`min_p = 0.05` ~ `0.1`

`vllm/v1/sample/logits_processor/builtin.py`

## Penalties（频率/重复/Presence）

Penalties 通过修改已出现 token 的 logits 抑制重复。

### Repetition Penalty

对已出现 token 的 logits 乘以惩罚系数：

```
if logit[i] > 0:
    logit[i] = logit[i] / repetition_penalty
else:
    logit[i] = logit[i] * repetition_penalty
```

- `repetition_penalty = 1.0`：无惩罚
- `repetition_penalty > 1.0`：惩罚重复（典型值 1.1 ~ 1.2）
- 应用范围：**prompt + output tokens**

### Frequency Penalty

按 token 出现次数线性惩罚：

```
logit[i] = logit[i] - frequency_penalty × count(token_i in output)
```

- `frequency_penalty > 0`：惩罚重复
- 应用范围：**仅 output tokens**

### Presence Penalty

只要 token 出现过就固定惩罚：

```
logit[i] = logit[i] - presence_penalty × (1 if token_i in output else 0)
```

- `presence_penalty > 0`：惩罚出现过
- 应用范围：**仅 output tokens**

### 三者对比

| Penalty | 数学形式 | 应用范围 | 效果 |
|---------|----------|----------|------|
| Repetition | 乘法惩罚 | prompt + output | 强力抑制，适合长文本 |
| Frequency | 线性惩罚 | output only | 按次数递增惩罚 |
| Presence | 固定惩罚 | output only | 轻度抑制，鼓励多样性 |

### 执行顺序

Penalties 在 temperature 缩放**之前**执行（在 `apply_logits_processors` 中）：

```python
# vllm/v1/sample/sampler.py
# Penalties 属于 non-argmax-invariant processors
logits = apply_penalties(logits, output_token_ids, prompt_token_ids)
# 之后才应用 temperature
logits = logits / temperature
```

`vllm/v1/sample/sampler.py`

## Bad Words 过滤

Bad words 过滤将指定 token 或 token 序列的概率强制置零。

### 匹配逻辑

Bad words 支持两种形式：

1. **单 token**：`[[token_id]]` — 直接屏蔽该 token
2. **多 token 序列**：`[[t1, t2, t3]]` — 只屏蔽**最后一个**会完成该序列的 token

```python
# vllm/v1/sample/sampler.py
def apply_bad_words_mask(logits, bad_words_ids, token_ids):
    for bad_word in bad_words_ids:
        if len(bad_word) == 1:
            # 单 token：直接屏蔽
            logits[:, bad_word[0]] = float('-inf')
        else:
            # 多 token：检查前缀是否匹配
            prefix = bad_word[:-1]
            last_token = bad_word[-1]
            if token_ids.endswith(prefix):
                logits[:, last_token] = float('-inf')
```

### 上下文敏感匹配

多 token bad word 只在**当前上下文会完成该序列时**才屏蔽：

- Bad word: `["不", "好"]` (token ids: `[100, 200]`)
- 当前输出: `["今天", "天气", "不"]`
- 下一个 token 如果是 `200`（"好"），则被屏蔽
- 如果输出是 `["今天", "天气"]`，token `200` 不被屏蔽

### 前缀空格变体

Bad words 同时处理带前缀空格和不带前缀空格的 tokenization：

```python
# 假设 bad word 是 "bad"
# 会同时屏蔽：
# - "bad" (token_id: 1234)
# - " bad" (token_id: 5678，带前缀空格)
```

### 投机解码变体

在投机解码中，bad words 过滤对每个 draft position 独立应用：

```python
# vllm/v1/sample/rejection_sampler.py
for draft_pos in range(num_draft_tokens):
    apply_bad_words_mask(logits[draft_pos], bad_words_ids, 
                         prefix_tokens + draft_tokens[:draft_pos])
```

`vllm/v1/sample/sampler.py`

## Logits Processor 接口

vLLM 提供可扩展的 logits processor 机制，允许自定义 logits 变换。

### 接口定义

```python
# vllm/v1/sample/logits_processor/base.py
class LogitsProcessor(ABC):
    @abstractmethod
    def apply(self, logits: torch.Tensor, 
              sampling_metadata: SamplingMetadata) -> torch.Tensor:
        """应用 logits 变换"""
        pass
    
    @abstractmethod
    def is_argmax_invariant(self) -> bool:
        """是否不影响 argmax 结果"""
        pass
    
    def update_state(self, new_token_ids: list[int]) -> None:
        """更新内部状态（如 min_tokens 计数）"""
        pass
```

### Argmax-invariant 分类

Logits processors 分为两组：

| 组别 | 执行时机 | 示例 |
|------|----------|------|
| Non-argmax-invariant | temperature 之前 | min_tokens, logit_bias, penalties |
| Argmax-invariant | temperature 之后 | min_p |

**Argmax-invariant** 的 processor 不改变 argmax 结果，可以在 temperature 缩放后执行而不影响 greedy 采样的正确性。

### 内置 Processors

| Processor | 功能 | Argmax-invariant |
|-----------|------|------------------|
| `MinTokensLogitsProcessor` | 强制生成 min_tokens 个 token | No |
| `LogitBiasLogitsProcessor` | 对特定 token 加偏置 | No |
| `MinPLogitsProcessor` | Min-p 过滤 | Yes |

### 自定义 Processor

通过三种方式注册自定义 processor：

1. **插件注册**：
```python
from vllm.plugins import register_logits_processor
register_logits_processor("my_processor", MyLogitsProcessor)
```

2. **FQCN 字符串**：
```python
sampling_params.logits_processors = ["my_module.MyLogitsProcessor"]
```

3. **直接传入类型**：
```python
sampling_params.logits_processors = [MyLogitsProcessor]
```

### AdapterLogitsProcessor

`AdapterLogitsProcessor` 包装 per-request 的 processors：

```python
# vllm/v1/sample/logits_processor/adapter.py
class AdapterLogitsProcessor(LogitsProcessor):
    def __init__(self, processors: list[LogitsProcessor]):
        self.processors = processors
    
    def apply(self, logits, metadata):
        for processor in self.processors:
            logits = processor.apply(logits, metadata)
        return logits
```

`vllm/v1/sample/logits_processor/builtin.py`

## 与投机解码的协作

投机解码中，标准采样器与 rejection sampler 协同工作。

### Rejection Sampler 算法

Rejection sampler 基于 Leviathan et al. 2023 的算法验证 draft token：

```
对于每个 draft token x:
    计算 P_target(x) 和 P_draft(x)
    如果 P_target(x) / P_draft(x) >= uniform_random(0, 1):
        接受 x
    否则:
        拒绝 x，从 max(0, P_target - P_draft) 采样恢复 token
        停止验证
如果所有 draft token 都被接受:
    从 P_target 采样一个 bonus token
```

### 三种输出类型

| 输出类型 | 说明 |
|----------|------|
| Accepted draft tokens | 通过验证的 draft token |
| Recovered token | 拒绝后从调整分布采样的 token |
| Bonus token | 全部接受后额外采样的 token |

### Greedy 场景

当 temperature = 0（greedy）时，验证简化为：

```python
# vllm/v1/sample/rejection_sampler.py
if draft_token_id == target_argmax_id:
    accept(draft_token)
else:
    reject_and_output(target_argmax_id)
```

### Gumbel-max 技巧

随机采样使用 Gumbel-max trick 的等价形式：

```python
# 标准 rejection sampling
accept if P_target(x) / P_draft(x) >= uniform_random

# Gumbel-max 等价形式
probs_adjusted = P_target / exponential_noise
sampled = argmax(probs_adjusted)
```

Gumbel-max 避免显式计算概率比，提高数值稳定性。

### 采样器准备的 Logits

为支持 rejection sampling，采样器需要准备：

1. **Target logits**：对 draft token 位置的 logits
2. **Bonus logits**：最后一个 draft token 之后的 logits

```python
# vllm/v1/sample/sampler.py
# 投机解码场景
target_logits = logits[:num_draft_positions]
bonus_logits = logits[num_draft_positions]
```

`vllm/v1/sample/rejection_sampler.py`

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `temperature` | 1.0 | 采样温度，< 1e-5 使用 greedy |
| `top_k` | -1 | Top-k 值，-1 不启用 |
| `top_p` | 1.0 | Top-p 值，1.0 不启用 |
| `min_p` | 0.0 | Min-p 值，0.0 不启用 |
| `repetition_penalty` | 1.0 | 重复惩罚系数 |
| `frequency_penalty` | 0.0 | 频率惩罚 |
| `presence_penalty` | 0.0 | 存在惩罚 |
| `logprobs` | None | 返回 top-N logprobs |
| `prompt_logprobs` | None | 返回 prompt 的 logprobs |

`vllm/v1/sample/sampler.py`
