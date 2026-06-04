---
id: speculative-decoding
title: 投机解码总览
category: decoding
level: advanced
status: draft
readingMinutes: 16
tags:
  - Speculative Decoding
  - Draft
  - Verify
codeRefs:
  - vllm/v1/spec_decode/llm_base_proposer.py
  - vllm/v1/worker/gpu/spec_decode/rejection_sampler.py
heroText: 快速起草、严格验证：[proposer](term:快速生成候选 token 的模块，也叫 drafter。) 生成候选 token，target model 一次 forward 并行验证，[rejection sampling](term:按概率比 p(x)/q(x) 决定是否接受每个 draft token 的算法。) 决定接受/拒绝。
---

## 心智模型

投机解码就像律师事务所的审稿流程：初级律师（proposer）快速起草一份文档，高级律师（target model）一次性审阅全部内容。高级律师接受正确的部分，遇到第一个错误就拒绝，并从错误位置重新改写。如果初级律师大部分写对了，整个流程比高级律师从头写要快得多。

:::diagram spec-mental-model-html
```html
<div class="arch-diagram">
  <div class="arch-row">
    <div class="arch-proc">
      <div class="arch-proc-title">初级律师 (Proposer)</div>
      <div class="arch-proc-sub">快速起草候选文档</div>
      <div class="arch-box">Draft: A B C D</div>
    </div>
    <div class="arch-channel">
      <div class="arch-arrow">→</div>
      <div class="arch-label">候选 token</div>
    </div>
    <div class="arch-proc">
      <div class="arch-proc-title">高级律师 (Target Model)</div>
      <div class="arch-proc-sub">一次审阅全部内容</div>
      <div class="arch-box">Verify: A ✓ B ✓ X ≠ C</div>
    </div>
    <div class="arch-channel">
      <div class="arch-arrow">→</div>
      <div class="arch-label">接受 A B + 奖励 X</div>
    </div>
    <div class="arch-proc">
      <div class="arch-proc-title">最终输出</div>
      <div class="arch-highlight">A B X（3 tokens / 1 forward）</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc spec-mental-model-html
投机解码的心智模型展示了两个角色的协作：初级律师（Proposer）快速起草候选 token（如 A B C D），高级律师（Target Model）一次 forward 并行验证所有候选。验证结果为 A ✓ B ✓ X ≠ C（第一个错误出现在 C），最终输出为 A B X（接受的前缀 + 奖励 token X）。一次 forward 产出了 3 个 token，而非传统的 1 个。
:::

## 投机解码核心流程

投机解码的每一步分为四个阶段：proposer 生成候选、target model 验证、rejection sampler 判断接受/拒绝、重新采样奖励 token。

:::diagram spec-core-flow-html
```html
<div class="sched-flow">
  <div class="sched-phase" data-phase="running">
    <div class="sched-phase-title">Phase 1: Proposer 生成</div>
    <div class="sched-phase-steps">
      <div class="sched-step">proposer.generate() → k 个 draft token</div>
    </div>
  </div>
  <div class="sched-phase" data-phase="running">
    <div class="sched-phase-title">Phase 2: Target Model Forward</div>
    <div class="sched-phase-steps">
      <div class="sched-step">一次 forward 处理 k+1 个位置（原始 + draft）</div>
    </div>
  </div>
  <div class="sched-phase" data-phase="waiting">
    <div class="sched-phase-title">Phase 3: Rejection Sampling</div>
    <div class="sched-phase-steps">
      <div class="sched-step">比较 target logits 与 draft token → 接受匹配前缀，拒绝首个不匹配</div>
    </div>
  </div>
  <div class="sched-phase" data-phase="output">
    <div class="sched-phase-title">Phase 4: 奖励 Token</div>
    <div class="sched-phase-steps">
      <div class="sched-step">从拒绝点采样 bonus token → 输出: accepted + bonus</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc spec-core-flow-html
投机解码核心流程分四个阶段：
1. **Proposer 生成**：proposer.generate() 生成 k 个 draft token。
2. **Target Model Forward**：target model 一次 forward 处理 k+1 个位置（原始位置 + k 个 draft 位置），产出所有位置的 logits。
3. **Rejection Sampling**：比较 target logits 与 draft token，接受匹配的前缀，在首个不匹配处拒绝。
4. **奖励 Token**：从拒绝点采样一个 bonus token（从 target 分布），最终输出为 accepted tokens + bonus token。
:::

:::steps id=spec-core-steps
### 1. Proposer 生成 k 个候选 token
proposer 根据自身方法（draft model / n-gram / EAGLE 等）生成 k 个 draft token。
`vllm/v1/spec_decode/llm_base_proposer.py`

### 2. Target Model 执行一次 forward
将原始 token + k 个 draft token 拼接，target model 一次 forward 并行计算所有位置的 logits。
`vllm/v1/worker/gpu_model_runner.py`

### 3. Rejection Sampler 验证
比较 target model 的 logits 与 draft token：接受匹配的前缀，在首个不匹配处拒绝。
`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`

### 4. 采样奖励 token
在拒绝点从 target 分布采样一个 bonus token。输出：accepted tokens + bonus token。
`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`
:::

## 接受/拒绝示例

假设 proposer 生成 4 个 draft token，target model 一次 forward 验证：

### 场景 1：部分接受

| 位置 | Draft Token | Target Top-1 | 结果 |
|------|-------------|-------------|------|
| 0 | A | A | ✓ 接受 |
| 1 | B | B | ✓ 接受 |
| 2 | C | X | ✗ 拒绝（X ≠ C） |
| 3 | D | — | 未验证 |

输出：**A B X**（A B 为接受的前缀，X 为从 target 分布采样的 bonus token）

### 场景 2：全部接受

| 位置 | Draft Token | Target Top-1 | 结果 |
|------|-------------|-------------|------|
| 0 | A | A | ✓ 接受 |
| 1 | B | B | ✓ 接受 |
| 2 | C | C | ✓ 接受 |
| 3 | D | D | ✓ 接受 |

输出：**A B C D E**（A B C D 全部接受，E 为 bonus token）

关键洞察：全部接受时，一次 forward 产出 k+1 个 token（k 个 draft + 1 个 bonus），加速比最高。部分接受时，产出 accepted_count + 1 个 token。

## Proposer 方法一览

vLLM 支持多种 proposer 方法，按是否使用神经网络、是否自回归可以分为以下几类：

| 方法 | 工作原理 | 神经模型？ | 自回归？ | 关键特征 |
|------|----------|-----------|---------|---------|
| `draft_model` | 独立的小模型自回归生成 | 是 | 是 | 最基础的投机解码，需要单独的 draft model |
| `eagle` / `eagle3` | 利用 target hidden states + 轻量 EAGLE head | 是 | 是 | 高接受率，EAGLE3 用中间层 hidden states |
| `ngram` / `ngram_gpu` | 在已有 token 序列中做 n-gram 模式匹配 | 否 | 否 | 零额外推理开销，GPU 版全向量化 |
| `medusa` | 多个预测头并行预测不同未来位置 | 是 | 否 | 单次 forward 生成所有 draft，低延迟 |
| `mtp` | 共享 embedding/LM head 的多 token 预测层 | 是 | 是 | 支持 deepseek/gemma4/step3.5/qwen3_next 等 |
| `dflash` | 非因果注意力一次生成所有 draft token | 是 | 否 | 并行起草，单次 pass 产出所有候选 |
| `suffix_decoding` | 后缀树模式匹配 | 否 | 否 | 可变长度匹配，比 n-gram 更长模式 |
| `extract_hidden_states` | 直接提取 target model 的中间预测 | 是 | 否 | 调试用，不做实际投机解码 |
| `custom_class` | 用户自定义 proposer | 视实现 | 视实现 | 灵活扩展 |

## Rejection Sampling 算法

Rejection sampling 是投机解码的核心验证算法，决定每个 draft token 的接受/拒绝。

### 贪心模式

对于贪心解码（temperature=0），判断极其简单：

```python
accept = (target_argmax == draft_token)
```

每个位置只需比较 target model 的 argmax 与 draft token 是否一致。

### 随机模式

对于随机采样（temperature>0），接受概率取决于 draft 分布 q(x) 和 target 分布 p(x) 的比值：

```python
accept = (p(x) / q(x)) > u
```

其中 u 是 [0, 1] 上的均匀随机数。当 draft 分布 q(x) 接近 target 分布 p(x) 时，p(x)/q(x) 接近 1，接受概率高。

拒绝后，从**残差分布**重新采样 bonus token：

```python
residual = max(p(x) - q(x), 0)
bonus_token = sample(normalize(residual))
```

### 三个 Triton Kernel

Rejection sampling 在 GPU 上通过三个 Triton kernel 实现：

| Kernel | 功能 |
|--------|------|
| `compute_block_stats` | 计算每个 block 的概率统计（max、sum），用于后续比较 |
| `rejection_kernel` | 执行接受/拒绝判断，生成 accept mask |
| `resample_kernel` | 从残差分布采样 bonus token |

`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`

## Scheduler 中的投机解码

投机解码深度集成到调度器中，影响 token 计数、KV 分配和状态更新。

### Token 计数

```python
num_tokens_with_spec = num_prompt_tokens + num_output_tokens + num_spec_tokens
```

投机 token 被计入 `num_tokens_with_spec`，调度器会为这些 token 预分配 KV blocks（lookahead）。

### KV 分配与回滚

调度时，`allocate_slots()` 为 spec tokens 分配 KV 空间。验证后：

```python
num_rejected = num_draft - num_accepted
num_computed_tokens -= num_rejected
spec_token_ids.clear()
spec_token_ids.extend(new_spec_tokens)
```

被拒绝的 token 对应的 KV 空间会被释放，`num_computed_tokens` 回退到实际接受的长度。

### SchedulerOutput 中的投机字段

```python
scheduled_spec_decode_tokens: dict[str, list[int]]
```

每个请求的投机 token IDs 通过此字段传递给 ModelRunner。

`vllm/v1/core/sched/scheduler.py`

## 配置方法

通过命令行参数启用和配置投机解码：

| 参数 | 说明 |
|------|------|
| `--speculative-model` | proposer 模型路径（ngram/suffix 不需要） |
| `--num-speculative-tokens` | 每步生成的 draft token 数（k） |
| `--speculative-method` | proposer 方法：eagle/eagle3/ngram/medusa/mtp/dflash/suffix 等 |
| `--draft-sample-method` | proposer 采样方法：greedy / probabilistic |
| `--rejection-sample-method` | 验证算法：rejection（默认）/ typical_acceptance |

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `speculative_model` | None | Proposer 模型路径 |
| `num_speculative_tokens` | None | Draft token 数量 |
| `speculative_method` | None | Proposer 方法名 |
| `draft_sample_method` | "greedy" | Proposer 采样策略 |
| `rejection_sample_method` | "rejection" | 验证算法 |
| `speculative_max_model_len` | None | Proposer 最大序列长度 |
| `speculative_disable_by_batch_size` | None | batch size 超过此值时禁用投机解码 |
| `ngram_prompt_lookup_min` | 1 | N-gram 最小匹配长度 |
| `ngram_prompt_lookup_max` | 3 | N-gram 最大匹配长度 |
| `speculative_acceptance_method` | "rejection_sampler" | 接受方法 |
| `disable_mqa_scorer` | False | 禁用 MQA scorer |

`vllm/config/speculative.py`
