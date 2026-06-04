---
id: ngram
title: N-gram 投机解码
category: decoding
level: intermediate
status: draft
readingMinutes: 10
tags:
  - Ngram
  - Speculative Decoding
codeRefs:
  - vllm/v1/spec_decode/ngram_proposer.py
  - vllm/v1/spec_decode/ngram_proposer_gpu.py
heroText: 无需神经模型，通过 [n-gram 匹配](term:在已生成 token 序列中查找与当前后缀最长的 n-gram 模式，用其后续 token 作为候选。) 在已有 token 序列中找到模式并预测后续 token。
---

## 心智模型

N-gram 投机解码就像手机输入法的联想补全：当你在一段文字中看到"不仅如此"，系统发现在之前的文本中"不仅如此"后面总是跟着"而且"，就建议"而且"作为下一个词。纯模式匹配，不需要任何神经网络。

:::diagram ngram-mental-model-html
```html
<div class="kv-mental-model">
  <div class="kv-gpu-memory">
    <div class="kv-gpu-title">已生成序列</div>
    <div class="kv-blocks">
      <span class="kv-block kv-block-used">A</span>
      <span class="kv-block kv-block-used">B</span>
      <span class="kv-block kv-block-cached">C</span>
      <span class="kv-block kv-block-cached">D</span>
      <span class="kv-block kv-block-free">?</span>
    </div>
  </div>
  <div class="kv-data-structures">
    <div class="kv-ds">
      <div class="kv-ds-title">查找后缀 "C D"</div>
      <div class="kv-ds-items">
        <div class="kv-ds-item">... C D E F ...</div>
      </div>
    </div>
    <div class="kv-ds">
      <div class="kv-ds-title">预测</div>
      <div class="kv-ds-items">
        <div class="kv-ds-item">E F G</div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc ngram-mental-model-html
N-gram 投机解码的心智模型：在已生成序列的末尾提取后缀（如 "C D"），在序列中搜索该后缀之前出现的位置，取出其后续的 token（如 E F G）作为 draft token。这是纯模式匹配，不需要任何神经网络推理。
:::

## CPU N-gram 算法

CPU 版本使用 KMP 类似算法查找最长匹配的 n-gram 模式。

:::steps id=ngram-cpu-steps
### 1. 反转 token 序列
将已生成的 token 序列反转，方便从末尾向前匹配。
`vllm/v1/spec_decode/ngram_proposer.py`

### 2. 计算 LPS 数组
使用类似 KMP 的算法计算 Longest Proper Prefix which is also Suffix 数组。
`vllm/v1/spec_decode/ngram_proposer.py`

### 3. 查找最长匹配
在 [min_n, max_n] 范围内找到最长的有效 n-gram 匹配。
`vllm/v1/spec_decode/ngram_proposer.py`

### 4. 提取候选 token
从匹配位置提取 k 个后续 token 作为 draft。
`vllm/v1/spec_decode/ngram_proposer.py`
:::

核心函数 `_find_longest_matched_ngram_and_propose_tokens()` 使用 Numba JIT 加速，将 Python 循环编译为机器码以获得可接受的性能。

`vllm/v1/spec_decode/ngram_proposer.py`

```python
@numba.jit(nopython=True)
def _find_longest_matched_ngram_and_propose_tokens(
    context, max_n, min_n, num_proposals
):
    reversed_context = context[::-1]
    lps = compute_lps(reversed_context)
    longest_match_length = 0
    longest_match_pos = 0
    for i in range(len(lps)):
        match_len = lps[i]
        if min_n <= match_len <= max_n and match_len > longest_match_length:
            longest_match_length = match_len
            longest_match_pos = i
    if longest_match_length > 0:
        start = longest_match_pos + 1
        return context[start:start + num_proposals]
    return []
```

## GPU N-gram 算法

GPU 版本（`NgramProposerGPU`）完全向量化，无需 CPU 同步，适合异步调度。

:::diagram ngram-gpu-flow-html
```html
<div class="engine-step-flow">
  <div class="engine-step">
    <div class="engine-step-num">1</div>
    <div class="engine-step-content">
      <div class="engine-step-title">unfold() 滑动窗口</div>
      <div class="engine-step-desc">生成所有 n-gram 的滑动窗口视图</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">2</div>
    <div class="engine-step-content">
      <div class="engine-step-title">布尔匹配</div>
      <div class="engine-step-desc">比较每个窗口与当前后缀 → boolean mask</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">3</div>
    <div class="engine-step-content">
      <div class="engine-step-title">argmax 选择</div>
      <div class="engine-step-desc">跨 n-gram size 选择最长匹配</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">4</div>
    <div class="engine-step-content">
      <div class="engine-step-title">提取 draft tokens</div>
      <div class="engine-step-desc">用 masking 提取匹配位置后的 k 个 token</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc ngram-gpu-flow-html
GPU N-gram 算法流程包含 4 个步骤：
1. **unfold() 滑动窗口**：对 token 序列执行 unfold 操作，生成所有 n-gram 的滑动窗口视图。
2. **布尔匹配**：将每个窗口与当前后缀比较，生成 boolean mask。
3. **argmax 选择**：跨不同 n-gram size 选择最长匹配的位置。
4. **提取 draft tokens**：用 masking 从匹配位置提取后续 k 个 token 作为 draft。
整个过程在 GPU 上完成，无需 CPU 同步。
:::

### 核心实现

`NgramGPUKernel` 是一个 `nn.Module`，支持 `@support_torch_compile`：

`vllm/v1/spec_decode/ngram_proposer_gpu.py`

```python
class NgramGPUKernel(nn.Module):
    @support_torch_compile
    def forward(self, input_ids, ...):
        for n in range(self.min_n, self.max_n + 1):
            windows = input_ids.unfold(dimension=1, size=n, step=1)
            suffix = input_ids[:, -n:]
            matches = (windows == suffix).all(dim=-1)
            best_match_pos = matches.argmax(dim=-1)
        draft_tokens = extract_tokens(best_match_pos, num_proposals)
        return draft_tokens
```

### CPU vs GPU 对比

| 特性 | CPU 版本 | GPU 版本 |
|------|---------|---------|
| 算法 | KMP-like + LPS | unfold + boolean match + argmax |
| 加速 | Numba JIT | Torch compile |
| CPU 同步 | 需要将序列传到 CPU | 完全在 GPU 上 |
| 异步调度 | 不兼容 | 兼容 |
| 适用场景 | 小 batch / 调试 | 生产环境 |

## 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--speculative-method` | None | 设为 `ngram` |
| `--prompt-lookup-min` | 1 | N-gram 最小匹配长度（min_n） |
| `--prompt-lookup-max` | 3 | N-gram 最大匹配长度（max_n） |
| `--num-speculative-tokens` | None | Draft token 数量 |

启动示例：

```bash
vllm serve model_name \
  --speculative-method ngram \
  --num-speculative-tokens 5 \
  --prompt-lookup-min 1 \
  --prompt-lookup-max 3
```
