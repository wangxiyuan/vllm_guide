---
id: suffix-decoding
title: Suffix Decoding
category: decoding
level: intermediate
status: draft
readingMinutes: 10
tags:
  - Suffix Decoding
  - Speculative Decoding
codeRefs:
  - vllm/v1/spec_decode/suffix_decoding.py
heroText: 基于 [后缀树](term:从 prompt 和已生成 token 构建的数据结构，可高效查找最长后缀匹配。) 的模式匹配投机解码，无需神经模型。
---

## 心智模型

后缀解码就像爵士乐手的即兴演奏：当乐手听到当前旋律与之前演奏过的某个片段相似时，会自然地延续那个片段之后的旋律。`SuffixDecodingCache` 从所有 token 构建后缀树，使得最长后缀匹配成为可能。

:::diagram suffix-mental-model-html
```html
<div class="kv-mental-model">
  <div class="kv-gpu-memory">
    <div class="kv-gpu-title">后缀树（从所有 token 构建）</div>
    <div class="kv-blocks">
      <div class="arch-row">
        <div class="arch-box arch-highlight">root</div>
      </div>
      <div class="arch-row">
        <div class="arch-box arch-module">"A"</div>
      </div>
      <div class="arch-row">
        <div class="arch-box arch-module">"B"</div>
        <div class="arch-box arch-module">"C"</div>
      </div>
      <div class="arch-row">
        <div class="arch-box">→ [C, D, E]</div>
        <div class="arch-box">→ [F, G]</div>
      </div>
    </div>
  </div>
  <div class="kv-gpu-memory">
    <div class="kv-gpu-title">当前后缀: A B</div>
    <div class="kv-blocks">
      <div class="arch-row">
        <div class="arch-box arch-highlight">匹配 → C D E</div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc suffix-mental-model-html
后缀解码的心智模型：SuffixDecodingCache 从 prompt 和已生成 token 构建后缀树。当需要预测 draft token 时，提取当前序列的后缀（如 "A B"），在后缀树中查找最长匹配路径，取其后续 token（C D E）作为 draft token。后缀树支持可变长度匹配，可以找到比固定窗口 n-gram 更长的模式。
:::

## 核心算法

`SuffixDecodingProposer.propose()` 的完整流程：

:::steps id=suffix-algorithm-steps
### 1. 初始化请求
对新请求调用 `start_request()`，从 prompt token 构建后缀树。
`vllm/v1/spec_decode/suffix_decoding.py`

### 2. 追加已生成 token
每步调用 `add_active_response()`，将新采样的 token 追加到后缀树中。
`vllm/v1/spec_decode/suffix_decoding.py`

### 3. 提取最近 token
提取最近的 token（最多 `max_tree_depth` 个）作为匹配后缀。
`vllm/v1/spec_decode/suffix_decoding.py`

### 4. 后缀树匹配
调用 `suffix_cache.speculate()`，在后缀树中查找最长匹配并返回 draft token。
`vllm/v1/spec_decode/suffix_decoding.py`

### 5. 动态 draft 数量
每个请求的 draft token 数量是动态的，取决于匹配到的模式长度。
`vllm/v1/spec_decode/suffix_decoding.py`
:::

### SuffixDecodingCache

后缀树缓存的核心数据结构，负责构建和查询：

`vllm/v1/spec_decode/suffix_decoding.py`

```python
class SuffixDecodingCache:
    def __init__(self, max_tree_depth, max_cached_requests, ...):
        self.max_tree_depth = max_tree_depth
        self.max_cached_requests = max_cached_requests

    def speculate(self, request_id, recent_tokens, num_spec_tokens):
        node = self.root
        for token in reversed(recent_tokens):
            if token not in node.children:
                break
            node = node.children[token]
        return node.speculate_tokens(num_spec_tokens)
```

### 动态 Draft Token 数量

与 n-gram 固定 k 个 draft token 不同，suffix decoding 的 draft 数量取决于匹配长度和 `max_spec_factor` 参数：

```python
num_spec = min(
    len(matched_suffix) * max_spec_factor,
    max_spec_tokens
)
```

## 与 N-gram 的区别

两者都是非神经方法，但数据结构和匹配策略不同：

| 特性 | N-gram | Suffix Decoding |
|------|--------|-----------------|
| 数据结构 | 滑动窗口（固定大小） | 后缀树（可变深度） |
| 匹配方式 | 固定窗口内匹配 | 最长后缀匹配 |
| 匹配长度 | 受 min_n/max_n 限制 | 可匹配更长的模式 |
| Draft 数量 | 固定 k | 动态（取决于匹配长度） |
| GPU 实现 | 有（NgramProposerGPU） | 无（纯 CPU） |
| 外部依赖 | 无 | 需要 `arctic-inference` |
| Token 更新 | 不更新（无状态） | 实时更新后缀树 |

### 何时选择 Suffix Decoding

- 请求中有大量重复模式（如代码生成、模板化文本）
- 匹配长度可能很长（n-gram 的 max_n 不够用）
- 可以接受 CPU 开销和外部依赖

### 何时选择 N-gram

- 需要全 GPU 流水线（无 CPU 同步）
- 不想引入额外依赖
- 模式匹配长度较短

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--speculative-method` | None | 设为 `suffix` |
| `--suffix-decoding-max-tree-depth` | None | 后缀树最大深度 |
| `--suffix-decoding-max-cached-requests` | None | 最大缓存请求数 |
| `--suffix-decoding-max-spec-factor` | None | Draft 数量 = 匹配长度 × 此因子 |
| `--suffix-decoding-min-token-prob` | None | 候选 token 的最低概率阈值 |

**注意**：Suffix Decoding 需要 `arctic-inference` 包支持。
