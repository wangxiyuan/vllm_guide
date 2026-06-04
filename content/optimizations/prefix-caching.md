---
id: prefix-caching
title: Prefix Caching
category: optimizations
level: intermediate
status: draft
readingMinutes: 12
tags:
  - Prefix Caching
  - Hash
  - KV Cache
codeRefs:
  - vllm/v1/core/kv_cache_utils.py
  - vllm/v1/core/kv_cache_manager.py
heroText: "相同 prompt 前缀的 [KV block](term:固定大小的 GPU 显存块，存放 block_size 个 token 的 Key 和 Value 向量。) 只计算和保存一次，后续请求通过 [链式哈希](term:每个 block 的哈希依赖前一个 block 的哈希和本 block 的 token，形成向下封闭的哈希链。) 匹配直接复用。"
---

## 心智模型

想象办公室里有一块**共享白板**。多个同事要写同样的标题（system prompt），第一个人写完后留在白板上，后面的人只需**指一指**已有的标题，不用重新写。标题一直保留，直到白板写满需要擦掉最旧的内容。

:::diagram pc-mental-model
```html
<div class="diagram-container">
  <div class="diagram" data-ref="vllm/v1/core/kv_cache_manager.py">
    <div class="diagram-title">Prefix Caching 心智模型</div>
    <div class="kv-mental-model">
      <div class="kv-data-structures">
        <div class="kv-ds">
          <div class="kv-ds-title">请求 A（先到）</div>
          <div class="kv-ds-items">
            <div class="kv-ds-item">System Prompt</div>
            <div class="kv-ds-item">上下文</div>
            <div class="kv-ds-item">用户问题 A</div>
          </div>
        </div>
        <div class="kv-ds">
          <div class="kv-ds-title">请求 B（后到）</div>
          <div class="kv-ds-items">
            <div class="kv-ds-item">System Prompt</div>
            <div class="kv-ds-item">上下文</div>
            <div class="kv-ds-item">用户问题 B</div>
          </div>
        </div>
      </div>
      <div class="kv-gpu-memory">
        <div class="kv-gpu-title">KV Cache（物理 block）</div>
        <div class="kv-blocks">
          <div class="kv-block kv-block-cached" data-bid="0">B0</div>
          <div class="kv-block kv-block-cached" data-bid="1">B1</div>
          <div class="kv-block kv-block-used" data-bid="2">B2</div>
          <div class="kv-block kv-block-used" data-bid="3">B3</div>
        </div>
      </div>
      <div class="kv-legend">
        <div class="kv-legend-item"><span class="kv-legend-dot kv-block-cached"></span> 共享 block（哈希匹配，只存一份）</div>
        <div class="kv-legend-item"><span class="kv-legend-dot kv-block-used"></span> 请求独占 block</div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc pc-mental-model
Prefix Caching 心智模型展示了两个请求共享相同前缀时的 KV block 复用：

**请求 A**（先到）：包含 System Prompt + 上下文 + 用户问题 A，三个部分分别占用 block。首次计算时，所有 block 的 KV 都需要从头算。

**请求 B**（后到）：包含 System Prompt + 上下文 + 用户问题 B。前两个 block 与 A 的前缀完全相同，通过哈希匹配发现缓存命中，直接复用 B0 和 B1，只需计算用户问题 B 对应的新 block。

**KV Cache**：物理 block 中 B0 和 B1 被两个请求共享（ref_cnt > 1），B2 和 B3 分别被 A 和 B 独占。共享 block 不会被单独回收，直到所有引用者释放。

关键收益：相同前缀只计算一次、显存占用大幅减少、首 token 延迟降低。
:::

关键收益：**相同前缀只计算一次**、**显存占用大幅减少**、**首 token 延迟降低**。

## 链式哈希

Prefix Caching 的核心是**链式哈希**——每个 block 的哈希不仅依赖自身的 token，还依赖前一个 block 的哈希，形成一条哈希链。

```python
# vllm/v1/core/kv_cache_utils.py
def hash_block_tokens(
    parent_hash: Optional[int],
    token_ids: Sequence[int],
    extra_keys: Optional[Hashable] = None,
) -> int:
    return hash((parent_hash, *token_ids, extra_keys))
```

哈希链的构建过程：

```
hash₀ = H(seed, tokens[0:16])
hash₁ = H(hash₀, tokens[16:32])
hash₂ = H(hash₁, tokens[32:48])
...
```

**向下封闭性**：如果 hash₃ 匹配，则 hash₀-₂ 必然也匹配。因为 hash₃ 依赖 hash₂，hash₂ 依赖 hash₁，以此类推。这意味着一旦某个 block 命中缓存，它之前的所有 block 都必然命中——不需要逐个验证。

:::diagram pc-chain-hash
```html
<div class="diagram-container">
  <div class="diagram" data-ref="vllm/v1/core/kv_cache_utils.py">
    <div class="diagram-title">链式哈希结构</div>
    <div class="kv-hash-chain">
      <div class="kv-hash-block">
        <div class="kv-hash-block-title">hash₀</div>
        <div class="kv-hash-block-detail">H(seed, tokens[0:16])</div>
        <div class="kv-hash-block-detail">Block 0: tokens 0-15</div>
      </div>
      <div class="kv-hash-arrow">→</div>
      <div class="kv-hash-block">
        <div class="kv-hash-block-title">hash₁</div>
        <div class="kv-hash-block-detail">H(hash₀, tokens[16:32])</div>
        <div class="kv-hash-block-detail">Block 1: tokens 16-31</div>
      </div>
      <div class="kv-hash-arrow">→</div>
      <div class="kv-hash-block">
        <div class="kv-hash-block-title">hash₂</div>
        <div class="kv-hash-block-detail">H(hash₁, tokens[32:48])</div>
        <div class="kv-hash-block-detail">Block 2: tokens 32-47</div>
      </div>
      <div class="kv-hash-arrow">→</div>
      <div class="kv-hash-block">
        <div class="kv-hash-block-title">hash₃</div>
        <div class="kv-hash-block-detail">H(hash₂, tokens[48:64])</div>
        <div class="kv-hash-block-detail">Block 3: tokens 48-63</div>
      </div>
    </div>
    <div class="arch-flow-label"><strong>向下封闭</strong>：hash₃ 匹配 ⟹ hash₀~hash₂ 必然匹配</div>
  </div>
</div>
```
:::

:::diagram-desc pc-chain-hash
链式哈希结构展示了 block 哈希的依赖关系：

**Block 0**：hash₀ = H(seed, tokens[0:16])，依赖初始种子和自身 token。
**Block 1**：hash₁ = H(hash₀, tokens[16:32])，依赖父哈希 hash₀ 和自身 token。
**Block 2**：hash₂ = H(hash₁, tokens[32:48])，依赖父哈希 hash₁ 和自身 token。
**Block 3**：hash₃ = H(hash₂, tokens[48:64])，依赖父哈希 hash₂ 和自身 token。

**向下封闭性**：如果 hash₃ 匹配，则 hash₂ 必然匹配（因为 hash₃ 依赖 hash₂），进而 hash₁、hash₀ 也必然匹配。这意味着缓存查找只需要从最新 block 向前匹配，一旦命中就保证前缀完全一致。
:::

## 什么会影响哈希

除了 token_ids 和 parent_hash，`extra_keys` 也会参与哈希计算。不同场景下 extra_keys 包含的内容不同：

| 场景 | extra_keys 内容 | 说明 |
|------|----------------|------|
| **多模态（Multi-modal）** | `(mm_hash, offset_within_block)` | 多模态特征的哈希及其在 block 内的偏移 |
| **LoRA** | `lora_name` | 不同 LoRA 适配器的 KV 不可混用 |
| **Cache Salt** | `request.cache_salt`（仅 block 0） | 用户指定的盐值，强制隔离不同请求的缓存 |
| **Prompt Embeds** | SHA-256 hash of embedding tensor slice | 自定义 embedding 的哈希 |

这些 extra_keys 确保了不同上下文下的 block 即使 token_ids 相同也不会错误复用。

## 缓存查找流程

当一个新请求到达时，调度器需要确定哪些 block 已经在缓存中，可以直接复用：

:::steps id=cache-lookup-flow
### 1. 前置检查
如果 prefix caching 被禁用，直接跳过查找，所有 block 都需要重新计算。
`vllm/v1/core/kv_cache_manager.py`

### 2. 计算最大命中长度
`max_cache_hit_length = num_tokens - 1`。最后一个 token 必须重新计算以获取 logits，因此最多只能命中 `num_tokens - 1` 个 token 对应的 block。
`vllm/v1/core/kv_cache_manager.py`

### 3. 查找最长缓存命中
调用 `coordinator.find_longest_cache_hit(block_hashes, max_cache_hit_length)`，从 hash₀ 开始逐个匹配，找到最长的连续命中前缀。
`vllm/v1/core/kv_cache_utils.py`

### 4. 返回结果
返回 `(computed_blocks, num_new_computed_tokens)`。computed_blocks 是已命中的 block 列表，num_new_computed_tokens 是需要新计算的 token 数。
`vllm/v1/core/kv_cache_manager.py`
:::

## 哈希种子与可复现性

链式哈希的起点（seed）决定了整个哈希链。vLLM 通过 `init_none_hash()` 初始化种子：

```python
# vllm/v1/core/kv_cache_utils.py
def init_none_hash() -> int:
    seed = os.environ.get("PYTHONHASHSEED")
    if seed is not None:
        return hash(int(seed))
    return hash(os.urandom(32))
```

- 如果设置了 `PYTHONHASHSEED` 环境变量，使用它作为种子——这保证了**跨进程可复现性**（TP 场景下各 rank 的哈希一致）
- 否则使用 `os.urandom(32)` 生成随机种子——每个进程独立，缓存不跨进程共享

种子存储在全局变量 `NONE_HASH` 中，作为所有哈希链的起始 parent_hash。

## 全命中场景

当请求的所有 token 都命中缓存时（`num_computed_tokens == num_tokens`），vLLM 会强制重算最后一个 token：

```python
# vllm/v1/core/kv_cache_manager.py
if num_computed_tokens == num_tokens:
    num_computed_tokens = num_tokens - 1
```

这确保了最后一个 token 总是被**重新计算**，从而得到正确的 logits 用于采样。如果直接复用最后一个 token 的 KV，attention 输出可能不包含当前请求的完整上下文。

## Block 生命周期与缓存标记

一个 block 在 prefix caching 中会经历以下状态转换：

:::diagram pc-block-lifecycle
```html
<div class="diagram-container">
  <div class="diagram" data-ref="vllm/v1/core/kv_cache_manager.py">
    <div class="diagram-title">Block 生命周期</div>
    <div class="kv-lifecycle">
      <div class="kv-lc-state" data-state="free">
        Free
        <small>无哈希，无引用</small>
      </div>
      <div class="kv-lc-arrow">计算 KV → 设置 hash</div>
      <div class="kv-lc-state" data-state="used">
        In Use
        <small>hash 已设置，ref_cnt > 0</small>
      </div>
      <div class="kv-lc-arrow">请求释放 → ref_cnt = 0</div>
      <div class="kv-lc-state" data-state="cached-free">
        Cached+Free
        <small>hash 保留，ref_cnt = 0，可被复用</small>
      </div>
      <div class="kv-lc-arrow">缓存淘汰 → 重新分配</div>
      <div class="kv-lc-state" data-state="free">
        Free
        <small>hash 被清除</small>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc pc-block-lifecycle
Block 生命周期展示了 prefix caching 中 block 的状态转换：

**Free**（无哈希，无引用）：block 处于空闲状态，没有存储任何 KV 数据。

**In Use**（hash 已设置，ref_cnt > 0）：block 正在被某个请求使用。KV 计算完成后设置哈希值，用于后续缓存查找。

**Cached+Free**（hash 保留，ref_cnt = 0，可被复用）：请求释放了 block，但哈希值被保留。这意味着后续请求如果哈希匹配，可以直接复用这个 block 的 KV 数据，无需重新计算。这是 prefix caching 的核心——hash 在释放时不清除。

**Free**（hash 被清除）：当缓存空间不足时，Cached+Free 的 block 会被淘汰。此时哈希被清除，block 回到空闲状态。新分配的 block 会覆盖旧的 KV 数据。

关键点：hash 在请求释放时**不会**被清除，只有在 block 被淘汰并重新分配时才清除。
:::

关键点：**hash 在请求释放时不会被清除**，只有在 block 被淘汰并重新分配时才清除。这使得短时间内到达的相同前缀请求可以复用缓存。

## 混合模型的哈希粒度

对于包含多种 attention 类型的混合模型（如同时有 full attention 和 sliding window attention 的层），哈希和缓存查找需要特殊处理：

- **hash_block_size** = GCD（各 group block_size 的最大公约数），确保所有 attention group 的 block 边界对齐
- **max_cache_hit_length** 必须是 LCM（各 group block_size 的最小公倍数）的倍数
- 查找时采用**迭代搜索**：从最长的可能命中开始，逐步缩短，直到找到所有 group 都能接受的命中长度

```python
# vllm/v1/core/kv_cache_utils.py
# 混合模型：hash_block_size = GCD，max_cache_hit_length = LCM 的倍数
hash_block_size = math.gcd(*group_block_sizes)
lcm_block_size = math.lcm(*group_block_sizes)
```

## 关键配置

| 参数 | 默认值 | 说明 | 源码 |
|------|--------|------|------|
| `enable_prefix_caching` | True | 是否启用 prefix caching | kv_cache_utils.py |
| `prefix_caching_hash_algo` | "sha256" | 哈希算法（"sha256" 或 "builtin"） | kv_cache_utils.py |
| `hash_block_size` | None（= block_size） | 哈希粒度，混合模型自动取 GCD | kv_cache_utils.py |
| `cache_salt` | None | 用户指定的缓存盐值，强制隔离缓存 | kv_cache_manager.py |
