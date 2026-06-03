---
id: kv-cache
title: KV Cache
category: core
level: intermediate
status: ready
readingMinutes: 15
tags:
  - KV Cache
  - Block
  - Prefix Caching
codeRefs:
  - vllm/v1/core/block_pool.py
  - vllm/v1/attention/backends/flash_attn.py
  - vllm/v1/kv_cache_interface.py
  - vllm/v1/core/kv_cache_utils.py
  - vllm/v1/core/kv_cache_manager.py
  - vllm/v1/core/sched/scheduler.py
  - vllm/v1/core/kv_cache_coordinator.py
  - vllm/config/cache.py
heroText: GPU 显存分页管理、[前缀复用](term:相同 prompt 前缀的 KV block 只计算和保存一次，后续请求直接复用。)与驱逐策略
---

## 心智模型

想象你在图书馆管理书架。每本书（token 的 KV 向量）放在固定大小的书盒（block）里，书盒编号 0、1、2……。当新读者（请求）到来时，管理员（调度器）从空书盒堆里取书盒给他用；读者走了，书盒归还。如果两本书开头一样，后面的书盒可以**共享**——不用再存一份。

:::diagram kv-mental-model
```html
<div class="diagram-container">
  <div class="diagram" data-ref="vllm/v1/core/block_pool.py">
    <div class="diagram-title">KV Cache 分页管理</div>
    <div class="kv-mental-model">
      <div class="kv-gpu-memory">
        <div class="kv-gpu-title">GPU 显存 = 一排书盒（Blocks）</div>
        <div class="kv-blocks">
          <div class="kv-block kv-block-used">B0</div>
          <div class="kv-block kv-block-used">B1</div>
          <div class="kv-block kv-block-cached">B2</div>
          <div class="kv-block kv-block-free">B3</div>
          <div class="kv-block kv-block-free">B4</div>
          <div class="kv-block kv-block-cached">B5</div>
          <div class="kv-block kv-block-free">B6</div>
          <div class="kv-block kv-block-free">B7</div>
        </div>
      </div>
      <div class="kv-legend">
        <div class="kv-legend-item"><span class="kv-legend-dot kv-block-used"></span> 正在被某个请求使用</div>
        <div class="kv-legend-item"><span class="kv-legend-dot kv-block-cached"></span> 空闲但内容有缓存标记，可被新请求复用</div>
        <div class="kv-legend-item"><span class="kv-legend-dot kv-block-free"></span> 空闲，无缓存标记</div>
      </div>
      <div class="kv-data-structures">
        <div class="kv-ds">
          <div class="kv-ds-title">BlockPool</div>
          <div class="kv-ds-items">
            <div class="kv-ds-item">空闲书盒队列（按最久未用排序）</div>
            <div class="kv-ds-item">缓存标记 → 书盒 的查找表</div>
          </div>
        </div>
        <div class="kv-ds">
          <div class="kv-ds-title">每个请求的 Block Table</div>
          <div class="kv-ds-items">
            <div class="kv-ds-item">[B0, B1, B2, B5, ...]（逻辑顺序）</div>
            <div class="kv-ds-item">前缀相同的 block 可以被多个请求共享</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc kv-mental-model
KV Cache 分页管理示意图。GPU 显存被划分为一排固定大小的书盒（Blocks），编号 B0-B7。每个 block 有三种状态：正在使用（used，如 B0、B1）、缓存中（cached，如 B2、B5，空闲但带有缓存标记可被复用）、空闲（free，如 B3、B4、B6、B7，无缓存标记）。

BlockPool 维护两个数据结构：空闲书盒队列（按最久未用排序）和缓存标记到书盒的查找表。每个请求有自己的 Block Table，记录逻辑顺序如 [B0, B1, B2, B5, ...]，前缀相同的 block 可以被多个请求共享。

核心思路：分页（把 KV 向量按 block_size 分块管理）+ 共享（相同前缀的 block 只存一份）+ 驱逐（显存不够时淘汰最久没用的 block）。
:::

核心思路：**分页**（把 KV 向量按 block_size 分块管理）+ **共享**（相同前缀的 block 只存一份）+ **驱逐**（显存不够时淘汰最久没用的 block）。

### Block 里到底存了什么？

每个 block 是一块固定大小的 GPU 显存，存放 `block_size`（默认 16）个 token 的 **Key 和 Value 向量**。以 Llama-2-7B 为例（32 个 KV head，每个 head 128 维），一个 block 的形状是：

:::diagram kv-tensor-layout
```html
<div class="diagram-container">
  <div class="diagram" data-ref="vllm/v1/attention/backends/flash_attn.py">
    <div class="diagram-title">单个 Block 的存储结构</div>
    <div class="kv-tensor-layout">
      <div class="kv-tensor-dim" data-dim="kv">
        <div class="kv-tensor-dim-label">dim 0: K/V</div>
        <div class="kv-tensor-dim-cells">
          <div class="kv-tensor-cell kv-cell-k">K</div>
          <div class="kv-tensor-cell kv-cell-v">V</div>
        </div>
      </div>
      <div class="kv-tensor-bracket">×</div>
      <div class="kv-tensor-dim" data-dim="tokens">
        <div class="kv-tensor-dim-label">dim 1: block_size = 16</div>
        <div class="kv-tensor-dim-cells">
          <div class="kv-tensor-cell">t₀</div>
          <div class="kv-tensor-cell">t₁</div>
          <div class="kv-tensor-cell">...</div>
          <div class="kv-tensor-cell">t₁₅</div>
        </div>
      </div>
      <div class="kv-tensor-bracket">×</div>
      <div class="kv-tensor-dim" data-dim="heads">
        <div class="kv-tensor-dim-label">dim 2: num_kv_heads = 32</div>
        <div class="kv-tensor-dim-cells">
          <div class="kv-tensor-cell kv-cell-head">h₀</div>
          <div class="kv-tensor-cell kv-cell-head">h₁</div>
          <div class="kv-tensor-cell kv-cell-head">...</div>
          <div class="kv-tensor-cell kv-cell-head">h₃₁</div>
        </div>
      </div>
      <div class="kv-tensor-bracket">×</div>
      <div class="kv-tensor-dim" data-dim="dim">
        <div class="kv-tensor-dim-label">dim 3: head_size = 128</div>
        <div class="kv-tensor-dim-cells">
          <div class="kv-tensor-cell kv-cell-dim">d₀</div>
          <div class="kv-tensor-cell kv-cell-dim">d₁</div>
          <div class="kv-tensor-cell kv-cell-dim">...</div>
          <div class="kv-tensor-cell kv-cell-dim">d₁₂₇</div>
        </div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc kv-tensor-layout
单个 Block 的存储结构。一个 block 的张量形状为 (2, 16, 32, 128)，四个维度分别是：

- dim 0: K/V（2 层，分别存储 Key 和 Value）
- dim 1: block_size = 16（存储 16 个 token）
- dim 2: num_kv_heads = 32（32 个 KV head）
- dim 3: head_size = 128（每个 head 128 维）

完整的 KV cache 张量形状是 (num_blocks, 2, block_size, num_kv_heads, head_size)。可以理解为：一排书盒，每个书盒分上下两层（K 和 V），每层放 16 个 token 的 32 个 head 各 128 维向量。
:::

完整的 KV cache 张量形状就是 `(num_blocks, 2, block_size, num_kv_heads, head_size)`。可以理解为：**一排书盒，每个书盒分上下两层（K 和 V），每层放 16 个 token 的 32 个 head 各 128 维向量**。

### NHD vs HND：两种物理排布

上面的逻辑形状叫 **NHD**（N=block数, H=head数, D=维度），token 在 head 里面是连续的。另一种排布是 **HND**，把 head 维度挪到 token 前面，使得同一个 head 的所有 token 连续存放：

| 排布 | 物理形状（去掉 num_blocks 维度） | 谁连续 |
|------|----------------------------------|--------|
| **NHD**（默认） | `(2, 16, 32, 128)` | 同一个 token 的所有 head 连续 |
| **HND** | `(2, 32, 16, 128)` | 同一个 head 的所有 token 连续 |

HND 排布对某些 attention kernel 更友好（同一个 head 的 token 连续读取），可以通过 `VLLM_KV_CACHE_LAYOUT=HND` 开启。

### 一个 Block 占多少显存？

```python
# ref: vllm/v1/kv_cache_interface.py
# 单个 block 的字节数 = 2 × block_size × num_kv_heads × head_size × dtype_size
# 例：Llama-2-7B, fp16:
# = 2 × 16 × 32 × 128 × 2 = 262,144 字节 = 256 KB/block
```

## Block 的三种状态

每个 block 有一个**引用计数**（ref_cnt）和一个**缓存标记**（block_hash），组合出三种状态：

| 状态 | ref_cnt | block_hash | 大白话 |
|------|---------|------------|--------|
| 正在使用 | > 0 | 可能有 | 有人在读这本书，不能动 |
| 缓存中 | 0 | 有 | 没人读了，但书还在架上有标签，新读者可以复用 |
| 空闲 | 0 | 无 | 空书盒，随时可以装新书 |

**关键设计**：请求释放 block 时，ref_cnt 降到 0，block 进入空闲队列，但**不会清除缓存标记**。这意味着后续请求仍能通过前缀匹配找到它。只有当这个 block 被分配给新请求装不同内容时，旧标记才会被清除（驱逐）。

## Block Pool 四个核心操作

### 1. 分配：get_new_blocks

从空闲队列头部（最久未用端）取 block，清除旧缓存标记，设 ref_cnt=1：

```python
# ref: vllm/v1/core/block_pool.py
ret = self.free_block_queue.popleft_n(num_blocks)  # 取最久未用的
for block in ret:
    self._maybe_evict_cached_block(block)  # 清除旧缓存标记
    block.ref_cnt += 1                      # 标记为"正在使用"
```

### 2. 释放：free_blocks

ref_cnt 减 1。只有减到 0 的 block 才回到空闲队列尾部（最近释放端）：

```python
# ref: vllm/v1/core/block_pool.py
for block in blocks_list:
    block.ref_cnt -= 1
# 只有 ref_cnt==0 的 block 才归还空闲队列
self.free_block_queue.append_n(
    [b for b in blocks_list if b.ref_cnt == 0 and not b.is_null]
)
```

### 3. Touch：缓存命中时引用

新请求发现前缀匹配时，对已有 block 的 ref_cnt 加 1，并从空闲队列中移除（防止被驱逐）：

```python
# ref: vllm/v1/core/block_pool.py
for block in blocks:
    if block.ref_cnt == 0 and not block.is_null:
        self.free_block_queue.remove(block)  # 从空闲队列移除
    block.ref_cnt += 1                       # 引用计数 +1
```

### 4. 驱逐：_maybe_evict_cached_block

分配新 block 时，如果它还带着旧缓存标记，需要清除：

```python
# ref: vllm/v1/core/block_pool.py
if block.block_hash is None:
    return False  # 没有缓存标记，不用驱逐
self.cached_block_hash_to_block.pop(block_hash, block.block_id)
block.reset_hash()  # 清除标记
```

## LRU 驱逐策略

空闲队列是一个**双向链表**（自定义实现，支持 O(1) 中间删除），按 LRU 排序：

:::diagram kv-lru-queue
```html
<div class="diagram-container">
  <div class="diagram" data-ref="vllm/v1/core/kv_cache_utils.py">
    <div class="diagram-title">空闲队列的 LRU 顺序</div>
    <div class="kv-lifecycle">
      <div class="kv-lc-state" data-state="free">头部<br><small>最久未用<br>先被分配/驱逐</small></div>
      <div class="kv-lc-arrow">← 更久 →</div>
      <div class="kv-lc-state" data-state="cached-free">中间<br><small>较早释放</small></div>
      <div class="kv-lc-arrow">← 更久 →</div>
      <div class="kv-lc-state" data-state="free">尾部<br><small>最近释放<br>最后被驱逐</small></div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc kv-lru-queue
空闲队列的 LRU 顺序示意图。空闲队列是一个双向链表，按 LRU（最久未用）排序：

- 头部：最久未用的 block，最先被分配或驱逐
- 中间：较早释放的 block
- 尾部：最近释放的 block，最后被驱逐

驱逐顺序规则：
1. 最久未用的先驱逐：从队列头部取 block 分配，头部是最久没被 touch 的
2. 同一请求的尾部 block 先驱逐：释放时按逆序追加（尾部 block 包含更多 hash token，驱逐它对前缀链的破坏更小）
3. Touch 保护：被 touch 的 block 从空闲队列中移除，不会被驱逐
:::

### 驱逐顺序规则

- **最久未用的先驱逐**：从队列头部取 block 分配，头部是最久没被 touch 的。
- **同一请求的尾部 block 先驱逐**：释放时按逆序追加（尾部 block 包含更多 hash token，驱逐它对前缀链的破坏更小）。
- **Touch 保护**：被 touch 的 block 从空闲队列中移除，不会被驱逐。

## Prefix Caching：前缀复用

如果两个请求的开头一样（比如相同的 system prompt），后面的请求可以**直接复用**前面请求的 KV cache，跳过重复计算。

### 链式哈希：怎么判断"开头一样"？

每个 block 的哈希 = hash(前一个 block 的哈希, 本 block 的 token)。这形成一条**哈希链**：

:::diagram kv-hash-chain
```html
<div class="diagram-container">
  <div class="diagram" data-ref="vllm/v1/core/kv_cache_utils.py">
    <div class="diagram-title">链式哈希结构</div>
    <div class="kv-hash-chain">
      <div class="kv-hash-block">
        <div class="kv-hash-block-title">Block 0</div>
        <div class="kv-hash-block-detail">hash₀ = H(种子, tokens[0:16])</div>
      </div>
      <div class="kv-hash-arrow">→</div>
      <div class="kv-hash-block">
        <div class="kv-hash-block-title">Block 1</div>
        <div class="kv-hash-block-detail">hash₁ = H(hash₀, tokens[16:32])</div>
      </div>
      <div class="kv-hash-arrow">→</div>
      <div class="kv-hash-block">
        <div class="kv-hash-block-title">Block 2</div>
        <div class="kv-hash-block-detail">hash₂ = H(hash₁, tokens[32:48])</div>
      </div>
      <div class="kv-hash-arrow">→</div>
      <div class="kv-hash-block">
        <div class="kv-hash-block-title">Block N</div>
        <div class="kv-hash-block-detail">hashₙ = H(hashₙ₋₁, tokens[...])</div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc kv-hash-chain
链式哈希结构示意图。每个 block 的哈希值依赖前一个 block 的哈希值和本 block 的 token：

- Block 0: hash₀ = H(种子, tokens[0:16])
- Block 1: hash₁ = H(hash₀, tokens[16:32])
- Block 2: hash₂ = H(hash₁, tokens[32:48])
- Block N: hashₙ = H(hashₙ₋₁, tokens[...])

因为每个哈希依赖前面的哈希，所以如果 Block 3 命中，那 Block 0-2 一定也命中（向下封闭）。查找时从左到右扫描，遇到第一个未命中就停。

除了 token 本身，哈希还会包含：多模态特征、LoRA 名称、cache salt 等，确保不同上下文不会错误命中。
:::

因为每个哈希依赖前面的哈希，所以**如果 Block 3 命中，那 Block 0-2 一定也命中**（向下封闭）。查找时从左到右扫描，遇到第一个未命中就停。

### 什么会影响哈希？

除了 token 本身，哈希还会包含：多模态特征、LoRA 名称、cache salt 等，确保不同上下文不会错误命中。

## 共享 Block 的完整流程

多个请求可以引用同一个 block。只有当**所有引用都释放**后，block 才能被驱逐。

:::diagram cache-flow-html
```html
<div class="cache-flow">
  <div class="cache-step" data-step="1">
    <div class="cache-step-num">1</div>
    <div class="cache-step-content">
      <div class="cache-step-title">请求 A 计算了前 10 个 block</div>
      <div class="cache-step-desc">每个 block 设置缓存标记（hash），ref_cnt = 1</div>
    </div>
  </div>
  <div class="cache-step" data-step="2">
    <div class="cache-step-num">2</div>
    <div class="cache-step-content">
      <div class="cache-step-title">请求 B 到来，前缀和 A 一样</div>
      <div class="cache-step-desc">find_longest_cache_hit 找到 A 的 10 个 block，touch() 把 ref_cnt 加到 2</div>
    </div>
  </div>
  <div class="cache-step" data-step="3">
    <div class="cache-step-num">3</div>
    <div class="cache-step-content">
      <div class="cache-step-title">A 和 B 共享这 10 个 block</div>
      <div class="cache-step-desc">ref_cnt=2，block 不会进入空闲队列，不会被驱逐</div>
    </div>
  </div>
  <div class="cache-step" data-step="4">
    <div class="cache-step-num">4</div>
    <div class="cache-step-content">
      <div class="cache-step-title">请求 A 完成，释放 block</div>
      <div class="cache-step-desc">ref_cnt 从 2 降到 1，block 不回空闲队列（B 还在用），缓存标记保留</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc cache-flow-html
共享 Block 的完整流程，展示多个请求如何共享同一个 block：

步骤 1：请求 A 计算了前 10 个 block，每个 block 设置缓存标记（hash），ref_cnt = 1。

步骤 2：请求 B 到来，前缀和 A 一样。find_longest_cache_hit 找到 A 的 10 个 block，touch() 把 ref_cnt 加到 2。

步骤 3：A 和 B 共享这 10 个 block。ref_cnt=2，block 不会进入空闲队列，不会被驱逐。

步骤 4：请求 A 完成，释放 block。ref_cnt 从 2 降到 1，block 不回空闲队列（B 还在用），缓存标记保留。

只有当所有引用都释放后（ref_cnt 降到 0），block 才能被驱逐。
:::

## Block 生命周期

:::diagram kv-lifecycle
```html
<div class="diagram-container">
  <div class="diagram" data-ref="vllm/v1/core/block_pool.py">
    <div class="diagram-title">Block 从分配到驱逐的完整路径</div>
    <div class="kv-lifecycle">
      <div class="kv-lc-state" data-state="free">Free<br><small>空闲，无标记</small></div>
      <div class="kv-lc-arrow">分配</div>
      <div class="kv-lc-state" data-state="used">In Use<br><small>正在使用</small></div>
      <div class="kv-lc-arrow">计算完成，写入缓存标记</div>
      <div class="kv-lc-state" data-state="cached-used">Cached + In Use<br><small>有标记，正在使用</small></div>
      <div class="kv-lc-arrow">请求释放</div>
      <div class="kv-lc-state" data-state="cached-free">Cached + Free<br><small>有标记，空闲（可被新请求复用）</small></div>
      <div class="kv-lc-arrow">被分配给新请求（驱逐旧标记）</div>
      <div class="kv-lc-state" data-state="free">Free<br><small>空闲，无标记</small></div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc kv-lifecycle
Block 生命周期示意图，展示 block 从分配到驱逐的完整路径：

1. Free（空闲，无标记）→ 分配 → In Use（正在使用）
2. In Use → 计算完成，写入缓存标记 → Cached + In Use（有标记，正在使用）
3. Cached + In Use → 请求释放 → Cached + Free（有标记，空闲，可被新请求复用）
4. Cached + Free → 被分配给新请求（驱逐旧标记）→ Free（空闲，无标记）

注意 "Cached + Free" 状态：block 虽然空闲，但因为还带着缓存标记，新请求如果前缀匹配可以直接 touch 复用，不用重新计算。只有当显存紧张、这个 block 被分配给其他请求时，旧标记才被清除。
:::

注意 "Cached + Free" 状态：block 虽然空闲，但因为还带着缓存标记，新请求如果前缀匹配可以直接 touch 复用，不用重新计算。只有当显存紧张、这个 block 被分配给其他请求时，旧标记才被清除。

## allocate_slots：分配的全过程

当调度器决定调度一个请求时，调用 `allocate_slots()` 为它分配 KV cache 空间。分三步：

### Block 布局

```python
# ref: vllm/v1/core/kv_cache_manager.py
# | 已计算 | 新命中本地缓存 | 外部缓存 | 待计算 | 投机解码预留 |
# | comp  |   new_comp    | ext_comp |  new  |   lookahead   |
```

### 三步流程

| 步骤 | 做什么 | 如果失败 |
|------|--------|----------|
| 1. 检查容量 | 释放滑动窗口外不需要的 block，计算还需要多少新 block，和空闲数比较 | 返回 None → 调度器触发抢占 |
| 2. 处理前缀命中 | touch 本地缓存命中的 block（ref_cnt++），分配外部缓存 token 的新 block | — |
| 3. 分配新 block | 从空闲队列分配新 block，写入缓存标记 | — |

## 与 Scheduler 的交互

调度器和 KV cache 的交互很简单，只有三个入口：

| 场景 | 操作 | 说明 |
|------|------|------|
| 新请求调度 | `get_computed_blocks()` → `allocate_slots()` | 先查前缀命中，再分配新 block |
| 已运行请求 | `allocate_slots()` | 直接分配新 block（decode 每步只需 1 个新 token） |
| 抢占/完成 | `free()` | 释放所有 block，ref_cnt 降为 0 |

`vllm/v1/core/sched/scheduler.py`

## 混合模型：多种 Attention 共存

有些模型同时有 Full Attention 和 Sliding Window Attention，它们的 block 大小和缓存策略不同。vLLM 把层按 attention 类型**分组**管理：

| 场景 | 用的 Coordinator |
|------|------------------|
| 只有一种 attention | `UnitaryKVCacheCoordinator` |
| 多种 attention（Full + SWA 等） | `HybridKVCacheCoordinator` |
| Prefix caching 禁用 | `KVCacheCoordinatorNoPrefixCache` |

混合模型的关键约束：缓存命中长度必须是所有 group block 大小的**最小公倍数**的倍数，哈希粒度是**最大公约数**。查找时用迭代法求各类型都能接受的最长命中长度。

`vllm/v1/core/kv_cache_coordinator.py`

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `block_size` | 16 | 每个 block 存几个 token 的 KV |
| `gpu_memory_utilization` | 0.92 | GPU 显存中 KV cache 占比 |
| `enable_prefix_caching` | True | 是否启用前缀复用 |
| `prefix_caching_hash_algo` | "sha256" | 哈希算法 |
| `hash_block_size` | None (= block_size) | 哈希粒度，混合模型时自动取 GCD |

`vllm/config/cache.py`
