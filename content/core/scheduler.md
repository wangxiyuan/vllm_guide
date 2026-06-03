---
id: scheduler
title: 调度原理
category: core
level: intermediate
status: ready
readingMinutes: 18
tags:
  - Scheduler
  - Continuous Batching
  - Preemption
  - Chunked Prefill
codeRefs:
  - vllm/v1/core/sched/scheduler.py
  - vllm/v1/core/sched/output.py
  - vllm/v1/request.py
  - vllm/v1/core/sched/utils.py
  - vllm/config/scheduler.py
  - vllm/v1/core/sched/request_queue.py
  - vllm/v1/engine/core.py
heroText: 连续批处理、[token budget](term:调度器每一轮最多允许处理的新 token 数，是吞吐和延迟的核心控制阀。)、抢占与状态更新
---

## 心智模型

vLLM v1 调度器的核心思想：**每个请求只有一个 `num_computed_tokens` 计数器**，调度器的任务是在每一步为请求分配 token，让 `num_computed_tokens` 追上 `num_tokens_with_spec`。这个模型统一了 prefill、decode、chunked prefill、prefix caching、speculative decoding 等所有场景。

:::diagram sched-html
```html
<div class="sched-mental-model">
  <div class="sched-queue-group">
    <div class="sched-queue" data-queue="waiting">
      <div class="sched-queue-title">waiting</div>
      <div class="sched-queue-desc">等待调度的新请求</div>
    </div>
    <div class="sched-queue" data-queue="skipped">
      <div class="sched-queue-title">skipped_waiting</div>
      <div class="sched-queue-desc">被阻塞的请求（KV 传输、grammar 编译、流式输入）</div>
    </div>
  </div>
  <div class="sched-arrow-down">
    <span class="sched-arrow-label">schedule()</span>
    <span class="sched-arrow-icon">↓</span>
  </div>
  <div class="sched-queue" data-queue="running">
    <div class="sched-queue-title">running</div>
    <div class="sched-queue-desc">正在执行的请求（FIFO 顺序）</div>
  </div>
  <div class="sched-arrow-down">
    <span class="sched-arrow-label">update_from_output()</span>
    <span class="sched-arrow-icon">↓</span>
  </div>
  <div class="sched-finish-group">
    <div class="sched-finish" data-status="finished">
      <div class="sched-finish-title">FINISHED_*</div>
      <div class="sched-finish-desc">完成/中止/长度限制</div>
    </div>
    <div class="sched-finish" data-status="preempted">
      <div class="sched-finish-title">PREEMPTED → WAITING</div>
      <div class="sched-finish-desc">抢占后重新入队</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc sched-html
调度器心智模型展示了请求在三个主要队列间的流转：请求首先进入 waiting 队列（等待调度的新请求）和 skipped_waiting 队列（被阻塞的请求，包括等待 KV 传输、grammar 编译、流式输入的请求）。通过 schedule() 操作，请求从等待队列进入 running 队列（正在执行的请求，FIFO 顺序）。通过 update_from_output() 操作，运行中的请求流转到结束状态：FINISHED_*（完成/中止/长度限制）或 PREEMPTED → WAITING（抢占后重新入队等待）。
:::

### 核心约束

| 约束 | 默认值 | 说明 |
|------|--------|------|
| `max_num_scheduled_tokens` | 2048 | 单步最大调度 token 数（token budget） |
| `max_num_seqs` | 128 | 最大并发请求数 |
| KV blocks | GPU 显存决定 | 可用 KV cache 块数 |
| `encoder_compute_budget` | = max_num_batched_tokens | 多模态 encoder 计算预算 |

`vllm/config/scheduler.py`

### 调度策略

vLLM 支持两种调度策略：

- **FCFS（默认）**：先到先服务，`waiting` 队列用 deque 实现，新请求追加到尾部，调度时从头部弹出。
- **PRIORITY**：优先级调度，用最小堆实现，按 `(priority, arrival_time, request_id)` 排序，priority 值越小越优先。

`vllm/v1/core/sched/request_queue.py`

## 调度全景图

每次 `EngineCore.step()` 调用 `scheduler.schedule()`，返回一个 `SchedulerOutput`，包含本步要处理的所有请求信息。

:::diagram sched-flow-html
```html
<div class="sched-flow">
  <div class="sched-phase" data-phase="running">
    <div class="sched-phase-title">Phase 1: 调度 RUNNING 请求</div>
    <div class="sched-phase-steps">
      <div class="sched-step">1. 遍历 running 列表</div>
      <div class="sched-step">2. 计算 <code>num_new_tokens = num_tokens_with_spec - num_computed_tokens</code></div>
      <div class="sched-step">3. 应用 chunked prefill 阈值限制</div>
      <div class="sched-step">4. 调用 <code>allocate_slots()</code> 分配 KV blocks</div>
      <div class="sched-step">5. 若分配失败 → 抢占最低优先级请求</div>
      <div class="sched-step">6. 更新 token_budget</div>
    </div>
  </div>
  <div class="sched-phase-arrow">↓</div>
  <div class="sched-phase" data-phase="waiting">
    <div class="sched-phase-title">Phase 2: 调度 WAITING 请求</div>
    <div class="sched-phase-steps">
      <div class="sched-step">1. 前提：本步无抢占发生（<code>not preempted_reqs</code>）</div>
      <div class="sched-step">2. 检查 <code>len(running) &lt; max_num_seqs</code></div>
      <div class="sched-step">3. 查询 prefix cache: <code>get_computed_blocks()</code></div>
      <div class="sched-step">4. 查询外部 KV connector 缓存命中</div>
      <div class="sched-step">5. 计算 <code>num_new_tokens = num_tokens - num_computed_tokens</code></div>
      <div class="sched-step">6. 调用 <code>allocate_slots()</code> 分配 KV blocks</div>
      <div class="sched-step">7. 若分配失败 → break（不抢占，停止调度新请求）</div>
      <div class="sched-step">8. 若异步加载 → 设为 <code>WAITING_FOR_REMOTE_KVS</code></div>
      <div class="sched-step">9. 否则加入 running 列表</div>
    </div>
  </div>
  <div class="sched-phase-arrow">↓</div>
  <div class="sched-phase" data-phase="output">
    <div class="sched-phase-title">Phase 3: 构造 SchedulerOutput</div>
    <div class="sched-phase-steps">
      <div class="sched-step">1. 收集 scheduled_new_reqs / scheduled_resumed_reqs / scheduled_running_reqs</div>
      <div class="sched-step">2. 计算 num_common_prefix_blocks（cascade attention）</div>
      <div class="sched-step">3. 构造 NewRequestData / CachedRequestData</div>
      <div class="sched-step">4. 调用 <code>_update_after_schedule()</code> 更新 num_computed_tokens</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc sched-flow-html
schedule() 主流程分三个阶段：

**Phase 1: 调度 RUNNING 请求** — 遍历 running 列表，计算每个请求的 num_new_tokens = num_tokens_with_spec - num_computed_tokens，应用 chunked prefill 阈值限制，调用 allocate_slots() 分配 KV blocks，若分配失败则抢占最低优先级请求，更新 token_budget。

**Phase 2: 调度 WAITING 请求** — 前提是本步无抢占发生。检查 len(running) < max_num_seqs，查询 prefix cache（get_computed_blocks()），查询外部 KV connector 缓存命中，计算 num_new_tokens = num_tokens - num_computed_tokens，调用 allocate_slots() 分配 KV blocks，若分配失败则 break（不抢占，停止调度新请求），若异步加载则设为 WAITING_FOR_REMOTE_KVS，否则加入 running 列表。

**Phase 3: 构造 SchedulerOutput** — 收集 scheduled_new_reqs / scheduled_resumed_reqs / scheduled_running_reqs，计算 num_common_prefix_blocks（cascade attention），构造 NewRequestData / CachedRequestData，调用 _update_after_schedule() 更新 num_computed_tokens。
:::

### SchedulerOutput 结构

```python
# vllm/v1/core/sched/output.py
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]      # 首次调度的请求
    scheduled_cached_reqs: CachedRequestData      # 已调度过的请求（只发 diff）
    num_scheduled_tokens: dict[str, int]          # req_id → 本步 token 数
    total_num_scheduled_tokens: int               # 总 token 数
    scheduled_spec_decode_tokens: dict[str, list[int]]  # spec decode token ids
    scheduled_encoder_inputs: dict[str, list[int]]      # encoder 输入索引
    num_common_prefix_blocks: list[int]           # 每 KV cache group 的公共前缀块数
    finished_req_ids: set[str]                    # 上一步完成的请求
    preempted_req_ids: set[str]                   # 本步被抢占的请求
    free_encoder_mm_hashes: list[str]             # 待释放的 encoder 输出
    kv_connector_metadata: KVConnectorMetadata | None   # KV connector 元数据
    new_block_ids_to_zero: list[int] | None       # 需要清零的新块
```

## 请求状态流转

每个请求在 Scheduler 中经历以下状态：

:::diagram state-flow-html
```html
<div class="state-flow">
  <div class="state-node state-waiting">WAITING</div>
  <div class="state-arrow">schedule()</div>
  <div class="state-node state-running">RUNNING</div>
  <div class="state-arrow">KV 不足</div>
  <div class="state-node state-preempted">PREEMPTED</div>
  <div class="state-arrow">重新入队</div>
  <div class="state-node state-waiting">WAITING</div>
</div>
<div class="state-flow">
  <div class="state-node state-running">RUNNING</div>
  <div class="state-arrow">完成/中止</div>
  <div class="state-node state-finished">FINISHED_*</div>
</div>
```
:::

:::diagram-desc state-flow-html
请求状态流转有两条路径：
1. WAITING → schedule() → RUNNING → KV 不足 → PREEMPTED → 重新入队 → WAITING（抢占循环）
2. RUNNING → 完成/中止 → FINISHED_*（正常完成）
:::

### 状态枚举

```python
# vllm/v1/request.py
class RequestStatus(enum.IntEnum):
    # 活跃/等待状态
    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # 完成状态（> PREEMPTED 的都被视为 finished）
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()
```

### 特殊等待状态

- `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`：结构化输出的 grammar 正在编译，编译完成后转为 `WAITING`。
- `WAITING_FOR_REMOTE_KVS`：等待远程 KV 传输完成（Disaggregated Prefill 场景），传输完成后转为 `WAITING` 或 `PREEMPTED`。
- `WAITING_FOR_STREAMING_REQ`：等待流式输入到达（多轮对话场景），新输入到达后转为 `WAITING`。

### 状态转换关键点

| 转换 | 触发条件 | 代码位置 |
|------|----------|----------|
| WAITING → RUNNING | 成功分配 KV blocks 并加入 running 列表 | `scheduler.py` |
| RUNNING → PREEMPTED | `allocate_slots()` 返回 None | `scheduler.py` |
| PREEMPTED → WAITING | 调用 `_preempt_request()` 重新入队 | `scheduler.py` |
| RUNNING → FINISHED_* | EOS / stop token / 长度限制 / 重复检测 | `utils.py` |
| WAITING_FOR_REMOTE_KVS → WAITING/PREEMPTED | KV 传输完成 | `scheduler.py` |

## 抢占机制

vLLM v1 **只支持重计算抢占**，不支持 swap-to-CPU。当 KV blocks 不足时，调度器会抢占一个 running 请求，释放其所有 KV blocks，重置 `num_computed_tokens = 0`，然后将其放回 waiting 队列头部。

### 抢占流程

```python
# vllm/v1/core/sched/scheduler.py
def _preempt_request(self, request: Request, timestamp: float) -> None:
    assert request.status == RequestStatus.RUNNING
    self.kv_cache_manager.free(request)         # 释放所有 KV blocks
    self.encoder_cache_manager.free(request)    # 释放 encoder cache
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0             # 重置 → 下次重新计算
    if request.spec_token_ids:
        request.spec_token_ids = []             # 清空 spec tokens
    request.num_preemptions += 1
    self.waiting.prepend_request(request)       # 放回 waiting 队列头部
```

### 抢占策略

- **FCFS 模式**：抢占 running 列表中**最后一个**请求（最近加入的，LIFO）。

```python
preempted_req = self.running.pop()  # 移除最后一个
```

- **PRIORITY 模式**：抢占**优先级最低**的请求（priority 值最大、到达时间最晚）。

```python
preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
```

### 抢占时的回滚

如果被抢占的请求在本步已经被调度过（在 `scheduled_running_reqs` 中），需要回滚其分配：

```python
# vllm/v1/core/sched/scheduler.py
if preempted_req in scheduled_running_reqs:
    scheduled_running_reqs.remove(preempted_req)
    token_budget += num_scheduled_tokens.pop(preempted_req_id)  # 归还 token budget
    req_to_new_blocks.pop(preempted_req_id)                     # 归还 blocks
    scheduled_spec_decode_tokens.pop(preempted_req_id, None)
    # 归还 encoder compute budget
    if preempted_encoder_inputs := scheduled_encoder_inputs.pop(preempted_req_id, None):
        num_embeds_to_restore = sum(
            preempted_req.get_num_encoder_embeds(i) for i in preempted_encoder_inputs
        )
        encoder_compute_budget += num_embeds_to_restore
```

## Chunked Prefill

Chunked prefill 允许将长 prefill 请求分块处理，避免单个长请求阻塞所有 decode 请求。vLLM v1 默认启用 chunked prefill。

### 分块逻辑

```python
# vllm/v1/core/sched/scheduler.py
# RUNNING 请求
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
num_new_tokens = min(num_new_tokens, token_budget)
```

### 相关配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enable_chunked_prefill` | True | 是否启用 chunked prefill |
| `long_prefill_token_threshold` | 0 (auto = 4% max_model_len) | "长 prefill" 的阈值，超过此值会被分块 |
| `max_num_partial_prefills` | 1 | 最大并发部分 prefill 请求数 |
| `max_long_partial_prefills` | 1 | 最大并发长 prefill 请求数（允许短请求插队） |

### is_prefill_chunk 标记

在 `_update_after_schedule()` 中设置，用于判断请求是否仍在 prefill 阶段：

```python
# vllm/v1/core/sched/scheduler.py
request.is_prefill_chunk = request.num_computed_tokens < (
    request.num_tokens + request.num_output_placeholders
)
```

## Prefix Caching 对调度的影响

Prefix caching 显著影响调度决策：缓存命中可以减少需要调度的 token 数，甚至让请求直接跳过 prefill。

### 缓存查询流程

:::diagram cache-flow-html
```html
<div class="cache-flow">
  <div class="cache-step" data-step="1">
    <div class="cache-step-num">1</div>
    <div class="cache-step-content">
      <div class="cache-step-title">请求到达，num_computed_tokens == 0</div>
      <div class="cache-step-desc">新请求首次进入调度</div>
    </div>
  </div>
  <div class="cache-step" data-step="2">
    <div class="cache-step-num">2</div>
    <div class="cache-step-content">
      <div class="cache-step-title">get_computed_blocks(request)</div>
      <div class="cache-step-desc">查询本地 prefix cache，返回命中的 blocks 和 token 数</div>
    </div>
  </div>
  <div class="cache-step" data-step="3">
    <div class="cache-step-num">3</div>
    <div class="cache-step-content">
      <div class="cache-step-title">connector.get_num_new_matched_tokens()</div>
      <div class="cache-step-desc">查询外部 KV connector 缓存（如 LMCache、P/D）</div>
    </div>
  </div>
  <div class="cache-step" data-step="4">
    <div class="cache-step-num">4</div>
    <div class="cache-step-content">
      <div class="cache-step-title">num_computed_tokens = local + external</div>
      <div class="cache-step-desc">计算总命中 token 数</div>
    </div>
  </div>
  <div class="cache-step" data-step="5">
    <div class="cache-step-num">5</div>
    <div class="cache-step-content">
      <div class="cache-step-title">num_new_tokens = num_tokens - num_computed_tokens</div>
      <div class="cache-step-desc">只需调度未命中的 token</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc cache-flow-html
Prefix Cache 查询流程分五步：
1. 请求到达时 num_computed_tokens == 0，新请求首次进入调度。
2. 调用 get_computed_blocks(request) 查询本地 prefix cache，返回命中的 blocks 和 token 数。
3. 调用 connector.get_num_new_matched_tokens() 查询外部 KV connector 缓存（如 LMCache、P/D）。
4. 计算 num_computed_tokens = local + external，即总命中 token 数。
5. 计算 num_new_tokens = num_tokens - num_computed_tokens，只需调度未命中的 token。
:::

### 关键代码

```python
# vllm/v1/core/sched/scheduler.py
if request.num_computed_tokens == 0:
    # 查询本地缓存
    new_computed_blocks, num_new_local_computed_tokens = (
        self.kv_cache_manager.get_computed_blocks(request)
    )
    
    # 查询外部缓存（KV connector）
    if self.connector is not None:
        ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
            request, num_new_local_computed_tokens
        )
        num_external_computed_tokens = ext_tokens or 0
    
    # 总命中 token 数
    num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
```

### 全命中场景

如果 `num_computed_tokens == request.num_tokens`（全命中），需要重新计算最后一个 token 以便采样：

```python
# vllm/v1/core/sched/scheduler.py
if request.num_computed_tokens == request.num_tokens:
    request.num_computed_tokens = request.num_tokens - 1  # 重算最后一个 token
```

## update_from_output 流程

模型执行完成后，`update_from_output()` 更新请求状态、处理生成的 token、检查停止条件。

### 主要步骤

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 处理 spec decode 结果 | 计算 accepted/rejected token 数，调整 num_computed_tokens |
| 2 | 释放 encoder inputs | 已处理的 encoder 输出从 cache 中释放 |
| 3 | 追加 output tokens | 调用 `_update_request_with_output()` |
| 4 | 检查停止条件 | 调用 `check_stop()` 检查 EOS/stop token/长度限制 |
| 5 | 处理 structured output | grammar 验证生成的 token |
| 6 | 处理完成的请求 | 调用 `_handle_stopped_request()`，释放资源 |
| 7 | 更新 KV connector 状态 | 处理 finished_recving/finished_sending |

### check_stop 停止条件

```python
# vllm/v1/core/sched/utils.py
def check_stop(request: Request, max_model_len: int) -> bool:
    # 1. 检查最小 token 数
    if request.num_output_tokens < sampling_params.min_tokens:
        return False
    
    # 2. 检查 EOS token
    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True
    
    # 3. 检查 stop token ids
    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        return True
    
    # 4. 检查长度限制
    if request.num_tokens >= max_model_len or request.num_output_tokens >= request.max_tokens:
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True
    
    # 5. 检查重复模式
    if check_sequence_repetition(request.output_token_ids, repetition_detection):
        request.status = RequestStatus.FINISHED_REPETITION
        return True
    
    return False
```

### 流式请求处理

对于 resumable 的流式请求（多轮对话），`_handle_stopped_request()` 不会直接结束，而是更新请求状态等待下一轮输入：

```python
# vllm/v1/core/sched/scheduler.py
def _handle_stopped_request(self, request: Request) -> bool:
    if not request.resumable:
        return True  # 非流式请求，直接结束
    
    if request.streaming_queue:
        update = request.streaming_queue.popleft()
        if update is None:
            return True  # 流式请求结束
        self._update_request_as_session(request, update)  # 更新请求，等待下一轮
    else:
        request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
        self.num_waiting_for_streaming_input += 1
    
    self._enqueue_waiting_request(request)  # 重新加入等待队列
    return False  # 请求未结束，继续等待
```

## Async Scheduling

异步调度允许调度与执行重叠，提高 GPU 利用率。当 `async_scheduling=True` 时，使用 `AsyncScheduler`。

### 核心机制

- **num_output_placeholders**：为 decode 请求预分配的输出占位符，允许在当前步执行时调度下一步。
- **提前终止检测**：如果 `num_computed_tokens + 2 - num_output_placeholders >= num_prompt_tokens + max_tokens`，跳过调度以避免无效步骤。

### 关键代码

```python
# vllm/v1/core/sched/scheduler.py
# 检查是否需要跳过调度（已达到 max_tokens）
if (
    request.num_output_placeholders > 0
    and request.num_computed_tokens + 2 - request.num_output_placeholders
    >= request.num_prompt_tokens + request.max_tokens
):
    # 避免调度额外的步骤
    req_index += 1
    continue
```

## 与 EngineCore 的交互

:::diagram engine-step-flow-html
```html
<div class="engine-step-flow">
  <div class="engine-step" data-step="1">
    <div class="engine-step-num">1</div>
    <div class="engine-step-content">
      <div class="engine-step-title">scheduler.schedule()</div>
      <div class="engine-step-desc">生成 SchedulerOutput</div>
    </div>
  </div>
  <div class="engine-step" data-step="2">
    <div class="engine-step-num">2</div>
    <div class="engine-step-content">
      <div class="engine-step-title">model_executor.execute_model()</div>
      <div class="engine-step-desc">提交到 GPU 执行（异步）</div>
    </div>
  </div>
  <div class="engine-step" data-step="3">
    <div class="engine-step-num">3</div>
    <div class="engine-step-content">
      <div class="engine-step-title">scheduler.get_grammar_bitmask()</div>
      <div class="engine-step-desc">获取 structured output bitmask</div>
    </div>
  </div>
  <div class="engine-step" data-step="4">
    <div class="engine-step-num">4</div>
    <div class="engine-step-content">
      <div class="engine-step-title">future.result()</div>
      <div class="engine-step-desc">等待 GPU 执行完成</div>
    </div>
  </div>
  <div class="engine-step" data-step="5">
    <div class="engine-step-num">5</div>
    <div class="engine-step-content">
      <div class="engine-step-title">model_executor.sample_tokens()</div>
      <div class="engine-step-desc">采样生成 token（如需要）</div>
    </div>
  </div>
  <div class="engine-step" data-step="6">
    <div class="engine-step-num">6</div>
    <div class="engine-step-content">
      <div class="engine-step-title">scheduler.update_from_output()</div>
      <div class="engine-step-desc">更新请求状态，返回 EngineCoreOutputs</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc engine-step-flow-html
EngineCore.step() 主循环包含六个步骤：
1. scheduler.schedule() — 生成 SchedulerOutput。
2. model_executor.execute_model() — 提交到 GPU 执行（异步）。
3. scheduler.get_grammar_bitmask() — 获取 structured output bitmask。
4. future.result() — 等待 GPU 执行完成。
5. model_executor.sample_tokens() — 采样生成 token（如需要）。
6. scheduler.update_from_output() — 更新请求状态，返回 EngineCoreOutputs。
:::

### PP 场景：step_with_batch_queue

当 `pipeline_parallel_size > 1` 时，使用 batch queue 实现调度与执行的重叠。核心流程：

1. 如果 batch queue 未满，调度新 batch 并提交到 executor（非阻塞），将 future 加入 queue。
2. 如果 queue 已满或无新请求可调度，阻塞等待 queue 中最早的 batch 完成。
3. 调用 `update_from_output()` 更新已完成 batch 的状态。

关键设计：填充 batch queue 的优先级高于获取模型输出。如果新 batch 成功提交且 queue 仍未满，直接返回空输出，不阻塞等待。

`vllm/v1/engine/core.py`

### add_request / abort_requests

EngineCore 通过以下方法与调度器交互：

- `add_request(request)`：将请求加入调度器的 waiting 队列。如果是重复 request_id 且请求是 resumable 的，则作为流式更新处理。
- `abort_requests(request_ids)`：调用 `scheduler.finish_requests()`，将请求标记为 `FINISHED_ABORTED`，从 running/waiting 队列中移除并释放资源。

`vllm/v1/engine/core.py`

## 关键配置汇总

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `max_num_batched_tokens` | 2048 | 单次迭代最大处理 token 数 |
| `max_num_scheduled_tokens` | = max_num_batched_tokens | 调度器单步最大发出 token 数 |
| `max_num_seqs` | 128 | 最大并发请求数 |
| `policy` | "fcfs" | 调度策略：fcfs / priority |
| `enable_chunked_prefill` | True | 是否启用 chunked prefill |
| `long_prefill_token_threshold` | 0 (auto) | 长 prefill 阈值 |
| `scheduler_reserve_full_isl` | True | 准入前检查完整序列是否能放入 KV cache |
| `async_scheduling` | None (auto) | 是否启用异步调度 |
| `disable_chunked_mm_input` | False | 禁止部分调度多模态输入 |

`vllm/config/scheduler.py`
