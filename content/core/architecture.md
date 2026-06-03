---
id: architecture
title: 架构总览
category: core
level: beginner
status: ready
readingMinutes: 15
tags:
  - EngineCore
  - Lifecycle
  - Overview
codeRefs:
  - vllm/v1/engine/core.py:94
  - vllm/v1/engine/core.py:439
  - vllm/v1/engine/core.py:464
heroText: 一次推理请求从 HTTP 到 token 输出的完整路径：[跨进程通信](term:跨进程、跨线程或跨 GPU 传递请求、张量或控制消息。)、核心循环、调度与执行的协作方式。
---

## 心智模型

vLLM V1 是**多进程 + 多线程**架构。以最常见的 MultiprocExecutor（TP/PP 场景）为例：1 个 API 进程 + 1 个 EngineCore 进程 + N 个 Worker 进程。EngineCore 进程内部有 3 个线程，通过 Python Queue 协作。

:::diagram arch-html
```html
<div class="arch-diagram">
  <div class="arch-proc arch-proc-api">
    <div class="arch-proc-title">API 服务进程 <span class="arch-proc-sub">(asyncio 事件循环)</span></div>
    <div class="arch-box arch-module" data-ref="vllm/entrypoints/openai/api_server.py">HTTP / OpenAI API</div>
    <div class="arch-box arch-highlight" data-ref="vllm/v1/engine/async_llm.py">AsyncLLM</div>
    <div class="arch-row">
      <div class="arch-box arch-module" data-ref="vllm/v1/engine/input_processor.py">InputProcessor</div>
      <div class="arch-box arch-module" data-ref="vllm/v1/engine/output_processor.py">OutputProcessor</div>
    </div>
    <div class="arch-coroutine">
      <span class="arch-coroutine-tag">asyncio Task</span>
      <span class="arch-coroutine-name">output_handler</span>
      <span class="muted">→ OutputProcessor → yield</span>
    </div>
  </div>
  <div class="arch-channel">
    <div class="arch-arrow">→</div>
    <div class="arch-label">ZMQ ROUTER<br>EngineCoreRequest</div>
    <div class="arch-arrow">←</div>
    <div class="arch-label">ZMQ PUSH<br>EngineCoreOutputs</div>
  </div>
  <div class="arch-proc arch-proc-core">
    <div class="arch-proc-title">EngineCore 进程 <span class="arch-proc-sub">(3 线程 + busy loop)</span></div>
    <div class="arch-thread-row">
      <div class="arch-thread arch-thread-side">
        <div class="arch-thread-tag">输入线程</div>
        <div class="arch-box arch-module" data-ref="vllm/v1/engine/core.py">ZMQ 接收<br>preprocess</div>
      </div>
      <div class="arch-thread arch-thread-main">
        <div class="arch-thread-tag">核心线程 (busy loop)</div>
        <div class="arch-box arch-highlight" data-ref="vllm/v1/engine/core.py:439">EngineCore.step()</div>
        <div class="arch-row">
          <div class="arch-box arch-module" data-ref="vllm/v1/core/sched/scheduler.py:334">Scheduler</div>
          <div class="arch-box arch-module" data-ref="vllm/v1/executor/abstract.py">Executor</div>
        </div>
        <div class="arch-flow-label">schedule → execute → update</div>
      </div>
      <div class="arch-thread arch-thread-side">
        <div class="arch-thread-tag">输出线程</div>
        <div class="arch-box arch-module" data-ref="vllm/v1/engine/core.py">ZMQ 发送<br>serialize</div>
      </div>
    </div>
  </div>
  <div class="arch-channel">
    <div class="arch-arrow">→</div>
    <div class="arch-label">SchedulerOutput<br>(broadcast)</div>
    <div class="arch-arrow">←</div>
    <div class="arch-label">ModelRunnerOutput<br>(gather)</div>
  </div>
  <div class="arch-proc arch-proc-worker">
    <div class="arch-proc-title">Worker 进程 <span class="arch-proc-sub">(×N, TP/PP)</span></div>
    <div class="arch-box arch-module" data-ref="vllm/v1/worker/gpu_worker.py">Worker</div>
    <div class="arch-box arch-highlight" data-ref="vllm/v1/worker/gpu_model_runner.py">ModelRunner</div>
    <div class="arch-box arch-module" data-ref="vllm/model_executor/models/llama.py">Model</div>
    <div class="arch-flow-label">forward → sample</div>
  </div>
</div>
```
:::

:::diagram-desc arch-html
vLLM V1 采用多进程 + 多线程架构，以 MultiprocExecutor（TP/PP 场景）为例包含三个进程：

**API 服务进程**（asyncio 事件循环）：运行 HTTP/OpenAI API 服务，通过 AsyncLLM 处理请求。InputProcessor 负责输入处理（分词、校验、多模态），OutputProcessor 负责输出处理（反分词、流式输出）。内部有 asyncio Task（output_handler）持续从 EngineCore 读取结果。

**EngineCore 进程**（3 线程 + busy loop）：核心推理引擎。包含三个线程：输入线程（从 ZMQ 接收请求并预处理）、核心线程（busy loop 执行 EngineCore.step()，即 schedule → execute → update 循环）、输出线程（将结果序列化后通过 ZMQ 发送）。核心线程中的 Scheduler 负责调度，Executor 负责模型执行。

**Worker 进程**（×N，TP/PP）：每个 Worker 包含一个 ModelRunner 和 Model。接收 SchedulerOutput，执行 GPU forward pass 和采样，返回 ModelRunnerOutput。

进程间通信：API 与 EngineCore 之间通过 ZMQ ROUTER/DEALER 发送 EngineCoreRequest，通过 ZMQ PUSH/PULL 返回 EngineCoreOutputs。EngineCore 与 Worker 之间通过共享内存环形缓冲区（ShmRingBuffer）广播 SchedulerOutput 和收集 ModelRunnerOutput。
:::

## 请求生命周期：10 步走完一次推理

以下是一个 `generate()` 请求从 HTTP 到输出 token 的完整路径。每一步标注了源码位置。

:::steps id=lifecycle-player
### 1. HTTP 请求到达
OpenAI 兼容 API 收到请求，调用 `AsyncLLM.generate()`。
`vllm/v1/engine/async_llm.py`

### 2. 输入处理
`InputProcessor` 将原始 prompt 分词、校验、处理多模态特征，组装为 `EngineCoreRequest`（msgspec Struct）。
`vllm/v1/engine/input_processor.py`

### 3. 跨进程发送
`AsyncMPClient` 将 `EngineCoreRequest` 序列化后通过 ZMQ ROUTER socket 发送到 EngineCore 进程。
`vllm/v1/engine/core_client.py`

### 4. EngineCore 接收并预处理
EngineCore 的输入线程从 ZMQ 接收、反序列化，`preprocess_add_request()` 转为内部 `Request` 对象（含结构化输出 grammar 初始化），推入 `input_queue`。
`vllm/v1/engine/core.py`

### 5. 进入调度队列
Busy loop 的 `_process_input_queue()` 取出请求，调用 `Scheduler.add_request()`，请求状态变为 `WAITING`，进入等待队列。
`vllm/v1/core/sched/scheduler.py`

### 6. 调度（schedule）
核心循环调用 `Scheduler.schedule()`。先为 RUNNING 请求分配 token budget 和 KV blocks，再从 WAITING 队列中接纳新请求。KV 不足时抢占低优先级请求。产出 `SchedulerOutput`。
`vllm/v1/core/sched/scheduler.py`

### 7. 模型执行（execute）
`Executor.execute_model(SchedulerOutput)` 将调度结果分发到 Worker 进程。`ModelRunner` 准备输入张量、执行 forward pass、返回 `ModelRunnerOutput`（含采样结果）。
`vllm/v1/executor/abstract.py`

### 8. 输出处理（update_from_output）
`Scheduler.update_from_output()` 处理每个请求的采样结果：追加 token、检查停止条件（max_tokens / stop strings / 重复）、处理投机解码接受/拒绝。已完成的请求释放 KV blocks。产出 `EngineCoreOutputs`。
`vllm/v1/core/sched/scheduler.py`

### 9. 跨进程返回
EngineCore 的输出线程将 `EngineCoreOutputs` 序列化后通过 ZMQ PUSH socket 发送回 API 服务进程。
`vllm/v1/engine/core.py`

### 10. 反分词与流式输出
`OutputProcessor` 调用 `IncrementalDetokenizer` 将 token ID 转回文本，检测 stop strings，组装 `RequestOutput`。生成器 `yield` 给 API 层，以 SSE 流式返回给客户端。
`vllm/v1/engine/output_processor.py`
:::

## 核心循环：EngineCore.step()

步骤 6-8 构成 EngineCore 的**核心迭代**，每步循环执行一次。这是 vLLM 推理的"心跳"。

```python
# vllm/v1/engine/core.py
def step(self):
    if not self.scheduler.has_requests():
        return {}, False

    scheduler_output = self.scheduler.schedule()        # 调度
    future = self.model_executor.execute_model(          # 执行
        scheduler_output, non_block=True)
    grammar_output = self.scheduler.get_grammar_bitmask( # 结构化输出
        scheduler_output)
    model_output = future.result()                       # 取结果
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)

    engine_core_outputs = self.scheduler.update_from_output(  # 更新
        scheduler_output, model_output)

    return engine_core_outputs, \
           scheduler_output.total_num_scheduled_tokens > 0
```

Busy loop 不断调用 `_process_input_queue()` → `_process_engine_step()`，形成"接收请求 → 调度执行 → 输出结果"的连续循环。
