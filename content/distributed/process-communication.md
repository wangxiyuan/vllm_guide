---
id: process-communication
title: 进程与通信
category: distributed
level: intermediate
status: ready
readingMinutes: 22
tags:
  - ZMQ
  - Shared Memory
  - NCCL
  - TP
  - SP
  - CP
codeRefs:
  - vllm/v1/engine/core_client.py
  - vllm/distributed/device_communicators/shm_broadcast.py
  - vllm/v1/engine/tensor_ipc.py
  - vllm/v1/engine/core.py
  - vllm/v1/executor/multiproc_executor.py
  - vllm/distributed/parallel_state.py
heroText: ZMQ 帧协议、共享内存环形缓冲区、线程 Queue 协作、[NCCL](term:GPU 间集合通信库，vLLM 的 TP/SP/CP 等并行策略依赖它传递张量。) 集合通信、TP/SP/CP 卡间通信模式——vLLM V1 各层通信机制的实现细节。
---

## 心智模型

vLLM V1 的通信可以按**跨进程**、**进程内线程间**、**跨 GPU** 三个层级理解。每一层使用不同的通信原语，各有其延迟特性和数据格式。跨 GPU 通信按并行策略进一步分为 TP/SP/CP/PP/DP 五种模式。

:::diagram comm-layers-html
```html
<div class="comm-layers">
  <div class="comm-layer">
    <div class="comm-layer-title">跨进程</div>
    <div class="comm-layer-items">
      <div class="comm-item" data-ref="vllm/v1/engine/core_client.py">ZMQ ROUTER↔DEALER / PUSH→PULL (API ↔ EngineCore)</div>
      <div class="comm-item" data-ref="vllm/distributed/device_communicators/shm_broadcast.py">ShmRingBuffer + SpinCondition (EngineCore ↔ Worker)</div>
      <div class="comm-item" data-ref="vllm/v1/engine/tensor_ipc.py">torch.multiprocessing.Queue (多模态张量 IPC)</div>
    </div>
    <div class="comm-layer-note">延迟：μs 级（共享内存）/ 亚 ms 级（ZMQ）</div>
  </div>
  <div class="comm-layer">
    <div class="comm-layer-title">进程内线程间</div>
    <div class="comm-layer-items">
      <div class="comm-item" data-ref="vllm/v1/engine/core.py">queue.Queue (input_queue / output_queue / aborts_queue)</div>
      <div class="comm-item" data-ref="vllm/v1/engine/core_client.py">asyncio.Queue + asyncio.Event (API 进程协程协作)</div>
      <div class="comm-item" data-ref="vllm/v1/executor/multiproc_executor.py">queue.Queue (Worker async_output_queue)</div>
    </div>
    <div class="comm-layer-note">延迟：μs 级（GIL 保护，无锁竞争）</div>
  </div>
  <div class="comm-layer">
    <div class="comm-layer-title">跨 GPU — 集合通信</div>
    <div class="comm-layer-items">
      <div class="comm-item" data-ref="vllm/distributed/parallel_state.py">TP: all_reduce / all_gather (线性层 forward)</div>
      <div class="comm-item" data-ref="vllm/compilation/passes/fusion/sequence_parallelism.py">SP: reduce_scatter + all_gather (替代 TP all_reduce)</div>
      <div class="comm-item" data-ref="vllm/v1/attention/ops/dcp_alltoall.py">DCP: all_gather + LSE 校正 + reduce_scatter / all_to_all</div>
      <div class="comm-item" data-ref="vllm/model_executor/layers/fused_moe/runner/moe_runner.py">PCP: all_gather + reduce_scatter (MoE prefill)</div>
      <div class="comm-item" data-ref="vllm/distributed/parallel_state.py">PP: send/recv (阶段间中间张量)</div>
    </div>
    <div class="comm-layer-note">延迟：μs 级（NVLink）/ ms 级（PCIe + 网络）</div>
  </div>
</div>
```
:::

:::diagram-desc comm-layers-html
vLLM V1 通信按三个层级组织：

**跨进程**：API 与 EngineCore 之间通过 ZMQ ROUTER↔DEALER / PUSH→PULL 通信；EngineCore 与 Worker 之间通过 ShmRingBuffer + SpinCondition 共享内存通信；多模态张量通过 torch.multiprocessing.Queue IPC 传输。延迟为 μs 级（共享内存）到亚 ms 级（ZMQ）。

**进程内线程间**：EngineCore 内部使用 queue.Queue（input_queue / output_queue / aborts_queue）；API 进程使用 asyncio.Queue + asyncio.Event 协程协作；Worker 使用 queue.Queue（async_output_queue）。延迟为 μs 级，GIL 保护无锁竞争。

**跨 GPU 集合通信**：TP 使用 all_reduce / all_gather；SP 使用 reduce_scatter + all_gather 替代 TP all_reduce；DCP 使用 all_gather + LSE 校正 + reduce_scatter / all_to_all；PCP 使用 all_gather + reduce_scatter（MoE prefill）；PP 使用 send/recv。延迟为 μs 级（NVLink）到 ms 级（PCIe + 网络）。
:::

## 通信全景图

下图展示 vLLM V1 各进程/线程之间的通信原语与数据流向。每条连线标注了通信方式、关键数据结构和源码位置。

:::diagram comm-panorama-html
```html
<div class="comm-panorama">
  <div class="comm-proc comm-proc-api">
    <div class="comm-proc-title">API 服务进程</div>
    <div class="comm-proc-body">
      <div class="comm-node" data-ref="vllm/v1/engine/async_llm.py">AsyncLLM</div>
      <div class="comm-node" data-ref="vllm/v1/engine/core_client.py">AsyncMPClient</div>
      <div class="comm-node-sub">outputs_queue (asyncio.Queue)</div>
      <div class="comm-node-sub">RequestOutputCollector (asyncio.Event)</div>
    </div>
  </div>

  <div class="comm-channel-group">
    <div class="comm-channel-item">
      <div class="comm-arrow">EngineCoreRequest ▶</div>
      <div class="comm-label">ZMQ ROUTER↔DEALER + msgpack</div>
    </div>
    <div class="comm-channel-item">
      <div class="comm-arrow">◀ EngineCoreOutputs</div>
      <div class="comm-label">ZMQ PUSH→PULL + msgpack 零拷贝</div>
    </div>
    <div class="comm-channel-item">
      <div class="comm-arrow">◀ GPU 张量</div>
      <div class="comm-label">mp.Queue (torch_shm, 仅多模态)</div>
    </div>
  </div>

  <div class="comm-proc comm-proc-core">
    <div class="comm-proc-title">EngineCore 进程</div>
    <div class="comm-proc-body">
      <div class="comm-thread-row">
        <div class="comm-thread">
          <div class="comm-thread-tag">input_thread</div>
          <div class="comm-node-sub">ZMQ DEALER → input_queue</div>
        </div>
        <div class="comm-thread comm-thread-main">
          <div class="comm-thread-tag">主线程 (busy loop)</div>
          <div class="comm-node" data-ref="vllm/v1/engine/core.py">EngineCore.step()</div>
          <div class="comm-node-sub">input_queue → output_queue</div>
        </div>
        <div class="comm-thread">
          <div class="comm-thread-tag">output_thread</div>
          <div class="comm-node-sub">output_queue → ZMQ PUSH</div>
        </div>
      </div>
    </div>
  </div>

  <div class="comm-channel-group">
    <div class="comm-channel-item">
      <div class="comm-arrow">SchedulerOutput ▶</div>
      <div class="comm-label">rpc_broadcast_mq (ShmRingBuffer)</div>
    </div>
    <div class="comm-channel-item">
      <div class="comm-arrow">◀ ModelRunnerOutput</div>
      <div class="comm-label">worker_response_mq (ShmRingBuffer)</div>
    </div>
  </div>

  <div class="comm-proc comm-proc-worker">
    <div class="comm-proc-title">Worker 进程 × N</div>
    <div class="comm-proc-body">
      <div class="comm-node" data-ref="vllm/v1/worker/gpu_model_runner.py">ModelRunner</div>
      <div class="comm-node-sub">GPU 0 / GPU 1 / ...</div>
      <div class="comm-node-sub">TP: all_reduce / all_gather</div>
      <div class="comm-node-sub">PP: send / recv</div>
      <div class="comm-node-sub">CP: all_gather + LSE 校正</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc comm-panorama-html
vLLM V1 通信全景图展示三个进程之间的通信：

**API 服务进程**：包含 AsyncLLM 和 AsyncMPClient，内部有 outputs_queue（asyncio.Queue）和 RequestOutputCollector（asyncio.Event）。

**API ↔ EngineCore 通道**：EngineCoreRequest 通过 ZMQ ROUTER↔DEALER + msgpack 发送；EngineCoreOutputs 通过 ZMQ PUSH→PULL + msgpack 零拷贝返回；GPU 张量通过 mp.Queue（torch_shm，仅多模态）回传。

**EngineCore 进程**：包含三个线程——input_thread（ZMQ DEALER → input_queue）、主线程 busy loop（EngineCore.step()，input_queue → output_queue）、output_thread（output_queue → ZMQ PUSH）。

**EngineCore ↔ Worker 通道**：SchedulerOutput 通过 rpc_broadcast_mq（ShmRingBuffer）广播；ModelRunnerOutput 通过 worker_response_mq（ShmRingBuffer）收集。

**Worker 进程 × N**：包含 ModelRunner，分布在 GPU 0/1/... 上，使用 TP all_reduce/all_gather、PP send/recv、CP all_gather + LSE 校正等集合通信。
:::

## 进程间通信：API ↔ EngineCore

API 进程与 EngineCore 进程之间通过两对 ZMQ socket 通信，分别承载输入和输出。所有消息使用 **msgpack** 序列化（`msgspec` 实现），大张量通过零拷贝多帧发送。

### Socket 拓扑

| 方向 | Socket 类型 | 绑定方 | 用途 |
|------|------------|--------|------|
| Client → EngineCore | ROUTER ↔ DEALER | Client bind | 请求/中止/工具方法 |
| EngineCore → Client | PUSH → PULL | Client bind | 推理输出/调度统计 |
| Coordinator → Frontend | XPUB → XSUB | Coordinator bind | DP 负载统计发布 |
| Coordinator → EngineCore | XPUB → XSUB | Coordinator bind | DP wave 控制消息 |

ROUTER socket 允许 Client 端按 `EngineIdentity`（2 字节 DP rank little-endian 编码）路由到不同 DP rank 的 EngineCore。DEALER 端自动重连，无需手动管理连接状态。XPUB/XSUB 的发布-订阅模式用于 DP 场景的 wave 协调，单 DP 时不启用。

### 输入帧格式（ROUTER → DEALER）

输入消息为多帧 ZMQ 消息，结构如下：

```
# Frame 0: EngineIdentity (2 bytes, DP rank little-endian)
# Frame 1: EngineCoreRequestType (1 byte hex)
#   ADD              = b"\x00"   新请求
#   ABORT            = b"\x01"   中止请求
#   START_DP_WAVE    = b"\x02"   DP wave 开始
#   UTILITY          = b"\x03"   工具方法调用
#   EXECUTOR_FAILED  = b"\x04"   Executor 崩溃信号
#   WAKEUP           = b"\x05"   唤醒空闲引擎
# Frame 2+: Msgpack 序列化的请求数据
#   大张量提取底层 buffer，通过 send_multipart 多帧零拷贝发送
```
`vllm/v1/engine/__init__.py`

### 输出帧格式（PUSH → PULL）

输出消息使用 `MsgpackEncoder.encode_into()` 编码为多帧：

```
# Encoder 将 EngineCoreOutputs 编码为 msgpack 帧
# 小张量（< VLLM_MSGPACK_ZERO_COPY_THRESHOLD，默认 256B）内联序列化
# 大张量提取底层 buffer 放入 aux_buffers
# CUDA 张量先 .cpu() 再提取 buffer
# 最终通过 send_multipart(buffers, copy=False, track=True) 发送
```
Client 端的 `process_outputs_socket` 协程通过 `recv_multipart(copy=False)` 接收，`MsgpackDecoder` 反序列化后推入 `asyncio.Queue`。

`vllm/v1/serial_utils.py`

### OOB 张量 IPC（多模态场景）

当启用 `mm_tensor_ipc == "torch_shm"` 时，多模态张量绕过 msgpack 序列化，通过 `torch.multiprocessing.Queue` 共享内存传输：

:::steps id=oob-tensor-ipc
### 1. 发送端
调用 `tensor.share_memory_()` 将张量移入共享内存，封装为 `TensorIpcData(sender_id, message_id, tensor_id, tensor)`，通过 `mp.Queue.put()` 发送。msgpack 中仅存放轻量 handle（`sender_id, message_id, tensor_id`）。
`vllm/v1/engine/tensor_ipc.py`

### 2. 接收端
采用 drain-and-buffer 模式：从 `mp.Queue` 中取出所有待处理张量并缓存到 `sender.tensors[message_id]` 字典。当需要特定张量时，按 handle 查找。超时 10 秒。
`vllm/v1/engine/tensor_ipc.py`
:::

## 进程间通信：EngineCore ↔ Worker

EngineCore 进程与 N 个 Worker 进程之间通过**共享内存环形缓冲区 + ZMQ 混合广播**通信（`MessageQueue` 类），避免大张量走 ZMQ 的序列化开销。

### MessageQueue 双模式

| 数据大小 | 传输路径 | 通知机制 |
|----------|---------|---------|
| ≤ 24 MiB | ShmRingBuffer (共享内存) | SpinCondition (自旋等待) |
| > 24 MiB | ZMQ XPUB/SUB | ZMQ 事件通知 |

本地 Worker（同节点）从共享内存直接读取，远程 Worker（跨节点）通过 ZMQ SUB socket 接收。

### 环形缓冲区协议

```
# Metadata buffer layout:
#   [0]:       written_flag (0=未写, 1=已写)
#   [1..n_reader]: read_flags (0=未读, 1=已读)
#
# Data buffer layout:
#   [0]:       overflow 标志 (0=在共享内存中, 1=走 ZMQ)
#   [1:3]:     buffer 数量 (2 bytes big-endian)
#   Per-buffer:
#     [4 bytes size][size bytes data]...
```
Writer 写入后设置 `written_flag` 并通知所有 Reader。Reader 读取后设置对应的 `read_flag`。当所有 Reader 均已读取，Writer 才能覆盖该槽位。

`vllm/distributed/device_communicators/shm_broadcast.py`

### collective_rpc 流程

EngineCore 通过 `collective_rpc` 向所有 Worker 广播方法调用，并收集结果：

:::steps id=collective-rpc
### 1. 广播调用
Executor 将 `(method_name, args, kwargs, output_rank)` 通过 `rpc_broadcast_mq`（MessageQueue）广播到所有 Worker。
`vllm/v1/executor/multiproc_executor.py`

### 2. Worker 执行
每个 Worker 的 `worker_busy_loop()` 从 `rpc_broadcast_mq` 中取出消息，调用 `getattr(worker, method)(*args, **kwargs)`。
`vllm/v1/executor/multiproc_executor.py`

### 3. 返回结果
Worker 通过 `worker_response_mq`（另一条 MessageQueue）将执行结果发回 Executor。Executor 收集所有 Worker 的结果，返回 `FutureWrapper`（非阻塞）或直接结果（阻塞）。
`vllm/v1/executor/multiproc_executor.py`
:::

### 两条 MessageQueue 的方向

| 队列 | 方向 | 内容 |
|------|------|------|
| rpc_broadcast_mq | Executor → Worker | SchedulerOutput, 方法调用 |
| worker_response_mq | Worker → Executor | ModelRunnerOutput, 工具方法结果 |

## 线程间通信

### EngineCore 进程内部

EngineCore 进程有 3 个线程：input_thread、主线程（busy loop）、output_thread。它们通过 Python `queue.Queue` 协作，GIL 保证线程安全。

:::steps id=engine-core-queues
### 1. input_queue
`Queue[tuple[EngineCoreRequestType, Any]]`。input_thread 从 ZMQ DEALER 接收消息、反序列化后推入此队列。主线程的 `_process_input_queue()` 从中取出处理。

元组格式：`(ADD, (Request, request_wave))` / `(ABORT, list[str])` / `(UTILITY, (client_idx, call_id, method, args))`。
`vllm/v1/engine/core.py`

### 2. aborts_queue
独立的 `queue.Queue`，与 `input_queue` 并行。abort 请求同时推入两个队列——`input_queue` 保证顺序性，`aborts_queue` 保证主线程在 `step()` 执行间隙也能立即处理中止请求。当 `input_queue` 为空时，主线程会顺便清空 `aborts_queue`。
`vllm/v1/engine/core.py`

### 3. output_queue
`Queue[tuple[int, EngineCoreOutputs] | bytes]`。主线程的 `_process_engine_step()` 将调度+执行+更新的结果推入此队列。output_thread 从中取出，通过 ZMQ PUSH socket 发回 API 进程。特殊值 `ENGINE_CORE_DEAD`（bytes）表示引擎崩溃。
`vllm/v1/engine/core.py`

### 4. batch_queue（Pipeline Parallel）
`deque[tuple[Future, SchedulerOutput, Future]]`，非 Queue。PP 需要多 batch 并发执行以消除 pipeline bubble。`batch_queue_size` 由 `model_executor.max_concurrent_batches` 决定。
`vllm/v1/engine/core.py`
:::

### API 进程内部（asyncio 协作）

`AsyncMPClient` 启动后台协程 `process_outputs_socket`，通过 `asyncio.Queue` 与 `AsyncLLM.output_handler` 协程协作：

:::steps id=api-async-queues
### 1. outputs_queue
`asyncio.Queue[EngineCoreOutputs | Exception]`。`process_outputs_socket` 协程从 ZMQ PULL socket 接收帧、反序列化后推入此队列。utility_output 走单独处理路径（设置 `asyncio.Event` 唤醒等待方），其余推入队列。
`vllm/v1/engine/core_client.py`

### 2. RequestOutputCollector
每个请求对应一个 `RequestOutputCollector`，内部使用 `asyncio.Event` 实现生产者-消费者模式。`put()` 非阻塞写入并设置 `ready` 事件；`get()` 等待事件后取出数据。`get_nowait()` 非阻塞取出，避免不必要的任务切换。
`vllm/v1/engine/output_processor.py`

### 3. output_handler 分块处理
`AsyncLLM.output_handler` 从 `outputs_queue` 取出输出后，按 `chunk_size` 分块处理（避免长时间阻塞事件循环）。每块处理后 `await asyncio.sleep(0)` 让出控制权给其他协程（如新的 HTTP 请求）。
`vllm/v1/engine/async_llm.py`
:::

### Worker 进程内部（异步输出线程）

启用 async scheduling 时，Worker 启动额外的 `async_output_copy_thread`：

:::steps id=worker-async-output
### 1. async_output_queue
`queue.Queue`。Worker 主循环执行 `execute_model` 后产出 `AsyncModelRunnerOutput`，`handle_output` 将其放入此队列。
`vllm/v1/executor/multiproc_executor.py`

### 2. 异步拷贝与发送
`async_output_copy_thread` 从队列取出 `AsyncModelRunnerOutput`，调用 `.get_output()` 阻塞等待 GPU 计算完成（GPU→CPU 拷贝），再通过 `worker_response_mq` 发回 Executor。这样主循环无需等待 GPU 完成，可立即启动下一次 `execute_model`。
`vllm/v1/executor/multiproc_executor.py`
:::

## 卡间通信：NCCL 集合操作与后端

GPU 间通信由 `GroupCoordinator` 管理，它维护两个 ProcessGroup：`device_group`（NCCL 后端，GPU 张量）和 `cpu_group`（Gloo 后端，CPU 对象广播/屏障）。所有集合操作封装在 `GroupCoordinator` 方法中。

### 通信操作一览

| 操作 | 典型用途 | 后端 |
|------|---------|------|
| all_reduce | RowParallelLinear 后部分和归约 | NCCL / Custom All-Reduce |
| all_gather | ColumnParallelLinear gather_output、SP/CP 序列维度拼接 | NCCL |
| reduce_scatter | SP 激活值序列切分、PCP MoE 结果分发 | NCCL |
| all_to_all | DCP A2A 模式（打包 output+LSE 单次通信）、EP 专家路由 | NCCL |
| all_gatherv / reduce_scatterv | 变长数据的集合操作 | NCCL |
| broadcast | 广播张量字典（配置同步等） | NCCL / Gloo |
| send / recv | Pipeline Parallel 阶段间中间张量传递 | NCCL |

`vllm/distributed/parallel_state.py`

### 通信后端选择

vLLM 支持多种 GPU 通信后端，通过 `CudaCommunicator` 统一调度，优先级从高到低：

| 优先级 | 后端 | 实现类 | 特点 |
|--------|------|--------|------|
| 1 | NCCL + Symmetric Memory | CudaCommunicator | 对称内存优化，融合 all-reduce 与计算 |
| 2 | Custom All-Reduce | CustomAllreduce | 对称内存 + IPC 信号量，绕过 NCCL，可入 CUDA Graph |
| 3 | FlashInfer All-Reduce | flashinfer_all_reduce | FlashInfer 实现的 all-reduce |
| 4 | PyTorch Symmetric Memory | symm_mem | PyTorch 对称内存通信 |
| 5 | PyNCCL | PyNcclCommunicator | 默认回退后端，直接调用 NCCL C API |
| — | Gloo | cpu_group | CPU 端通信（对象广播、屏障同步） |

`vllm/distributed/device_communicators/cuda_communicator.py`

### Custom All-Reduce 原理

Custom All-Reduce 完全绕过 NCCL，使用 **对称内存 + IPC 信号量** 实现，可在 CUDA Graph 中调用：

:::steps id=custom-allreduce
### 1. 对称内存注册
每个 GPU rank 在初始化时将通信缓冲区注册为对称内存（所有 rank 的 buffer 在各自 GPU 上的地址偏移相同），通过 `torch.cuda.ipc` 打开其他 rank 的共享内存句柄。
`vllm/distributed/device_communicators/custom_all_reduce.py`

### 2. 信号量同步
使用 CUDA IPC 信号量做跨进程同步，无需 CPU 介入。GPU kernel 直接在对称内存上完成归约。
`vllm/distributed/device_communicators/custom_all_reduce.py`

### 3. 阈值回退
小消息走 Custom All-Reduce（低延迟），大消息回退 NCCL（高带宽）。阈值可通过 `VLLM_CUSTOM_ALL_REDUCE_THRESHOLD` 调整。
`vllm/distributed/device_communicators/custom_all_reduce.py`
:::

### all-reduce 调用链（从模型层到 NCCL）

```
RowParallelLinear.forward()
  → tensor_model_parallel_all_reduce()       [vllm/distributed/communication_op.py]
    → get_tp_group().all_reduce()            [GroupCoordinator]
      → torch.ops.vllm.all_reduce()          [custom op, 支持 CUDA Graph]
        → CudaCommunicator.all_reduce()
          → [NCCL_SYMM_MEM / CUSTOM / FLASHINFER / SYMM_MEM / PYNCCL]
            → PyNcclCommunicator.all_reduce()
              → ncclAllReduce()              [NCCL C API]
```
`vllm/distributed/device_communicators/cuda_communicator.py`

## 卡间通信：Tensor Parallel

TP 将模型权重沿 output/input 维度切分到多个 GPU，每个 rank 持有部分权重，forward 中通过集合通信协作。vLLM 通过四种线性层封装不同的切分策略。

### 线性层切分方式

| 线性层类型 | 切分维度 | forward 通信 | 典型用途 |
|-----------|---------|-------------|---------|
| ColumnParallelLinear | 权重沿 output 维度切分 | gather_output=True 时 all_gather | gate_proj / up_proj |
| RowParallelLinear | 权重沿 input 维度切分 | reduce_results=True 时 all_reduce | down_proj / o_proj |
| MergedColumnParallelLinear | gate+up 输出维度拼接后切分 | gather_output=True 时 all_gather（通常=False） | gate_proj + up_proj 融合 |
| QKVParallelLinear | Q 按 num_heads 切分，K/V 按 num_kv_heads 切分 | 无（gather_output=False） | Q/K/V 投影融合 |

`vllm/model_executor/layers/linear.py`

### 典型 Transformer 层的 TP 通信模式

```
输入 (完整)
  │
  ├─ QKVParallelLinear (本地 GEMM, 无通信)
  │    每个 rank 持有 Q/K/V 的部分 head
  ├─ Attention (本地 head, 无 TP 通信)
  ├─ o_proj = RowParallelLinear (本地 GEMM)
  │    └─ all_reduce(o_proj_output)     ← 第 1 次 TP 通信
  ├─ 残差连接
  ├─ MergedColumnParallelLinear (本地 GEMM, gather_output=False)
  │    每个 rank 持有 gate/up 的部分输出
  ├─ 激活函数 (本地)
  ├─ down_proj = RowParallelLinear (本地 GEMM)
  │    └─ all_reduce(down_proj_output)  ← 第 2 次 TP 通信
  └─ 残差连接
```
每层 Transformer 有 **2 次 all-reduce**，分别发生在 o_proj 和 down_proj 之后。QKV 和 gate/up 的 ColumnParallelLinear 因为 `gather_output=False` 不触发通信——每个 rank 只需自己持有的 head 切片即可继续计算。

### GQA 的 KV 复制

当 GQA 场景下 `tp_size ≥ num_kv_heads` 时，K/V 无法继续切分，各 rank 会复制相同的 KV head。`QKVParallelLinear` 自动处理这种情况，无需额外通信。

### TP 通信与 CUDA Graph

集合操作通过 `torch.ops.vllm.all_reduce` 等 custom op 注册，使其可被 CUDA Graph 捕获。`GroupCoordinator.graph_capture()` 上下文管理器在独立 CUDA stream 上捕获 graph，同时进入 custom all-reduce 的捕获上下文，注册对称内存 buffer。

`vllm/distributed/parallel_state.py`

## 卡间通信：Sequence Parallel

SP 是 TP 的附带优化。当 TP > 1 时，SP 将 Transformer 层间的 all-reduce 分解为 **reduce_scatter + all_gather**，使激活值在序列维度上保持切分状态，为 GEMM-通信融合创造条件。vLLM 中 SP 有两种形态。

### 形态一：编译期 SP（SequenceParallelismPass）

通过 `torch.compile` 的 FX graph pattern matching，将 all-reduce + RMSNorm 模式替换为 reduce_scatter + RMSNorm + all_gather：

```
# 原始模式 (TP without SP):
Input → AllReduce → RMSNorm → Output

# SP 变换后:
Input → ReduceScatter → RMSNorm → AllGather → Output
```
编译期 SP 支持三种 pattern：

| Pattern | 原始模式 | SP 变换 |
|---------|---------|--------|
| FirstAllReduceRMSNorm | all_reduce(input) → rms_norm | reduce_scatter(input) → rms_norm → all_gather |
| MiddleAllReduceRMSNorm | all_reduce(mm_out) → fused_add_rms_norm | reduce_scatter(mm_out) → fused_add_rms_norm → all_gather |
| 量化变体 | StaticFP8 / NVFP4 的相同模式 | 对应的量化 reduce_scatter + all_gather |

`vllm/compilation/passes/fusion/sequence_parallelism.py`

### SP 在 forward 中的通信位置

```
# SP 启用后，激活值在序列维度上保持切分状态
# 只有在进入 ColumnParallelLinear 前才 all_gather 拼接
# 在 RowParallelLinear 后立即 reduce_scatter 切分

Embedding (完整)
  → [第一层] AllReduce(o_proj) → RMSNorm        ← 原始 TP
             ReduceScatter → RMSNorm → AllGather ← SP 变换

... (每层 Transformer 重复) ...

[中间层] RowParallelLinear → AllReduce → Add+RMSNorm   ← 原始 TP
         RowParallelLinear → ReduceScatter → Add+RMSNorm → AllGather ← SP 变换
```

### 形态二：MoE Sequence Parallel

在 MoE 模型中（DeepSeek-V2、Qwen3-Next 等），`sequence_parallel_chunk` 将 hidden_states 沿 token 维度（dim=0）按 `tp_rank` 切分，每个 rank 处理 `seq_len / tp_size` 个 token。在 MoE Runner 中与 EP 的 all-to-all 配合使用。

`vllm/model_executor/layers/fused_moe/layer.py`

### SP 对 PP 通信的影响

SP 启用时，residual 张量沿序列维度分散在各 TP rank 上（而非 replicated）。PP send/recv 时需通过 `all_gather_tensors` 字典标记 residual 不应使用 all-gather 优化（`all_gather_tensors["residual"] = False`），否则会错误地合并各 rank 的 residual 切片。

`vllm/v1/worker/utils.py`

## 卡间通信：Context Parallel

CP 将长序列的上下文切分到多个 GPU，扩展单请求可处理的最大序列长度。vLLM 中 CP 分为 **DCP**（Decode Context Parallel）和 **PCP**（Prefill Context Parallel），二者可组合使用。

### DCP 通信模式（Decode 场景）

DCP 复用 TP 组的 GPU，将一个 TP 组拆分为 `tp_size / dcp_size` 个 DCP 组。约束：`tp_size % dcp_size == 0`。每个 DCP rank 持有 KV cache 的不同序列段，但 query 在所有 DCP rank 上都完整（通过 all-gather 获得）。

#### FlashAttention 中的 DCP

:::steps id=dcp-flash-attn
### 1. All-Gather Query
`get_dcp_group().all_gather(query, dim=1)` 沿 head 维度 all-gather Q，使每个 DCP rank 拥有完整的 query，可以对本地 KV cache 做 attention。
`vllm/v1/attention/backends/flash_attn.py`

### 2. 本地 Attention
用完整 Q 对本地 KV cache 做 attention，产出 context_attn_out 和 LSE（log-sum-exp）。同时用本地 Q 对新 token 做 query attention。
`vllm/v1/attention/backends/flash_attn.py`

### 3. LSE 校正合并
各 DCP rank 的 attention 输出需要用 LSE 做数值校正（因为每个 rank 只看到部分 KV，softmax 分母不同）。校正后合并 context 和 query attention 输出。具体合并有两种后端。
`vllm/v1/attention/backends/flash_attn.py`
:::

#### DCP 合并后端

| 后端 | NCCL 通信次数 | 流程 | 适用场景 |
|------|-------------|------|---------|
| AG+RS | 3 次 | all_gather(LSE) + all_gather(output) → Triton LSE 校正 → reduce_scatter(corrected) | 通用 |
| A2A | 1 次 | Triton 打包 output+LSE → all_to_all → Triton 解包+LSE 加权合并 | 低延迟优先 |

A2A 后端通过 Triton kernel 将 output 和 LSE 打包到同一 buffer，用单次 all-to-all 代替 3 次集合通信，显著降低通信延迟。

`vllm/v1/attention/ops/dcp_alltoall.py`

#### MLA Attention 中的 DCP

MLA（Multi-head Latent Attention，DeepSeek 系列）的 DCP 在 prefill 路径中对每个 chunk 先 all-gather KV cache 再做 attention，decode 路径与标准 FlashAttention 类似（all-gather MQA query → LSE 校正合并）。

`vllm/model_executor/layers/attention/mla_attention.py`

### PCP 通信模式（Prefill 场景）

PCP 是独立并行维度，主要在 MoE 场景中用于加速 prefill。GPU 进程布局为 `reshape(ExternalDP, DP, PP, PCP, TP)`，PCP 增加总 GPU 数。

:::steps id=pcp-moe
### 1. MoE Dispatch 前：All-Gather
`get_pcp_group().all_gather(hidden_states, dim=0)` 沿 token 维度 all-gather，将分散在各 PCP rank 的 token 汇聚，使 MoE router 能看到完整 token 集合。
`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`

### 2. MoE Combine 后：Reduce-Scatter
`get_pcp_group().reduce_scatter(hidden_states, dim=0)` 沿 token 维度 reduce-scatter，将 MoE 计算结果重新切分到各 PCP rank。
`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
:::

### CP 与 TP/SP 的组合

| 并行维度 | 切分对象 | 通信操作 | 与 TP 的关系 |
|---------|---------|---------|-------------|
| TP | 权重（output/input 维度） | all_reduce / all_gather | — |
| SP | 激活值（序列/token 维度） | reduce_scatter + all_gather | TP 的附带优化，tp>1 时启用 |
| DCP | KV cache（上下文序列） | all_gather(Q) + LSE 校正 + reduce_scatter / all_to_all | 复用 TP GPU，tp_size % dcp_size == 0 |
| PCP | Prefill token | all_gather + reduce_scatter | 独立维度，增加 GPU 数 |

组合约束：`total_cp = pcp_world_size × dcp_world_size`，`total_cp_rank = pcp_rank × dcp_world_size + dcp_rank`。KV cache 按 `total_cp_rank` 交错存储。

## 卡间通信：Pipeline Parallel

PP 阶段间的通信使用 NCCL `send/recv`，辅以 Gloo 传输元数据。关键设计是**懒等待**和**多 batch 并发**。

### 阶段间数据传输

:::steps id=pp-send-recv
### 1. 发送端（send_tensor_dict）
先通过 `cpu_group`（Gloo）发送元数据（张量形状、dtype 等），再通过 `device_group`（NCCL）异步发送 GPU 张量。若 TP 组内有多个 PP stage，先沿 TP 切分再 all-gather 优化。
`vllm/distributed/parallel_state.py`

### 2. 接收端（recv_tensor_dict）
先通过 `cpu_group` 接收元数据，据此分配 GPU 张量缓冲区，再通过 `device_group` 异步接收。返回 `(tensor_dict, handles, postprocess)`。
`vllm/distributed/parallel_state.py`

### 3. 懒等待（AsyncIntermediateTensors）
`AsyncIntermediateTensors` 继承 `IntermediateTensors`，重写 `__getattribute__`：首次访问 `.tensors` 属性时才调用 `wait_for_comm()` 同步 NCCL 操作。这样下一阶段的计算可以与当前阶段的通信重叠。
`vllm/v1/worker/gpu_worker.py`
:::

### PP + TP 的 all-gather 优化

PP stage 间传递的 tensor 在各 TP rank 上通常是 replicated。利用这一点，每个 TP rank 只发送自己持有的切片，接收端通过 all-gather 重建完整 tensor，将通信量从 `numel × tp_size` 减少到 `numel`。SP 启用时 residual 在各 TP rank 上不同，需通过 `all_gather_tensors` 字典排除。

`vllm/distributed/parallel_state.py`

### batch_queue 消除 Pipeline Bubble

EngineCore 维护 `batch_queue: deque`（`maxlen = max_concurrent_batches`），存储 `(execute_future, SchedulerOutput, sample_future)` 三元组。多个 batch 同时在 pipeline 中流动，形成类似 software pipelining 的效果，减少 GPU 空闲时间。

`vllm/v1/engine/core.py`

## 卡间通信：KV Cache 传输

Disaggregated Prefill 场景下，Prefill 节点和 Decode 节点之间需要传输 KV Cache。`KVConnectorBase_V1` 定义了 Scheduler 侧和 Worker 侧的异步接口。

### KVConnector 双侧接口

| 侧 | 方法 | 说明 |
|----|------|------|
| Scheduler | get_num_new_matched_tokens() | 查询远程 KV cache 中已存在的 token 数 |
| Scheduler | update_state_after_alloc() | 缓冲区分配后更新状态 |
| Scheduler | request_finished() | 请求完成时决定是否异步释放 blocks |
| Worker | start_load_kv() | 开始异步加载所有 KV |
| Worker | wait_for_layer_load(i) | 阻塞直到第 i 层加载完成 |
| Worker | save_kv_layer(i) | 开始异步保存第 i 层 KV |
| Worker | wait_for_save() | 阻塞直到所有保存完成 |

`vllm/distributed/kv_transfer/kv_connector/v1/base.py`

### P→D 传输流程

:::steps id=kv-transfer-pd
### 1. Prefill 节点正常执行
请求到达 Prefill 节点，`get_num_new_matched_tokens()` 返回 0（远程无缓存），正常 prefill。
`vllm/distributed/kv_transfer/kv_connector/v1/base.py`

### 2. 异步保存 KV
请求完成后 `request_finished()` 返回 True，触发异步保存。逐层调用 `save_kv_layer(i)`，通过 NIXL/Mooncake 等传输协议发送到 Decode 节点。
`vllm/distributed/kv_transfer/kv_connector/v1/base.py`

### 3. Decode 节点异步加载
请求到达 Decode 节点，`get_num_new_matched_tokens()` 返回已缓存 token 数。调用 `start_load_kv()` 开始异步加载，`wait_for_layer_load(i)` 逐层等待。加载完成后正常 decode。
`vllm/distributed/kv_transfer/kv_connector/v1/base.py`
:::

## 卡间通信：Data Parallel 协调

多 DP rank 的 EngineCore 通过 **XPUB/XSUB 发布-订阅**模式与 **DPCoordinator** 协调，并通过 NCCL all-reduce 同步全局状态。

### Wave 协调流程

:::steps id=dp-wave-coordination
### 1. 全局状态同步
每个 EngineCore 在 `step()` 中每 32 步执行一次 `all-reduce`（通过 stateless ProcessGroup），调用 `_has_global_unfinished_reqs()` 判断全局是否还有未完成请求。
`vllm/v1/engine/core.py`

### 2. Wave 完成通知
DP rank 0 的 EngineCore 将 `wave_complete` 消息通过 `output_queue`（`client_index=-1`）发送给 DPCoordinator。
`vllm/v1/engine/coordinator.py`

### 3. Coordinator 广播
DPCoordinator 更新 wave 计数，通过 XPUB 通知所有 Frontend，再广播 `START_DP_WAVE` 消息给所有 EngineCore。所有引擎收到后设置 `engines_running=True`，开始新 wave。
`vllm/v1/engine/coordinator.py`
:::

### Stateless Process Group

DP 通信使用 `stateless_init_dp_group` 创建跨节点的 ProcessGroup，无需 `torch.distributed.init_process_group` 的全局初始化。这样 DP rank 可以独立启动和关闭。

`vllm/v1/engine/core.py`
