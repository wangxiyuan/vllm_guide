---
id: kv-connector
title: "KV Connector"
category: system
level: advanced
status: draft
readingMinutes: 14
tags:
  - KV Connector
  - KV Transfer
  - Disaggregated
  - RDMA
  - KV Offload
codeRefs:
  - vllm/v1/worker/gpu/kv_connector.py
  - vllm/distributed/kv_transfer/kv_connector/v1/base.py
heroText: "跨进程/跨节点传输 KV Cache 的统一接口：[KV Connector](term:vLLM 定义的标准接口，支持多种底层实现（NIXL/Mooncake/LMCache/P2P NCCL 等）在 EngineCore 间传输 KV blocks。) 支持 15 种实现，从 P2P NCCL 到 RDMA（Mooncake）到分布式缓存（LMCache）。"
---

## 心智模型

:::diagram kvconn-mental-model-html
```html
<div class="arch-diagram">
<div class="arch-row">
  <div class="arch-box">EngineCore A</div>
  <div class="arch-arrow">→</div>
  <div class="arch-proc">
    <div class="arch-proc-title">KV Connector</div>
    <div class="arch-proc-sub">统一传输接口</div>
  </div>
  <div class="arch-arrow">↔</div>
  <div class="arch-highlight">
    <div class="arch-module">NCCL</div>
    <div class="arch-module">RDMA</div>
    <div class="arch-module">LMCache</div>
  </div>
  <div class="arch-arrow">→</div>
  <div class="arch-box">EngineCore B</div>
</div>
</div>
```
:::

:::diagram-desc kvconn-mental-model-html
KV Connector 是 EngineCore 之间传输 KV Cache 的统一接口。EngineCore A 通过 KV Connector 的统一 API 发起传输，底层可以是 NCCL（P2P 直接通信）、RDMA（Mooncake 远程直接内存访问）、LMCache（分布式缓存）等不同实现。EngineCore B 接收 KV blocks 并注入本地 KV cache。核心价值：解耦传输逻辑与调度/执行逻辑，多种传输后端无缝切换。
:::

KV Connector 的核心价值：**解耦 KV 传输逻辑与调度/执行逻辑**，支持多种传输后端无缝切换。

## KVConnectorFactory 接口

[KVConnectorFactory](term:使用延迟加载注册表模式的工厂类，根据名称创建对应的 KV Connector 实例。) 使用延迟加载注册表模式：

```python
_registry: dict[str, Callable[[], type[KVConnectorBase]]] = {}

@classmethod
def create_connector(cls, config, role, kv_cache_config):
    connector_cls = cls.get_connector_class(kv_transfer_config)
    return connector_cls(config, role, kv_cache_config)
```

关键设计：**显式分为两种角色**：
- `KVConnectorRole.SCHEDULER`：调度器侧，决定缓存命中和传输计划
- `KVConnectorRole.WORKER`：Worker 侧，执行实际的 KV 加载和保存

### 基类抽象方法

**Worker 侧：**

| 方法 | 说明 |
|------|------|
| `start_load_kv()` | 开始从 connector 加载 KV 到 paged buffer |
| `wait_for_layer_load()` | 阻塞等待特定层的 KV 加载完成 |
| `save_kv_layer()` | 开始保存一层 KV cache 到 connector |
| `wait_for_save()` | 阻塞等待所有保存操作完成 |

**Scheduler 侧：**

| 方法 | 说明 |
|------|------|
| `get_num_new_matched_tokens()` | 返回 (命中 token 数, 是否异步加载) |
| `update_state_after_alloc()` | 块分配后更新 connector 状态 |
| `build_connector_meta()` | 构建本步的不透明元数据 |

`vllm/distributed/kv_transfer/kv_connector/v1/base.py`

## 主要 Connector 实现

15 种已注册的实现，按传输方式分类：

| Connector | 传输方式 | 特点 |
|-----------|---------|------|
| **P2pNcclConnector** | NCCL P2P | 显式逐层 send/recv，同步传输 |
| **NixlConnector** | NVIDIA NIXL (RDMA/DMA) | 批量传输，支持 HMA，强制 HND 布局 |
| **MooncakeConnector** | Mooncake RDMA | P→D 推模式，支持异构 TP，ZMQ 侧通道 |
| **LMCacheConnector** | LMCache 分布式缓存 | 支持逐层操作，PIECEWISE CG 要求 |
| **OffloadingConnector** | CPU/磁盘卸载 | 全跨层块，HND 布局，LRU/ARC 驱逐 |
| **SimpleCPUOffloadConnector** | CPU 卸载 | 简化版，8GB 默认容量 |

### 三种传输模式对比

| 特性 | P2P NCCL | NIXL/Mooncake | LMCache |
|------|----------|---------------|---------|
| 传输粒度 | 逐层 | 批量（全层） | 可选逐层/批量 |
| 异步支持 | 否 | 是 | 否 |
| 跨节点 | 是 | 是（RDMA） | 是（分布式缓存） |
| CG 兼容 | FULL | FULL | PIECEWISE（逐层时） |
| wait_for_layer_load | 实际等待 | No-op | 实际等待 |

## KV Connector 与 Scheduler 的交互

:::diagram kvconn-sched-flow-html
```html
<div class="cache-flow">
  <div class="cache-step" data-step="1">
    <div class="cache-step-num">1</div>
    <div class="cache-step-content">
      <div class="cache-step-title">get_num_new_matched_tokens()</div>
      <div class="cache-step-desc">调度时查询外部缓存命中，若返回 None 则延迟请求</div>
    </div>
  </div>
  <div class="cache-step" data-step="2">
    <div class="cache-step-num">2</div>
    <div class="cache-step-content">
      <div class="cache-step-title">WAITING_FOR_REMOTE_KVS</div>
      <div class="cache-step-desc">异步加载时请求进入等待状态，传输完成后重新调度</div>
    </div>
  </div>
  <div class="cache-step" data-step="3">
    <div class="cache-step-num">3</div>
    <div class="cache-step-content">
      <div class="cache-step-title">build_connector_meta()</div>
      <div class="cache-step-desc">序列化加载/保存计划，附加到 SchedulerOutput</div>
    </div>
  </div>
  <div class="cache-step" data-step="4">
    <div class="cache-step-num">4</div>
    <div class="cache-step-content">
      <div class="cache-step-title">finished_recving / finished_sending</div>
      <div class="cache-step-desc">Worker 完成传输后反馈，Scheduler 重新调度或释放 blocks</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc kvconn-sched-flow-html
KV Connector 与 Scheduler 的交互分四步：
1. 调度期间调用 get_num_new_matched_tokens() 查询外部缓存命中。若返回 None，请求被延迟（connector 需要更多时间）。
2. 异步加载时请求进入 WAITING_FOR_REMOTE_KVS 状态，等待远程 KV 传输完成。
3. 调度完成后调用 build_connector_meta() 序列化加载/保存计划，附加到 SchedulerOutput.kv_connector_metadata。
4. Worker 完成传输后返回 finished_recving（重新调度请求）和 finished_sending（释放 blocks）。
:::

## ModelRunner 侧集成

Worker 侧的 KV Connector 通过上下文管理器生命周期集成到 ModelRunner：

```python
# 前向计算前
kv_connector.bind_connector_metadata(metadata)
kv_connector.start_load_kv(forward_context)  # 开始异步加载

# 模型前向计算（KV 可能正在异步加载中）
model_output = model(input_ids, positions, ...)

# 前向计算后
kv_connector.wait_for_save()           # 等待保存完成
output = kv_connector.get_finished()   # 收集传输完成信息
kv_connector.clear_connector_metadata()
```

对于跨层统一 KV cache（`prefer_cross_layer_blocks=True`），所有层的 KV 数据分配在一块连续 buffer 中，实现高效批量传输。

`vllm/v1/worker/kv_connector_model_runner_mixin.py`

## KV Offload 子系统

KV Cache 卸载到 CPU/磁盘的三层架构：

| 层 | 组件 | 职责 |
|----|------|------|
| Connector | `OffloadingConnector` | Scheduler + Worker 侧封装 |
| Spec | `CPUOffloadingSpec` / `TieringOffloadingSpec` | 配置卸载目标和策略 |
| Manager | `OffloadingManager` + `OffloadingHandler` | LRU/ARC 追踪 + 异步传输引擎 |

### CPU 卸载流程

1. **查找**：`lookup(key)` 检查 block 是否已卸载到 CPU
2. **加载**：`prepare_load()` → 异步 CPU→GPU 传输
3. **存储**：`prepare_store()` → 异步 GPU→CPU 传输
4. **驱逐**：LRU/ARC 策略淘汰最久未用的 CPU 缓存条目

### OffloadKey 设计

```python
OffloadKey = NewType("OffloadKey", bytes)  # block_hash + group_idx
```

使用原始 bytes 避免 tuple GC 开销，通过 hash 值 + 组索引标识唯一 block。

`vllm/v1/kv_offload/base.py`

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `kv_connector` | `None` | Connector 名称（如 `"NixlConnector"`） |
| `kv_role` | `None` | 角色：`kv_producer` / `kv_consumer` / `kv_both` |
| `kv_parallel_size` | 1 | 并行实例数（P2pNccl 需 2） |
| `kv_buffer_device` | 平台设备 | 缓冲设备：`cuda` / `cpu` |
| `kv_buffer_size` | 1e9 | 缓冲区大小（字节） |
| `kv_connector_extra_config` | `{}` | Connector 特有配置 |
| `kv_load_failure_policy` | `"fail"` | 加载失败策略：`fail` 或 `recompute` |

`vllm/config/kv_transfer.py`
