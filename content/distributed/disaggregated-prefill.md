---
id: disaggregated-prefill
title: Disaggregated Prefill
category: distributed
level: advanced
status: draft
readingMinutes: 14
tags:
  - Disaggregated Prefill
  - KV Transfer
codeRefs:
  - vllm/distributed/kv_transfer/kv_connector/v1/base.py
heroText: Prefill 节点与 Decode 节点分离部署，KV Cache 通过 [KV Connector](term:Prefill 和 Decode 节点间传输 KV Cache 的异步接口。) 异步传输，消除长 prefill 对 decode 吞吐的干扰。
---

## 心智模型

想象一家餐厅厨房：prefill 好比食材准备（切菜、腌制——计算密集型），decode 好比烹饪和摆盘（内存带宽密集型）。Disaggregated Prefill 把这两种工作分离到不同的厨房（GPU），把准备好的食材（KV cache）从备菜间传到烹饪间。

:::diagram pd-mental-model-html
```html
<div class="diagram-container">
  <div class="diagram">
    <div class="comm-panorama">
      <div class="comm-proc">
        <div class="comm-proc-title">Prefill 节点</div>
        <div class="comm-proc-body">
          <div class="comm-node-sub">计算密集型</div>
          <div class="comm-node">处理长 prompt</div>
          <div class="comm-node">生成 KV Cache</div>
          <div class="comm-node">异步传输 KV</div>
        </div>
      </div>
      <div class="comm-channel-group">
        <div class="comm-channel-item">
          <div class="comm-arrow">KV Cache ▶</div>
          <div class="comm-label">KV Connector (NIXL / Mooncake / ...)</div>
        </div>
      </div>
      <div class="comm-proc">
        <div class="comm-proc-title">Decode 节点</div>
        <div class="comm-proc-body">
          <div class="comm-node-sub">内存带宽密集型</div>
          <div class="comm-node">加载 KV Cache</div>
          <div class="comm-node">逐 token 生成</div>
          <div class="comm-node">低延迟响应</div>
        </div>
      </div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc pd-mental-model-html
Disaggregated Prefill 将推理分为两种节点：

**Prefill 节点**：计算密集型，负责处理长 prompt 和生成 KV Cache。完成后通过 KV Connector 异步传输 KV Cache 到 Decode 节点。Prefill 节点可以使用大 GPU 集群并行处理多个长 prompt。

**Decode 节点**：内存带宽密集型，负责加载 KV Cache 后逐 token 生成输出。Decode 节点不受长 prefill 阻塞，可以持续保持低延迟响应。

两种节点之间通过 KV Connector（支持 NIXL、Mooncake 等传输协议）异步传输 KV Cache。
:::

## 为什么需要 P/D 分离

Prefill 是计算密集型（compute-bound），decode 是内存带宽密集型（memory-bandwidth-bound）。混合部署时：

1. **长 prefill 阻塞 decode**：一个长 prefill 请求占满计算资源，decode 请求排队等待，TTFT 和 TBT（Time Between Tokens）均增大
2. **资源配置矛盾**：prefill 需要高算力（更多 SM），decode 需要高带宽（更大显存带宽），同一 GPU 无法同时最优化
3. **批处理干扰**：prefill 和 decode 混合调度时，batch 组成变化导致 GPU 利用率波动

分离后，prefill 节点专注高吞吐的 prompt 处理，decode 节点专注低延迟的 token 生成，各自独立扩缩容。

## KVConnector 双侧接口

`KVConnectorBase_V1` 定义了 Scheduler 侧和 Worker 侧的异步接口，两侧分别运行在不同的 EngineCore 进程中：

### Scheduler 侧

| 方法 | 说明 |
|------|------|
| `get_num_new_matched_tokens()` | 查询远程 KV cache 中已存在的 token 数，减少重复计算 |
| `update_state_after_alloc()` | KV 缓冲区分配后更新 connector 状态 |
| `build_connector_meta()` | 构建传递给 Worker 的 connector 元数据 |
| `request_finished()` | 请求完成时决定是否异步释放 blocks 或触发 KV 保存 |

### Worker 侧

| 方法 | 说明 |
|------|------|
| `start_load_kv()` | 开始异步加载所有层的 KV Cache |
| `wait_for_layer_load(i)` | 阻塞直到第 i 层 KV 加载完成 |
| `save_kv_layer(i)` | 开始异步保存第 i 层 KV Cache |
| `wait_for_save()` | 阻塞直到所有层保存完成 |
| `get_finished()` | 返回已完成 KV 传输的请求列表 |

`vllm/distributed/kv_transfer/kv_connector/v1/base.py`

## 已注册的 Connector

| Connector | 传输协议 | 特点 |
|-----------|---------|------|
| P2pNcclConnector | NCCL P2P | 点对点传输，简单直接 |
| LMCacheConnectorV1 | LMCache | 外部 KV cache 服务 |
| NixlConnector | NIXL | NVIDIA 高性能传输库 |
| MultiConnector | 多 Connector 组合 | 聚合多个 connector |
| MoRIIOConnector | MoRI I/O | I/O based 传输 |
| MooncakeConnector | Mooncake | Mooncake 传输协议 |
| HF3FSKVConnector | 3FS | HuggingFace 3FS 存储 |
| OffloadingConnector | CPU Offload | CPU 内存卸载 |
| FlexKVConnectorV1 | FlexKV | FlexKV 分布式缓存 |
| SimpleCPUOffloadConnector | CPU Offload | 简单 CPU 卸载实现 |
| ExampleConnector | — | 示例实现，供开发者参考 |

`vllm/distributed/kv_transfer/kv_connector/v1/`

## P→D 传输流程

:::steps id=pd-transfer-flow
### 1. Prefill 节点：正常执行
请求到达 Prefill 节点，`get_num_new_matched_tokens()` 返回 0（远程无缓存），正常执行 prefill 计算所有层的 KV Cache。
`vllm/distributed/kv_transfer/kv_connector/v1/base.py`

### 2. Prefill 节点：异步保存 KV
请求完成后 `request_finished()` 返回 True，触发异步保存。逐层调用 `save_kv_layer(i)`，通过传输协议（NIXL/Mooncake 等）将 KV Cache 发送到 Decode 节点。调用 `wait_for_save()` 确保所有层保存完成。
`vllm/distributed/kv_transfer/kv_connector/v1/base.py`

### 3. Decode 节点：查询缓存命中
请求到达 Decode 节点，`get_num_new_matched_tokens()` 返回远程已缓存的 token 数。若有缓存命中，只需计算未缓存部分的 token。
`vllm/distributed/kv_transfer/kv_connector/v1/base.py`

### 4. Decode 节点：异步加载 KV
调用 `start_load_kv()` 开始异步加载所有层的 KV Cache。`wait_for_layer_load(i)` 逐层等待加载完成。
`vllm/distributed/kv_transfer/kv_connector/v1/base.py`

### 5. Decode 节点：正常 Decode
KV Cache 加载完成后，Decode 节点正常执行 decode 循环，逐 token 生成输出。
`vllm/distributed/kv_transfer/kv_connector/v1/base.py`
:::

## Worker 侧管线化

KV 加载与 attention 计算可以管线化（pipeline），隐藏传输延迟：

```python
def execute_model(self, scheduler_output):
    self.connector.start_load_kv(scheduler_output)
    for layer_id in range(num_layers):
        self.connector.wait_for_layer_load(layer_id)
        hidden_states = self.model.layers[layer_id](hidden_states)
```

`start_load_kv()` 在 forward pass 之前调用，启动所有层的异步加载。`wait_for_layer_load(i)` 在计算第 i 层 attention 之前阻塞等待，确保该层 KV 已就绪。这样第 i 层 attention 计算与第 i+1 层 KV 加载可以重叠执行：

```
时间轴：
  [加载 Layer 0 KV] [加载 Layer 1 KV] [加载 Layer 2 KV] ...
  ================== [计算 Layer 0 Attention] [计算 Layer 1 Attention] ...
                     ↑ 等 L0 就绪           ↑ 等 L1 就绪
```

`vllm/distributed/kv_transfer/kv_connector/v1/base.py`

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--kv-transfer-config` | None | KV 传输配置 JSON，包含 connector 类型、角色（prefill/decode）等 |
| `--kv-connector` | None | KV Connector 名称（如 "P2pNcclConnector", "NixlConnector" 等） |
| `--kv-role` | None | 节点角色：prefill 或 decode |
| `--kv-rank` | 0 | KV 传输的 rank ID |
| `--kv-parallel-size` | 1 | KV 传输并行度 |
| `--kv-buffer-device` | "cuda" | KV 缓冲区设备（cuda / cpu） |
| `--kv-ip` | "127.0.0.1" | 对端节点的 IP 地址 |
| `--kv-port` | 14579 | 对端节点的端口号 |
