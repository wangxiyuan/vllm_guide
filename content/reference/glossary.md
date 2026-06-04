---
id: glossary
title: 术语表
category: reference
level: beginner
status: draft
readingMinutes: 8
tags:
  - Glossary
  - Reference
codeRefs: []
heroText: vLLM 关键术语、缩写和页面交叉引用
---

## 核心概念

**EngineCore / 核心引擎**
vLLM V1 的推理核心进程，包含调度器（Scheduler）和执行器（Executor），以 `step()` 循环驱动整个推理流程。详见 [架构总览](../core/architecture.html)

**Scheduler / 调度器**
每轮决定哪些请求参与计算、分配多少 token，管理等待/运行/抢占等状态。详见 [调度原理](../core/scheduler.html)

**Worker / 工作进程**
执行 GPU 计算的进程，内部包含 ModelRunner，由 Executor 统一调度。详见 [架构总览](../core/architecture.html)

**ModelRunner / 模型运行器**
Worker 内负责模型前向计算的组件，接收调度输出、构造输入张量、调用模型并返回输出。

**Executor / 执行器**
EngineCore 内部组件，负责将调度结果分发给各 Worker 并收集模型输出。不同并行策略对应不同 Executor 实现。

**step() / 步进循环**
EngineCore 的主循环函数，每轮调用 schedule → execute → update，是推理推进的基本单位。详见 [架构总览](../core/architecture.html)

**token budget / token 预算**
调度器每一轮最多允许处理的新 token 数，是吞吐和延迟的核心控制阀。详见 [调度原理](../core/scheduler.html)

**continuous batching / 连续批处理**
请求完成后立即释放槽位、新请求立即填入的调度策略，避免传统 static batching 的 padding 浪费。详见 [调度原理](../core/scheduler.html)

**prefill / 预填充**
处理请求 prompt 阶段的首次前向计算，一次性计算所有 prompt token 的 KV cache。

**decode / 解码**
逐 token 生成阶段，每步仅计算一个新 token，复用已有 KV cache。

**prompt / 提示词**
发送给模型的输入文本，对应 prefill 阶段需要处理的全部 token。

**completion / 补全**
模型对 prompt 的生成响应，对应 decode 阶段逐步输出的 token 序列。

## KV Cache

**KV Cache / KV 缓存**
存储 attention 层中 key 和 value 向量的缓存，避免 decode 阶段重复计算历史 token。详见 [KV Cache](../core/kv-cache.html)

**Block / 块**
KV cache 的基本存储单元，每个 block 存放固定数量（block_size）token 的 KV 向量，以分页方式管理显存。详见 [KV Cache](../core/kv-cache.html)

**Block Table / 块表**
记录每个请求的 KV cache 逻辑块到物理块的映射关系，类似操作系统的页表。详见 [PagedAttention](../optimizations/paged-attention.html)

**Slot Mapping / 槽位映射**
将当前步需要写入的 token 位置映射到物理 block 中的具体槽位，用于 attention kernel 的 scatter 写入。详见 [PagedAttention](../optimizations/paged-attention.html)

**block_size / 块大小**
每个 block 存放的 token 数量，默认为 16。决定了 KV cache 分配的最小粒度。详见 [KV Cache](../core/kv-cache.html)

**num_blocks / 块数量**
GPU 显存中预分配的 block 总数，决定了系统可缓存的 token 上限。详见 [KV Cache](../core/kv-cache.html)

**ref_cnt / 引用计数**
记录一个 block 被多少请求引用，用于 prefix caching 场景下的共享块生命周期管理。详见 [KV Cache](../core/kv-cache.html)

**block_hash / 块哈希**
block 内容的哈希值，用于 prefix caching 中识别可复用的 block。详见 [KV Cache](../core/kv-cache.html)

**PagedAttention / 分页注意力**
让 attention kernel 从不连续的显存 block 中读取 KV cache 的算法，消除显存碎片。详见 [PagedAttention](../optimizations/paged-attention.html)

**NHD/HND layout / KV 张量布局**
KV cache 在 GPU 上的存储排布方式。NHD 为逐 block 存储（num_blocks, block_size, num_heads, head_dim），HND 为逐头存储（num_heads, num_blocks, block_size, head_dim）。详见 [PagedAttention](../optimizations/paged-attention.html)

## 调度

**waiting queue / 等待队列**
尚未开始 prefill 的请求队列，调度器按优先级从中选取请求进入运行。详见 [调度原理](../core/scheduler.html)

**running list / 运行列表**
当前正在 decode 或 prefill 的请求集合，调度器每轮为其分配 token budget。详见 [调度原理](../core/scheduler.html)

**skipped_waiting / 跳过等待**
因 token budget 或 KV cache 不足而暂时无法调度的请求，不回到等待队列而是暂存，避免饿死。详见 [调度原理](../core/scheduler.html)

**preemption / 抢占**
当显存不足时，调度器强制回收正在运行请求的 KV cache block，腾出空间给更高优先级的请求。详见 [调度原理](../core/scheduler.html)

**chunked prefill / 分块预填充**
将长 prompt 拆分为多个 chunk，在多个 step 中逐步完成 prefill，避免单个长请求占满 token budget。详见 [调度原理](../core/scheduler.html)

**prefix caching / 前缀缓存**
多个请求共享相同 prompt 前缀时，复用已计算的 KV cache block，避免重复计算。详见 [KV Cache](../core/kv-cache.html)

**num_computed_tokens / 已计算 token 数**
调度器为每个请求维护的计数器，记录已完成的 token 数，是调度决策的核心依据。详见 [调度原理](../core/scheduler.html)

**num_tokens_with_spec / 含投机 token 数**
请求当前需要计算的总 token 数，包含已确定的 prompt/decode token 和投机解码产生的 draft token。详见 [调度原理](../core/scheduler.html)

**is_prefill_chunk / 是否为 prefill 分块**
标识当前 step 的请求是否处于 chunked prefill 的中间阶段，影响调度器的预算分配策略。详见 [调度原理](../core/scheduler.html)

**FCFS / 先来先服务**
默认的请求调度策略，按请求到达顺序分配计算资源。详见 [调度原理](../core/scheduler.html)

**PRIORITY / 优先级调度**
用户可配置的调度策略，按请求优先级而非到达顺序分配资源。详见 [调度原理](../core/scheduler.html)

## 通信

**ZMQ / ZeroMQ**
vLLM 跨进程通信的核心库，用于 API 进程与 EngineCore 之间传递控制消息和请求。详见 [进程与通信](../distributed/process-communication.html)

**ROUTER/DEALER / 路由/经销商模式**
ZMQ 的通信模式，ROUTER 可异步接收多个 DEALER 的消息并路由回复，用于 EngineCore 与 Worker 间的命令分发。详见 [进程与通信](../distributed/process-communication.html)

**PUSH/PULL / 推/拉模式**
ZMQ 的单向通信模式，PUSH 端发送消息，PULL 端接收，用于日志收集等场景。详见 [进程与通信](../distributed/process-communication.html)

**ShmRingBuffer / 共享内存环形缓冲区**
基于 POSIX 共享内存实现的无锁环形缓冲区，用于 EngineCore 与 Worker 间高效传递张量数据。详见 [进程与通信](../distributed/process-communication.html)

**MessageQueue / 消息队列**
EngineCore 进程内部的线程间通信机制，基于 Python Queue 实现，连接主线程、调度线程和模型线程。详见 [进程与通信](../distributed/process-communication.html)

**NCCL**
NVIDIA GPU 集合通信库，vLLM 的 TP/SP/CP 等并行策略依赖它进行跨 GPU 张量通信。详见 [进程与通信](../distributed/process-communication.html)

**Gloo**
Facebook 开发的集合通信库，可作为 NCCL 的 CPU 后端替代，用于 CPU 张量的跨进程通信。

**all-reduce / 全归约**
集合通信原语，所有 rank 各自贡献一个张量，归约（求和等）后结果广播到所有 rank。用于 TP 的 ColumnParallelLinear + RowParallelLinear 组合。详见 [并行策略总览](../distributed/parallelism.html)

**all-gather / 全收集**
集合通信原语，每个 rank 贡献自己的张量片段，所有 rank 拼接后获得完整张量。用于 TP 中 RowParallelLinear 的前向计算。详见 [并行策略总览](../distributed/parallelism.html)

**reduce-scatter / 归约散射**
集合通信原语，先对所有 rank 的张量做归约，再将结果按片段分发给各 rank。是 all-reduce 的高效分解形式。详见 [并行策略总览](../distributed/parallelism.html)

**all-to-all / 全互联**
集合通信原语，每个 rank 向其他所有 rank 发送不同的数据片段，同时从其他 rank 接收。用于 CP 和 EP 的数据重分布。详见 [Context Parallel](../distributed/context-parallel.html)

**send/recv / 发送/接收**
点对点通信原语，一个 rank 发送、另一个 rank 接收。用于 PP 的阶段间中间张量传递。详见 [并行策略总览](../distributed/parallelism.html)

**broadcast / 广播**
集合通信原语，一个 rank 将数据发送给所有其他 rank。用于模型初始化时的参数同步。

**custom all-reduce / 自定义全归约**
vLLM 基于 symmetric memory 和 IPC semaphore 实现的用户态 all-reduce，绕过 NCCL 以降低小消息延迟。详见 [进程与通信](../distributed/process-communication.html)

**symmetric memory / 对称内存**
在所有 GPU 上分配相同虚拟地址的显存区域，使 IPC 直传指针即可访问远端数据，是 custom all-reduce 的基础。详见 [进程与通信](../distributed/process-communication.html)

**IPC semaphore / 进程间信号量**
用于跨进程同步的操作系统原语，在 custom all-reduce 中协调各 GPU 的通信阶段。详见 [进程与通信](../distributed/process-communication.html)

## 并行

**TP (Tensor Parallel) / 张量并行**
将模型权重张量切分到多个 GPU，每个 GPU 计算部分结果后通过 all-reduce 通信合并。适用于单机多卡场景。详见 [并行策略总览](../distributed/parallelism.html)

**PP (Pipeline Parallel) / 流水线并行**
将模型按层切分到不同 GPU，形成流水线，阶段间通过 send/recv 传递中间张量。适用于超大模型跨机场景。详见 [并行策略总览](../distributed/parallelism.html)

**DP (Data Parallel) / 数据并行**
多个 GPU 各自持有完整模型副本，独立处理不同请求批次，梯度同步更新。详见 [并行策略总览](../distributed/parallelism.html)

**EP (Expert Parallel) / 专家并行**
将 MoE 模型的专家网络切分到不同 GPU，路由器动态将 token 分配到对应专家所在 GPU。详见 [并行策略总览](../distributed/parallelism.html)

**SP (Sequence Parallel) / 序列并行**
在 TP 基础上将 attention 和 MLP 层的序列维度切分到多个 GPU，减少激活显存占用，通信使用 all-gather 和 reduce-scatter。详见 [并行策略总览](../distributed/parallelism.html)

**CP (Context Parallel) / 上下文并行**
将长序列的 KV cache 切分到多个 GPU，突破单卡序列长度限制。包含 DCP 和 PCP 两种子策略。详见 [Context Parallel](../distributed/context-parallel.html)

**DCP (Decode Context Parallel) / 解码上下文并行**
CP 的一种模式，复用 TP 的 GPU 资源扩展 decode 阶段的序列长度，通过 LSE 校正合并各 rank 的 attention 输出。详见 [Context Parallel](../distributed/context-parallel.html)

**PCP (Prefill Context Parallel) / 预填充上下文并行**
CP 的一种模式，作为独立并行维度加速 MoE 模型的 prefill 阶段，通过 all-to-all 重分布 token。详见 [Context Parallel](../distributed/context-parallel.html)

**rank / 秩**
并行组中每个进程的唯一编号，用于标识通信参与者。

**world_size / 世界大小**
并行组中的进程总数，决定了张量切分的份数和通信的参与者数量。

**GroupCoordinator / 组协调器**
vLLM 中管理一组进程通信的原语，封装了 NCCL/Gloo 的集合通信操作，按并行策略分组。详见 [进程与通信](../distributed/process-communication.html)

**ProcessGroup / 进程组**
PyTorch 分布式通信的基础抽象，定义了一组可互相通信的进程。vLLM 在其上构建 GroupCoordinator。

## 投机解码

**speculative decoding / 投机解码**
用小模型（proposer）快速生成多个候选 token，再由大模型（target）并行验证，加速推理的解码策略。

**proposer/drafter / 提议者/起草者**
投机解码中负责快速生成候选 token 的小模型或轻量模块，目标是猜测大模型的输出。

**target model / 目标模型**
投机解码中负责验证候选 token 的大模型，决定哪些 draft token 被接受。

**rejection sampling / 拒绝采样**
投机解码的核心验证机制：根据目标模型的概率分布决定是否接受 draft token，被拒绝的 token 及其后缀被丢弃。

**draft token / 草稿 token**
由 proposer 生成的候选 token，等待 target model 验证。

**bonus token / 奖励 token**
验证 draft token 后，目标模型额外生成的一个 token，确保至少每步推进一个 token。

**acceptance rate / 接受率**
draft token 被目标模型接受的比例，是衡量投机解码加速效果的关键指标。

**EAGLE**
基于特征预测的投机解码方法，利用 attention 层输出预测下一个 token 的特征，再映射为 token。

**n-gram**
基于 n-gram 统计的投机解码方法，从历史输出中查找匹配的 n-gram 序列作为 draft token。

**Medusa**
在模型头部添加多个预测头的投机解码方法，每个头独立预测不同位置的 token。

**MTP (Multi-Token Prediction) / 多 token 预测**
训练时让模型同时预测多个未来 token 的方法，也可用于投机解码的 draft 生成。

**DFlash**
一种基于动态规划的投机解码验证算法，优化批量 draft token 的验证效率。

**suffix decoding / 后缀解码**
基于后缀自动机的投机解码方法，从 prompt 和已生成文本中查找可复用的后缀作为 draft token。

## 量化

**quantization / 量化**
将模型权重或激活从高精度（FP32/FP16）压缩到低精度（INT8/INT4/FP8），减少显存占用和计算量。

**FP8**
8 位浮点格式（E4M3 或 E5M2），在保持数值范围的同时将显存需求减半，适合 weight-only 和混合精度计算。

**INT4**
4 位整数格式，极致压缩权重但精度损失较大，通常需要分组或补偿机制。

**INT8**
8 位整数格式，在压缩率和精度间取得较好平衡，广泛用于 weight-only 和激活量化。

**AWQ (Activation-aware Weight Quantization) / 激活感知权重量化**
基于激活值重要性对权重进行分组量化的方法，保护对输出影响大的权重通道。详见 [量化](../optimizations/quantization.html)

**GPTQ**
基于近似二阶信息的训练后量化方法，逐层最小化量化误差，支持 INT4/INT8 权重量化。

**Marlin kernel / Marlin 内核**
专为 INT4/AWQ 量化设计的高效 GPU kernel，通过混合精度矩阵乘法实现接近 FP16 的推理速度。

**weight-only quantization / 仅权重量化**
仅对模型权重进行量化，激活保持 FP16/BF16 计算，实现简单但加速有限。

**activation quantization / 激活量化**
对模型中间激活值也进行量化，进一步减少计算量和显存，但需要校准数据集和更复杂的数值处理。

**KV cache quantization / KV 缓存量化**
对 KV cache 的 key/value 向量进行 FP8 或 INT8 量化，减少 decode 阶段的显存占用和带宽压力。

**per-tensor / 逐张量**
量化粒度，整个张量使用同一组量化参数（scale/zero_point），实现简单但精度损失较大。

**per-block / 逐块**
量化粒度，将张量按块划分，每块使用独立量化参数，在精度和开销间取得平衡。

**per-channel / 逐通道**
量化粒度，每个输出通道使用独立量化参数，精度最高但存储开销较大。

## 模型结构

**MLA (Multi-head Latent Attention) / 多头潜在注意力**
DeepSeek 提出的注意力变体，将 KV 压缩到低维潜在空间，大幅减少 KV cache 显存占用。

**MoE (Mixture of Experts) / 混合专家**
将 FFN 层替换为多个专家网络，路由器按 token 动态选择少量专家计算，以较少的计算量扩展模型容量。

**routed expert / 路由专家**
MoE 中由路由器动态选择的专家，每个 token 仅激活 top-k 个路由专家。

**shared expert / 共享专家**
MoE 中始终对所有 token 激活的专家，提供通用知识，弥补路由专家的覆盖不足。

**top-k routing / top-k 路由**
MoE 路由器为每个 token 选择得分最高的 k 个专家进行计算，是 MoE 的标准路由策略。

**GQA (Grouped Query Attention) / 分组查询注意力**
将 query 头分组，每组共享一组 KV 头，在 MHA 和 MQA 之间取得 KV cache 大小与模型质量的平衡。

**MQA (Multi-Query Attention) / 多查询注意力**
所有 query 头共享一组 KV 头，极大减少 KV cache 体积，但可能影响模型质量。

**RoPE (Rotary Position Embedding) / 旋转位置编码**
通过旋转矩阵编码相对位置信息的编码方式，支持长度外推，被主流 LLM 广泛采用。

**RMSNorm**
均方根归一化，LayerNorm 的简化变体，去掉均值中心化步骤，计算更快，被 LLaMA 等模型采用。

**SwiGLU**
GLU（Gated Linear Unit）激活函数的 Swish 变体，替代传统 ReLU，在 FFN 层中提供更好的模型性能。

**ColumnParallelLinear / 列并行线性层**
将权重矩阵按列切分到多个 GPU，每个 GPU 计算部分输出，后接 all-reduce 合并。是 TP 的标准线性层切分方式。详见 [并行策略总览](../distributed/parallelism.html)

**RowParallelLinear / 行并行线性层**
将权重矩阵按行切分到多个 GPU，每个 GPU 计算部分结果，后接 all-reduce 或 reduce-scatter 合并。详见 [并行策略总览](../distributed/parallelism.html)
