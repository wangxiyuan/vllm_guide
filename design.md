# vLLM 技术可视化网站设计文档

## 1. 目标

建设一个由多个 HTML 页面组成的 vLLM 技术可视化网站，用动画、交互式 UI、流程图和源码索引解释 vLLM 的核心实现。

核心目标：

1. 展示请求从 API 进入到 token 输出的完整生命周期。
2. 用可视化方式讲清调度、KV Cache、并行、模型结构、投机解码等机制。
3. 将概念、动画和源码位置建立稳定映射。
4. 首页采用数据驱动卡片，方便持续新增专题。
5. 每个专题页独立维护，避免单页复杂化。

## 2. vLLM 核心链路

```text
用户请求 / OpenAI API / LLM.generate
  → Engine / AsyncLLM
  → EngineCore
  → Scheduler
  → Executor
  → Worker / ModelRunner
  → ModelExecutor / Attention / Sampler
  → Scheduler.update_from_output
  → OutputProcessor / Streaming Output
```

关键源码索引：

| 模块 | 作用 | 代码位置 |
|---|---|---|
| EngineCore | 内部主循环 | `vllm/v1/engine/core.py:94` |
| Executor 初始化 | 创建模型执行器 | `vllm/v1/engine/core.py:121` |
| KV Cache 初始化 | profile 并生成 cache config | `vllm/v1/engine/core.py:131` |
| Scheduler 初始化 | 创建调度器 | `vllm/v1/engine/core.py:148` |
| 单步执行 | schedule → execute → update | `vllm/v1/engine/core.py:439` |
| Scheduler 核心算法 | token budget 调度 | `vllm/v1/core/sched/scheduler.py:334` |
| RUNNING 请求调度 | 优先调度运行中请求 | `vllm/v1/core/sched/scheduler.py:369` |
| KV block 分配 | 为请求分配 cache blocks | `vllm/v1/core/sched/scheduler.py:446` |
| 请求抢占 | KV 不足时 preempt | `vllm/v1/core/sched/scheduler.py:459` |
| 输出更新 | 处理模型输出和请求状态 | `vllm/v1/core/sched/scheduler.py:1289` |
| 多进程执行器 | 本地多进程 worker | `vllm/v1/executor/multiproc_executor.py:103` |
| 分布式状态 | TP/PP/DP/EP group | `vllm/distributed/parallel_state.py:8` |
| Llama 模型 | dense decoder-only 结构 | `vllm/model_executor/models/llama.py:81` |
| DeepSeek / MoE | MLA 与 MoE 结构 | `vllm/model_executor/models/deepseek_v2.py:50` |
| Draft Model | 投机解码 draft 模型 | `vllm/v1/spec_decode/draft_model.py:17` |
| Rejection Sampler | 投机解码接受/拒绝 | `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:40` |

## 3. 实际目录结构

项目骨架已经初始化。

```text
guide/
├── index.html
├── design.md
├── README.md
├── pages/
│   ├── core/
│   │   ├── architecture.html
│   │   └── scheduler.html
│   ├── distributed/
│   │   ├── process-communication.html
│   │   ├── parallelism.html
│   │   ├── tensor-parallel.html
│   │   ├── pipeline-parallel.html
│   │   ├── expert-parallel.html
│   │   └── disaggregated-prefill.html
│   ├── decoding/
│   │   ├── speculative-decoding.html
│   │   ├── eagle.html
│   │   └── ngram.html
│   ├── optimizations/
│   │   ├── kv-cache.html
│   │   ├── paged-attention.html
│   │   ├── prefix-caching.html
│   │   ├── chunked-prefill.html
│   │   └── quantization.html
│   ├── models/
│   │   ├── index.html
│   │   ├── qwen3-moe.html
│   │   ├── llama.html
│   │   └── deepseek-mla.html
│   └── reference/
│       ├── glossary.html
│       └── contribution-guide.html
└── assets/
    ├── css/{base,layout,components,animations}.css
    ├── js/{app,home,site-data,animation-engine,diagrams,code-map}.js
    ├── data/{topics,learning-paths,code-map,scheduler-steps,kv-cache-steps,parallelism-cases,model-structures,speculative-decoding-steps}.json
    ├── templates/topic-page.html
    ├── images/
    └── ../scripts/bundle-data.py   # 把 assets/data/*.json 打包到 assets/js/site-data.js
```

旧的扁平 `pages/<topic>.html` 文件保留为重定向页，避免外部链接断裂。

**部署模式：纯静态 / 零服务**

- 站点不依赖任何运行时服务，所有数据通过 `scripts/bundle-data.py` 预打包到 `assets/js/site-data.js`，以 `window.SITE_DATA` 暴露给前端。
- 不允许在前端使用 `fetch` / `XMLHttpRequest` 加载本地 JSON，避免在 `file://` 与 GitHub Pages 子路径下出问题。
- 直接双击 `index.html` 即可预览；可选地起 `python -m http.server` 也可以。
- 可直接部署到 GitHub Pages 等静态托管。修改 `assets/data/*.json` 后必须运行打包脚本刷新 `site-data.js`。

## 4. 首页设计

首页定位为专题入口列表，简洁清晰：一个标题 + 一段说明 + 从上到下的卡片列表，每张卡片对应一个专题。

```text
Header: vLLM Visual Guide（仅品牌标识，无导航链接）
Hero:   标题 + 一句话说明
Cards:  从上到下排列，每张卡片 = 一个专题（分类标签 / 标题 / 副标题 / 状态 / 阅读时间）
```

卡片由 `assets/data/topics.json` 驱动，通过 `scripts/bundle-data.py` 预打包到 `assets/js/site-data.js`，前端只从 `window.SITE_DATA` 读取。

```json
{
  "id": "scheduler",
  "title": "调度原理",
  "subtitle": "连续批处理、chunked prefill、抢占与 token budget",
  "href": "pages/core/scheduler.html",
  "category": "core",
  "level": "intermediate",
  "status": "outline",
  "readingMinutes": 15,
  "tags": ["Scheduler", "Continuous Batching", "KV Cache"],
  "codeRefs": [
    "vllm/v1/core/sched/scheduler.py:334",
    "vllm/v1/engine/core.py:439"
  ]
}
```

状态枚举：`todo`、`outline`、`draft`、`ready`、`advanced`。

## 5. 专题页组织规则

`pages/` 按领域组织子目录，避免专题增多后难以维护：

```text
pages/
├── core/             # 架构、调度等核心运行链路
├── distributed/      # 进程通信、并行、KV 传输
├── decoding/         # 投机解码、采样等解码算法
├── optimizations/    # KV cache、prefix caching、量化等性能专题
├── models/           # 各模型结构专题
└── reference/        # 术语表、内容填充指南
```

规则：

1. 新增专题先选定子目录，没有合适分类则新增子目录。
2. 模型类专题统一放入 `pages/models/`。
3. 首页所有路径来自 `assets/data/topics.json`，移动页面时必须同步更新数据。
4. 旧路径必须保留重定向页或留下索引。
5. 新增页面通过复制 `assets/templates/topic-page.html`。

## 6. 专题页统一模板

```html
<main class="topic-page">
  <section class="topic-hero">       <!-- 标题、分类、状态、标签 -->
  <section>心智模型</section>         <!-- 概念图或简化流程 -->
  <section>交互演示</section>         <!-- step player、流程图、拓扑 -->
</main>
```

每页固定回答：这个机制解决什么问题、它位于请求生命周期何处、核心数据结构和状态流、关键源码入口、有哪些配置项会影响行为、常见误解和注意事项。

## 7. 专题规划

### 7.1 架构总览

页面：`pages/core/architecture.html`（旧路径 `pages/architecture.html` 保留为重定向）

核心问题：一次请求如何从 API 到 GPU 执行再返回结果。

动画：请求 token 从左到右流动，EngineCore 内部高亮 `schedule → execute → update`。

代码：`vllm/v1/engine/core.py:94`、`vllm/v1/engine/core.py:439`、`vllm/v1/engine/core.py:464`。

### 7.2 进程与通信

页面：`pages/distributed/process-communication.html`（旧路径 `pages/process-communication.html` 保留为重定向）

解释单进程 executor、多进程 executor、Ray executor、SchedulerOutput 广播、ModelRunnerOutput 聚合、TP/PP/DP rank 关系、MessageQueue、跨节点 distributed init。

交互：选择 `tp=2, pp=2, dp=1` 后生成 rank 拓扑，并动画展示下发与回传。

代码：`vllm/v1/executor/multiproc_executor.py:103`、`vllm/v1/executor/multiproc_executor.py:151`、`vllm/distributed/parallel_state.py:8`。

### 7.3 调度原理

页面：`pages/core/scheduler.html`（旧路径 `pages/scheduler.html` 保留为重定向）

核心概念：waiting queue、running list、token budget、KV block budget、`num_computed_tokens`、`num_tokens_with_spec`、chunked prefill、preemption、speculative tokens。

重要结论：V1 Scheduler 没有严格区分 prefill/decode phase，而是让每个请求的 `num_computed_tokens` 追赶 `num_tokens_with_spec`。

动画步骤：waiting 入队 → running 优先消耗 budget → 新请求获得剩余 budget → KV block 分配 → KV 不足抢占 → 输出 token 后更新状态。

代码：`vllm/v1/core/sched/scheduler.py:334`、`vllm/v1/core/sched/scheduler.py:369`、`vllm/v1/core/sched/scheduler.py:446`、`vllm/v1/core/sched/scheduler.py:1289`。

### 7.4 KV Cache 与 PagedAttention

页面：`pages/optimizations/kv-cache.html`（旧路径 `pages/kv-cache.html` 保留为重定向）

解释 KV Cache 为什么关键、连续内存问题、block-based cache、block table、prefix caching、preemption、remote KV connector。

动画：逻辑 token 序列映射到物理 KV blocks。

代码：`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/block_pool.py`、`vllm/v1/core/kv_cache_utils.py`。

### 7.5 并行策略

页面：`pages/distributed/parallelism.html`（旧路径 `pages/parallelism.html` 保留为重定向；细分专题见 `pages/distributed/{tensor,pipeline,expert}-parallel.html`）

覆盖 Tensor Parallel、Pipeline Parallel、Data Parallel、Expert Parallel、Context Parallel、Prefill Context Parallel、Decode Context Parallel。

交互：输入 GPU 数和 TP/PP/DP/EP 配置，生成 rank 拓扑，展示 forward 中的 all-reduce、all-gather、all-to-all。

代码：`vllm/distributed/parallel_state.py:12`、`vllm/model_executor/layers/linear.py:11`、`vllm/model_executor/models/llama.py:142`。

### 7.6 模型结构

页面：`pages/models/index.html`（旧路径 `pages/model-structures.html` 保留为重定向；各模型独立页见 `pages/models/{qwen3-moe,llama,deepseek-mla}.html`）

首批覆盖 Llama/Qwen dense decoder-only、Qwen3 MoE、DeepSeek MLA、Mixtral/DeepSeek MoE、Mamba/hybrid attention、多模态模型。

通用结构：Embedding → Decoder Layers → Attention/MLA → MLP/MoE → LM Head。

代码：`vllm/model_executor/models/llama.py:81`、`vllm/model_executor/models/llama.py:124`、`vllm/model_executor/models/deepseek_v2.py:50`。

### 7.7 投机解码

页面：`pages/decoding/speculative-decoding.html`（旧路径 `pages/speculative-decoding.html` 保留为重定向；细分专题 `pages/decoding/{eagle,ngram}.html`）

解释 draft model 生成候选 token，target model 并行验证，接受连续匹配前缀，遇到拒绝 token 后重新采样。

```text
Draft:   A  B  C  D
Target:  A  B  X
Accept:  A  B
Reject:      C D
Output:  A B X
```

覆盖 draft model、n-gram、suffix decoding、EAGLE、Medusa、MTP、DFlash。

代码：`vllm/v1/worker/gpu_model_runner.py:536`、`vllm/v1/spec_decode/draft_model.py:17`、`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:40`、`vllm/v1/core/sched/scheduler.py:1374`。

## 8. 动画系统设计

动画不应硬编码在页面中，推荐由 JSON step 数据驱动。

```json
[
  {
    "step": 1,
    "title": "新请求进入 waiting 队列",
    "actions": [
      {"type": "add-request", "target": "waiting", "id": "req-1"}
    ]
  },
  {
    "step": 2,
    "title": "Scheduler 分配 token budget",
    "actions": [
      {"type": "move-request", "from": "waiting", "to": "running", "id": "req-1"},
      {"type": "allocate-block", "request": "req-1", "block": "block-12"}
    ]
  }
]
```

统一动作类型：`add-request`、`move-request`、`allocate-block`、`free-block`、`send-message`、`all-reduce`、`all-gather`、`all-to-all`、`accept-token`、`reject-token`。

## 9. 视觉规范

建议采用深色技术风格，颜色语义固定：

```text
蓝色：Request / Token
绿色：KV Cache / Block
紫色：Model / Layer
橙色：Scheduler decision
黄色：Communication / Collective
红色：Preemption / Reject / Error
```

组件统一：技术卡片、源码引用块、步骤播放器、时间轴、拓扑图、状态徽标、参数控制面板。

## 10. MVP 范围

第一阶段建议完成：

1. `index.html`
2. `pages/core/architecture.html`
3. `pages/core/scheduler.html`
4. `pages/optimizations/kv-cache.html`
5. `pages/decoding/speculative-decoding.html`
6. `assets/data/topics.json`（数据源）+ `assets/js/site-data.js`（打包产物）
7. `assets/js/animation-engine.js`
8. `scripts/bundle-data.py`（数据打包脚本）

## 11. 维护原则

1. 新增专题优先新增页面和数据，不改首页结构。
2. 首页只从 `window.SITE_DATA.topics` 渲染卡片列表。
3. 不允许引入运行时服务或 `fetch` 本地 JSON；任何破坏纯静态约束的改动必须同步更新 `design.md`。
4. 动画优先用 step JSON 描述，少写页面专属逻辑（step 数据通过 `bundle-data.py` 打包到 `SITE_DATA.steps` / `SITE_DATA.cases`）。
5. 每个专题必须包含源码引用。
6. 每个专题必须说明它在请求生命周期中的位置。
7. 页面命名使用 kebab-case。
8. 模型专题统一放在 `pages/models/`。
