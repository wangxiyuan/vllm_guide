# 内容编写指南

本文档说明如何在 `content/` 目录下编写 Markdown 内容文件，以及如何构建和验证。

## 快速开始

```
1. 在 content/<分类>/ 下新建 .md 文件
2. 按 frontmatter + body 格式填写内容
3. python scripts/build.py <文件名>
4. 浏览器打开 pages/<分类>/<文件名>.html 预览
5. 确认后提交
```

## 目录结构

```
content/                  # 唯一信源（Markdown）
├── core/                 # 核心运行链路
├── distributed/          # 并行与通信
├── decoding/             # 解码算法
├── optimizations/        # 性能优化
├── models/               # 模型结构
└── reference/            # 术语表、指南

pages/                    # 构建产物（HTML，不要手动编辑）
├── core/
├── distributed/
└── ...

scripts/build.py          # Markdown → HTML 构建脚本
```

**核心原则**：`content/*.md` 是唯一信源，`pages/*.html` 是构建产物，每次 `build.py` 运行都会覆盖。不要手动编辑 HTML。

## Frontmatter 规范

每个 `.md` 文件必须以 YAML frontmatter 开头：

```yaml
---
id: my-topic                 # 页面唯一标识，用于 data-page-id
title: 我的专题               # 显示标题
category: core               # 分类，必须和文件所在目录名一致！
level: beginner              # beginner | intermediate | advanced
status: draft                # todo | outline | draft | ready | advanced
readingMinutes: 15           # 预估阅读时间
tags:                        # 标签列表
  - Tag1
  - Tag2
codeRefs:                    # 关键源码位置（首页卡片会引用）
  - vllm/v1/engine/core.py:94
  - vllm/v1/core/sched/scheduler.py:334
heroText: 一句话描述页面内容，支持[术语](term:解释)语法。
---
```

### 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `id` | 是 | 全局唯一，kebab-case |
| `title` | 是 | 中文显示标题 |
| `category` | 是 | **必须和 `content/` 下的子目录名完全一致**（如 `core`、`distributed`、`optimizations`） |
| `level` | 是 | 三选一 |
| `status` | 是 | 五选一，反映内容完成度 |
| `readingMinutes` | 是 | 整数 |
| `tags` | 是 | 列表，至少一个 |
| `codeRefs` | 否 | 列表，源码路径:行号 |
| `heroText` | 否 | 首页卡片副标题，支持 term 语法 |

### category 命名规范

| 目录 | category 值 | 说明 |
|------|-------------|------|
| `content/core/` | `core` | 架构、调度、通信等核心机制 |
| `content/distributed/` | `distributed` | TP/PP/DP/EP/CP 等并行策略 |
| `content/decoding/` | `decoding` | 投机解码、采样等解码算法 |
| `content/optimizations/` | `optimizations` | KV cache、prefix caching、量化等 |
| `content/models/` | `models` | 各模型结构专题 |
| `content/reference/` | `reference` | 术语表、编写指南 |

> **常见错误**：category 写成 `optimization`（单数），但目录是 `optimizations`（复数）。构建会输出到错误的目录。

## 正文结构

每个页面按 `##` 二级标题分节。构建脚本会按 `<h2>` 自动拆分 section。推荐结构：

```markdown
## 心智模型

用类比或简化模型帮读者建立直觉。

## 核心机制

详细解释核心实现。

## 关键配置

配置项汇总表。
```

## 术语提示

使用 `[显示文本](term:tooltip 内容)` 语法，构建时转为带 hover 提示的 `<span>`：

```markdown
[token budget](term:调度器每一轮最多允许处理的新 token 数，是吞吐和延迟的核心控制阀。)
```

- 显示文本会同时作为 `data-term` 属性和页面显示文字
- tooltip 内容放在 `term:` 后面，直到右括号 `)` 结束
- tooltip 中不要包含右括号 `)`

## 图表：双轨制（人看 HTML + Agent 看文字）

图表是内容与呈现分离的核心场景。每个图表写两份：

1. **`:::diagram <id>-html`** — 原始 HTML+CSS，构建时原样嵌入页面（人看）
2. **`:::diagram-desc <id>-html`** — 纯文字 Markdown 描述，构建时渲染为隐藏的 `<template>` 元素（AI agent 读 Markdown 时看到）

### 关键规则：ID 必须一致

`:::diagram` 和 `:::diagram-desc` 的 ID **必须完全相同**：

```markdown
✅ 正确：ID 一致
:::diagram my-topic-html
```html
<div class="...">...</div>
```
:::

:::diagram-desc my-topic-html
纯文字描述这个图表展示的内容...
:::

❌ 错误：ID 不一致
:::diagram my-topic-html
...
:::diagram-desc my-topic          ← 缺少 -html 后缀
...
:::
```

### 关键规则：每个图表 ID 必须唯一

同一页面如果有多个图表，每个必须用不同的 ID：

```markdown
✅ 正确：每个图表有唯一 ID
:::diagram kv-mental-model-html
...
:::
:::diagram-desc kv-mental-model-html
...
:::

:::diagram kv-tensor-layout-html
...
:::
:::diagram-desc kv-tensor-layout-html
...
:::

❌ 错误：多个图表共用同一个 ID
:::diagram kv-html    ← 第 1 个图表
...
:::
:::diagram-desc kv-html
...
:::

:::diagram kv-html    ← 第 2 个图表，ID 重复！
...
:::
:::diagram-desc kv-html
...
:::
```

### 命名建议

ID 格式：`<页面缩写>-<图表内容>-html`

| 页面 | 图表 | 推荐ID |
|------|------|--------|
| scheduler | 调度器心智模型 | `sched-mental-model-html` |
| scheduler | 调度流程 | `sched-flow-html` |
| kv-cache | 分页管理 | `kv-mental-model-html` |
| kv-cache | 张量布局 | `kv-tensor-layout-html` |
| paged-attention | 核心概念 | `pa-mental-model-html` |

### diagram 的 HTML 代码块

`:::diagram` 内部用 ```html 代码块包裹 HTML：

```markdown
:::diagram my-topic-html
```html
<div class="my-diagram">
  <div class="my-box">内容</div>
</div>
```
:::
```

HTML 中的 CSS 类名需要在 `assets/css/components.css` 中定义对应样式。

### diagram-desc 的内容

`:::diagram-desc` 内部写纯 Markdown，描述图表传达的知识：

```markdown
:::diagram-desc my-topic-html
系统包含三个进程：

**API 服务进程**：运行 HTTP API 服务，处理输入输出。

**EngineCore 进程**：核心推理引擎，包含调度器（Scheduler）和执行器（Executor）。

**Worker 进程**：执行 GPU 计算，返回模型输出。

进程间通过 ZMQ 通信。
:::
```

注意：`diagram-desc` 的内容会出现在 Markdown 视图（MD 按钮）中，所以不要和正文中的描述重复。

## 步骤播放器

用 `:::steps` 块定义可交互的步骤序列：

```markdown
:::steps id=my-steps
### 1. 第一步标题
步骤描述文字，支持 `代码` 和 **加粗**。
`vllm/v1/engine/core.py`        ← 源码引用，单独一行

### 2. 第二步标题
步骤描述文字。
`vllm/v1/core/sched/scheduler.py`
:::
```

规则：
- `id` 用于前端 JS 绑定，同一页面内唯一
- 每个步骤以 `### N. 标题` 开头，N 是序号
- 步骤体支持 Markdown 格式
- 单独一行的 `` `vllm/...` `` 会渲染为代码引用样式（`code.step-ref`）
- 构建时自动生成 step-player 控件 + step-list

## 配置表

用标准 Markdown 表格：

```markdown
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `max_num_seqs` | 128 | 最大并发请求数 |
| `block_size` | 16 | 每个 block 存几个 token 的 KV |
```

四列表格也支持：

```markdown
| 场景 | 操作 | 如果失败 | 代码位置 |
|------|------|----------|----------|
| 新请求 | `allocate_slots()` | 返回 None | `scheduler.py` |
```

## 代码块

标准 Markdown 围栏代码块，标注语言：

````markdown
```python
def step(self):
    scheduler_output = self.scheduler.schedule()
    model_output = self.model_executor.execute_model(scheduler_output)
    return self.scheduler.update_from_output(scheduler_output, model_output)
```
````

可以在代码块前一行用普通文字标注源码位置，不用特殊语法。

## 交互模拟

交互模拟（`data-sim`）是纯 JS 驱动的 UI 组件，**不在 Markdown 中声明**。它由 `assets/js/app.js` 中的 `createSim()` 函数管理。

如果页面需要交互模拟，需要同时：
1. 在 HTML 页面中放置 `<div class="interactive-sim" data-sim="name">` 容器
2. 在 `app.js` 中编写模拟步骤数据和渲染函数

目前这个流程无法通过 Markdown 自动生成，需要手动在构建后的 HTML 中添加。如果你需要交互模拟，请先写好其他内容，构建后单独处理模拟部分。

## 构建与验证

```bash
# 构建全部
python scripts/build.py

# 构建单个页面
python scripts/build.py scheduler

# 监听文件变化自动构建（需要 pip install watchdog）
python scripts/build.py --watch
```

验证清单：
1. 构建无报错
2. 浏览器打开 HTML，检查排版和图表
3. 点击 MD 按钮切换 Markdown 视图，确认文字正确、arch-html 块已过滤
4. 检查页面底部的 JS 控件（step-player 等）正常工作
5. 如果有交互模拟，确认模拟按钮可用

## 常见问题

### Q: category 应该写什么？

A: 必须和 `content/` 下的子目录名**完全一致**。`content/core/` 下写 `category: core`，`content/optimizations/` 下写 `category: optimizations`。

### Q: 图表应该用 HTML 还是 Markdown 画？

A: 强 UI 类型的图表（进程框图、流程图、状态机等）用 `:::diagram *-html` 保存原始 HTML。简单表格和列表用 Markdown。同时必须写 `:::diagram-desc` 给 AI agent 用。

### Q: `:::diagram-desc` 和正文内容重复了怎么办？

A: 避免重复。如果正文已经有一段描述，`diagram-desc` 应该只补充图表特有的视觉信息（如"左侧是 API 进程，中间是 EngineCore，右侧是 Worker"），而不是重新写一遍所有知识。或者反过来，正文中省略描述，把完整描述放在 `diagram-desc` 里。

### Q: 步骤中的源码引用怎么写？

A: 单独一行，用反引号包裹路径：`` `vllm/v1/engine/core.py` ``。构建脚本会自动识别以 `vllm/` 开头或包含 `/` 的单行 code，渲染为 `<code class="step-ref">`。

### Q: 怎么添加新的 CSS 样式？

A: 在 `assets/css/components.css` 中添加。`:::diagram *-html` 中的 HTML 可以使用任意 CSS 类名，只要在 `components.css` 中有对应样式即可。

### Q: 旧页面路径变了怎么办？

A: 在 `pages/` 旧路径下放一个 HTML 重定向页：

```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0;url=../新分类/新文件名.html">
  <title>旧标题 - vLLM Visual Guide</title>
</head>
<body>
  <p>此页面已迁移至 <a href="../新分类/新文件名.html">新地址</a></p>
</body>
</html>
```
