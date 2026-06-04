---
id: structured-output
title: 结构化输出
category: execution
level: intermediate
status: draft
readingMinutes: 14
tags:
  - Grammar
  - Bitmask
  - JSON
  - Regex
codeRefs:
  - vllm/v1/structured_output/__init__.py
  - vllm/v1/worker/gpu/structured_outputs.py
heroText: "通过 [grammar bitmask](term:由语法后端生成的位掩码张量，标记每个 token 是否符合当前语法约束，在采样阶段屏蔽非法 token。) 在采样阶段强制约束输出格式，支持 JSON/Regex/Choice/Grammar 四种结构化类型。"
---

## 心智模型

核心机制：**采样前由 grammar 后端根据当前前缀生成 bitmask 张量，标记每个 vocab token 是否合法，在 logits 上直接屏蔽非法 token。** 模型被约束在合法语法空间内采样。

:::diagram structured-output-html
```html
<div class="engine-step-flow">
  <div class="engine-step">
    <div class="engine-step-num">1</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Grammar 编译</div>
      <div class="engine-step-desc">JSON Schema / Regex / Grammar → 状态机</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">2</div>
    <div class="engine-step-content">
      <div class="engine-step-title">Bitmask 生成</div>
      <div class="engine-step-desc">根据前缀状态生成 vocab 大小的位掩码张量</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">3</div>
    <div class="engine-step-content">
      <div class="engine-step-title">采样屏蔽</div>
      <div class="engine-step-desc">logits 上应用 bitmask，非法 token 概率置零</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc structured-output-html
结构化输出的核心流程分三步：

**Grammar 编译**：请求到达时，grammar 后端将 JSON Schema / Regex / Grammar 等约束规范编译为状态机（FSM 或 CFG 解析器）。编译是异步进行的，期间请求处于 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR 状态。

**Bitmask 生成**：每步采样前，grammar 后端根据当前前缀对应的语法状态，生成形状为 [batch_size, vocab_size] 的位掩码张量。每个位置为 0 或 1，表示对应 token 在当前状态下是否合法。

**采样屏蔽**：将 bitmask 应用到 logits 上，执行 logits[~mask] = -inf，非法 token 的概率被归零。模型只能从合法 token 中采样，保证输出符合约束。
:::

## Grammar Bitmask 机制

Bitmask 是结构化输出的核心数据结构。形状 `[batch_size, vocab_size]`，每个位置 0/1 表示 token 是否允许出现在当前位置。

### 生成与消费

Bitmask 的生命周期完全嵌入 EngineCore 的 step 循环中：

- **生成时机**：每步采样前，`scheduler.get_grammar_bitmask()` 调用 grammar 后端
- **与 logits 结合**：`logits[~mask] = -inf`，在采样前直接屏蔽
- **状态机增量推进**：每生成一个 token，grammar 后端更新内部状态，为下一步生成新的 bitmask

```python
bitmask = scheduler.get_grammar_bitmask(scheduler_output, model_output)
sampled = model_executor.sample_tokens(logits, grammar_bitmask=bitmask)
scheduler.update_from_output(scheduler_output, model_output)
```

`vllm/v1/worker/gpu/structured_outputs.py`

### Bitmask 与 GPU 的协作

Bitmask 在 CPU 上生成，然后传输到 GPU 应用到 logits。这个设计的关键在于：GPU 前向计算与 CPU bitmask 生成是**并行**的——GPU 执行 model forward 的同时，CPU 在计算 grammar bitmask，两者通过 future 同步。

:::diagram so-gpu-cpu-html
```html
<div class="engine-step-flow">
  <div class="engine-step">
    <div class="engine-step-num">GPU</div>
    <div class="engine-step-content">
      <div class="engine-step-title">model forward</div>
      <div class="engine-step-desc">执行前向计算，产出 logits（异步 future）</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">CPU</div>
    <div class="engine-step-content">
      <div class="engine-step-title">grammar bitmask</div>
      <div class="engine-step-desc">根据前缀状态生成位掩码（与 GPU 并行）</div>
    </div>
  </div>
  <div class="engine-step">
    <div class="engine-step-num">同步</div>
    <div class="engine-step-content">
      <div class="engine-step-title">future + bitmask → 采样</div>
      <div class="engine-step-desc">logits[~mask] = -inf → sample</div>
    </div>
  </div>
</div>
```
:::

:::diagram-desc so-gpu-cpu-html
GPU 与 CPU 并行协作：GPU 执行 model forward 产出 logits（异步 future），同时 CPU 根据前缀状态生成 grammar bitmask。两者通过 future 同步后，将 bitmask 应用到 logits 上（logits[~mask] = -inf），然后执行采样。这种并行设计将 bitmask 生成延迟隐藏在 GPU 前向计算时间内。
:::

## 请求生命周期

当请求携带结构化输出约束时，其生命周期比普通请求多一个 grammar 编译阶段：

:::diagram so-lifecycle-html
```html
<div class="state-flow">
  <div class="state-node state-waiting">WAITING_FOR_GRAMMAR</div>
  <div class="state-arrow">编译完成</div>
  <div class="state-node state-waiting">WAITING</div>
  <div class="state-arrow">schedule()</div>
  <div class="state-node state-running">RUNNING</div>
  <div class="state-arrow">每步采样</div>
  <div class="state-node state-running">bitmask → 屏蔽 → 采样</div>
  <div class="state-arrow">完成</div>
  <div class="state-node state-finished">FINISHED</div>
</div>
```
:::

:::diagram-desc so-lifecycle-html
结构化输出请求的生命周期：请求到达后首先进入 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR 状态，grammar 后端异步编译约束规范为状态机。编译完成后状态变为 WAITING，可以正常被调度。进入 RUNNING 后，每步采样前都会生成 grammar bitmask 并应用到 logits 上，确保输出始终符合约束。最终完成时进入 FINISHED 状态。
:::

### 编译阶段

1. 请求到达 → grammar 后端开始异步编译（JSON Schema → FSM、Regex → FSM、CFG → 解析器）
2. 编译期间请求状态为 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`，存放在 [skipped_waiting](term:调度器中存放被阻塞请求的队列，包括等待 KV 传输、grammar 编译、流式输入的请求。) 队列
3. 编译完成 → 状态变为 `WAITING`，进入正常调度队列

`vllm/v1/structured_output/__init__.py`

### 运行阶段

每一步 EngineCore.step() 中，grammar bitmask 的生成与消费流程：

:::steps id=grammar-step-player
### 1. scheduler.schedule()
调度器选出本步要处理的请求，返回 SchedulerOutput。
`vllm/v1/core/sched/scheduler.py`

### 2. model_executor.execute_model()
GPU 异步执行模型前向计算，返回 future。
`vllm/v1/executor/abstract.py`

### 3. scheduler.get_grammar_bitmask()
CPU 并行生成 grammar bitmask，与 GPU 计算同时进行。
`vllm/v1/core/sched/scheduler.py`

### 4. future.result()
等待 GPU 前向计算完成，获取 logits。

### 5. model_executor.sample_tokens(grammar_output)
将 bitmask 应用到 logits 上（logits[~mask] = -inf），执行采样。
`vllm/v1/worker/gpu_model_runner.py`

### 6. scheduler.update_from_output()
更新请求状态，grammar 状态机根据采样结果推进。
`vllm/v1/core/sched/scheduler.py`
:::

## 四种后端

vLLM 支持四种 grammar 后端，通过 `--grammar-backend` 参数选择：

| 后端 | 实现语言 | 特点 | 适用场景 |
|------|----------|------|----------|
| **xgrammar** | C++ | 编译速度快，内存占用低 | 默认选择，JSON/Regex 约束 |
| **guidance** | Python 绑定 | 支持复杂语法和聊天模板 | 复杂约束、聊天模板控制 |
| **outlines** | Python | FSM 方式，社区贡献 | 社区生态兼容 |
| **lm-format-enforcer** | Python | 兼容性后端 | 迁移场景 |

### xgrammar（默认）

xgrammar 是当前默认后端，采用 C++ 实现核心逻辑，通过 pybind11 暴露 Python 接口。主要优势：

- 编译速度：JSON Schema 到 FSM 的编译比纯 Python 实现快数倍
- 运行时性能：bitmask 生成在 C++ 层完成，减少 Python GIL 开销
- 内存效率：紧凑的状态机表示，适合高并发场景

### guidance

guidance 后端通过 Python 绑定调用底层引擎，支持更复杂的语法约束和聊天模板控制。适合需要精细控制输出结构的场景，如混合文本与 JSON 的聊天模板。

### outlines 与 lm-format-enforcer

- **outlines**：社区贡献的后端，采用 FSM（有限状态机）方式实现，与 outlines 库生态兼容
- **lm-format-enforcer**：兼容性后端，方便从已有 lm-format-enforcer 工作流迁移

### 后端选择逻辑

启动时根据 `--grammar-backend` 参数实例化对应后端。若未指定，默认使用 xgrammar。每个请求的 grammar 对象由后端工厂创建，并绑定到请求的 `grammar_bitmask` 属性上。

`vllm/v1/structured_output/__init__.py`

## 结构化输出类型

vLLM 支持四种结构化输出类型，对应 OpenAI API 的不同参数：

| 类型 | API 参数 | 约束方式 | 典型场景 |
|------|----------|----------|----------|
| **JSON** | `guided_json` | JSON Schema → FSM | API 返回结构化数据 |
| **Regex** | `guided_regex` | 正则表达式 → FSM | 手机号、日期等格式 |
| **Choice** | `guided_choice` | 候选列表枚举 | 分类、Yes/No 选择 |
| **Grammar** | `guided_grammar` | 自定义 CFG | 编程语言、DSL 生成 |

### JSON 约束

最常用的结构化输出类型。用户提供 JSON Schema，grammar 后端编译为 FSM，确保输出是合法 JSON 且符合 Schema 定义。

```python
request_params = {
    "prompt": "列出三种水果",
    "guided_json": {
        "type": "object",
        "properties": {
            "fruits": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["fruits"]
    }
}
```

FSM 确保：花括号匹配、键名正确、值类型符合 Schema、数组元素类型正确。

### Regex 约束

正则表达式约束输出格式。grammar 后端将正则编译为 DFA，每个状态对应一组合法字符集。

```python
request_params = {
    "prompt": "提取手机号",
    "guided_regex": r"\d{3}-\d{4}-\d{4}"
}
```

### Choice 约束

限定输出为候选列表中的一个，本质上是枚举所有候选路径的 FSM。

```python
request_params = {
    "prompt": "情感分析",
    "guided_choice": ["positive", "negative", "neutral"]
}
```

### Grammar 约束

自定义上下文无关文法（CFG），最灵活但编译开销最大。适合需要复杂语法规则（如编程语言、DSL）的场景。

## 与投机解码的兼容

结构化输出与[投机解码](term:用小模型快速生成候选 token 序列，再由大模型并行验证的加速技术。)存在交互：draft token 也必须满足 grammar 约束，否则验证阶段会拒绝。

### 兼容策略

- **Draft token 的 bitmask 验证**：在 rejection sampling 阶段，draft token 同样受到 grammar bitmask 约束。若 draft token 不符合当前语法状态，直接拒绝
- **语法状态回滚**：当 draft token 被拒绝时，grammar 状态机需要回滚到接受点对应的状态，而非回退到拒绝前
- **性能影响**：grammar 约束降低了 draft model 的接受率，因为 draft model 可能倾向于生成语法上不合法的 token。约束越严格，接受率下降越明显

### 实现细节

在投机解码的 rejection sampling 阶段：

1. 对每个 draft token，检查其是否满足当前 grammar 状态的 bitmask
2. 若不满足，拒绝该 draft token 及其后续所有 token
3. 接受的 draft token 推进 grammar 状态
4. 拒绝后，grammar 状态回滚到最近一次接受点

`vllm/v1/sample/rejection_sampler.py`

## Reasoning 模式

部分模型（如 DeepSeek-R1、QwQ）混合推理过程与最终答案。结构化输出只约束答案部分，reasoning token 需要识别并跳过语法约束。

### 识别与跳过

- **Reasoning token 识别**：模型输出中，位于 `<think>...</think>` 等推理标签内的 token 被识别为 reasoning token
- **跳过语法约束**：reasoning token 不受 grammar bitmask 约束，可以自由生成
- **答案部分约束**：推理标签结束后，grammar bitmask 恢复生效，约束最终答案的格式

### 实现机制

grammar 后端在每步生成 bitmask 时，会检查当前 token 是否处于推理标签内部。若是，则生成全 1 的 bitmask（允许所有 token），否则正常生成约束 bitmask。这样既保证了推理过程的自由度，又确保了最终答案的结构化。

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--grammar-backend` | xgrammar | grammar 后端选择 |
| `--guided-decoding-backend` | xgrammar | 别名，同上 |
| `--disable-grammar-deciding` | False | 禁用结构化输出 |

`vllm/v1/structured_output/__init__.py`
