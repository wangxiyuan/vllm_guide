(() => {
  const body = document.body;
  body.dataset.builtAt = new Date().toISOString();

  const mdToggle = document.getElementById("md-toggle");
  const topicContent = document.getElementById("topic-content");
  const mdView = document.getElementById("topic-md-view");
  if (mdToggle && topicContent && mdView) {
    mdToggle.addEventListener("click", () => {
      const isActive = mdToggle.classList.toggle("is-active");
      topicContent.hidden = isActive;
      mdView.hidden = !isActive;
      mdToggle.textContent = isActive ? "HTML" : "MD";
    });
  }

  const main = document.querySelector("main");
  if (main && !document.getElementById("back-to-top")) {
    const button = document.createElement("button");
    button.id = "back-to-top";
    button.type = "button";
    button.textContent = "↑";
    button.setAttribute("aria-label", "回到顶部");
    button.style.cssText = "position:fixed;right:24px;bottom:24px;border:1px solid var(--border);background:var(--panel-2);color:var(--text);border-radius:999px;width:42px;height:42px;cursor:pointer;display:none;";
    button.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
    document.body.appendChild(button);
    window.addEventListener("scroll", () => {
      button.style.display = window.scrollY > 360 ? "block" : "none";
    });
  }

  document.querySelectorAll("pre").forEach((pre) => {
    if (pre.parentElement && pre.parentElement.classList.contains("code-copy-wrap")) return;
    const wrap = document.createElement("div");
    wrap.className = "code-copy-wrap";
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "code-copy-btn";
    copy.textContent = "复制";
    copy.addEventListener("click", async () => {
      const code = pre.textContent || "";
      if (navigator.clipboard) await navigator.clipboard.writeText(code);
      copy.textContent = "已复制";
      window.setTimeout(() => { copy.textContent = "复制"; }, 1200);
    });
    wrap.appendChild(copy);
  });

  const flowModels = {
    architecture: {
      title: "一次请求的数据流转",
      description: "把请求看成一个包裹：API 收件、EngineCore 调度、Worker 计算、结果再回到客户端。",
      nodes: ["HTTP", "AsyncLLM", "ZMQ", "EngineCore", "Scheduler", "Worker", "Output"],
      steps: [
        { from: 0, to: 1, title: "请求进入 API", text: "OpenAI API 收到 prompt，交给 AsyncLLM 做输入处理。" },
        { from: 1, to: 2, title: "跨进程投递", text: "EngineCoreRequest 通过 ZMQ ROUTER/DEALER 发送到 EngineCore。" },
        { from: 2, to: 3, title: "进入核心循环", text: "input_thread 反序列化请求并放入 input_queue。" },
        { from: 3, to: 4, title: "调度本轮 token", text: "Scheduler 选择 running/waiting 请求并分配 token budget。" },
        { from: 4, to: 5, title: "Worker 执行模型", text: "Executor 把 SchedulerOutput 广播给 Worker，GPU 执行 forward。" },
        { from: 5, to: 6, title: "结果返回", text: "ModelRunnerOutput 经 update_from_output 转成 EngineCoreOutputs 返回 API。" }
      ]
    },
    scheduler: {
      title: "调度器状态流转",
      description: "关注请求如何从 waiting 进入 running，遇到资源不足时如何 preempt，再如何完成。",
      nodes: ["waiting", "prefix cache", "allocate slots", "running", "execute", "update", "finished/preempted"],
      steps: [
        { from: 0, to: 1, title: "检查等待队列", text: "从 waiting 取请求，先判断 grammar、远端 KV、前缀缓存状态。" },
        { from: 1, to: 2, title: "计算新 token", text: "根据 num_tokens_with_spec - num_computed_tokens 得到本轮需求。" },
        { from: 2, to: 3, title: "分配 KV blocks", text: "allocate_slots 成功后，请求加入 running；失败则停止或触发抢占。" },
        { from: 3, to: 4, title: "执行本轮 batch", text: "running 请求被打包成 SchedulerOutput 交给模型执行。" },
        { from: 4, to: 5, title: "更新计数器", text: "模型输出后推进 num_computed_tokens，并检查 stop 条件。" },
        { from: 5, to: 6, title: "完成或回队", text: "完成则释放资源；被抢占则回到 waiting。" }
      ]
    },
    "kv-cache": {
      title: "KV Cache 生命周期",
      description: "把显存看成 block 池：申请、写入、复用、引用计数、释放和驱逐。",
      nodes: ["free queue", "allocate", "block table", "write KV", "prefix hit", "release", "evict"],
      steps: [
        { from: 0, to: 1, title: "从空闲队列取 block", text: "BlockPool 按 LRU 顺序拿出可用物理 block。" },
        { from: 1, to: 2, title: "写入 Block Table", text: "请求的逻辑 block 映射到具体物理 block id。" },
        { from: 2, to: 3, title: "写入 K/V 向量", text: "新 token 的 K/V 根据 slot_mapping 写入 block 内部位置。" },
        { from: 3, to: 4, title: "前缀复用", text: "相同 hash 的 block 可被多个请求共享并增加引用计数。" },
        { from: 4, to: 5, title: "请求结束释放", text: "引用计数归零后，block 回到 free queue，但 hash 可保留。" },
        { from: 5, to: 6, title: "显存紧张驱逐", text: "需要空间时，最久未用的缓存 block 被清除 hash 后复用。" }
      ]
    },
    "process-communication": {
      title: "通信流转分层图",
      description: "同一轮推理同时经过 API↔EngineCore、EngineCore↔Worker、GPU↔GPU 三层通信。",
      nodes: ["API", "ZMQ", "EngineCore", "ShmRingBuffer", "Worker", "NCCL", "GPU ranks"],
      steps: [
        { from: 0, to: 1, title: "API 发请求", text: "EngineCoreRequest 使用 msgpack 多帧经 ZMQ 发送。" },
        { from: 1, to: 2, title: "EngineCore 接收", text: "input_thread 接收消息，主线程 busy loop 消费 input_queue。" },
        { from: 2, to: 3, title: "广播调度输出", text: "SchedulerOutput 通过共享内存环形缓冲区发给 Worker。" },
        { from: 3, to: 4, title: "Worker 执行", text: "Worker 从共享内存读取命令并准备 GPU 输入。" },
        { from: 4, to: 5, title: "集合通信", text: "TP/SP/CP 等策略通过 NCCL all_reduce/all_gather/reduce_scatter 协作。" },
        { from: 5, to: 6, title: "多卡完成", text: "各 GPU rank 完成本轮计算并把结果回传给 EngineCore。" }
      ]
    },
    "paged-attention": {
      title: "PagedAttention 读写路径",
      description: "分清两条线：slot_mapping 负责写新 KV，block_table 负责读历史 KV。",
      nodes: ["SchedulerOutput", "block table", "slot mapping", "write KV", "metadata", "attention", "logits"],
      steps: [
        { from: 0, to: 1, title: "拿到物理块映射", text: "Scheduler/KV manager 为请求准备 block table。" },
        { from: 1, to: 2, title: "生成写入地址", text: "ModelRunner 根据 token 位置生成 slot_mapping。" },
        { from: 2, to: 3, title: "Scatter 写 KV", text: "新算出的 K/V 被写入不连续的物理 block。" },
        { from: 3, to: 4, title: "构建 metadata", text: "query_start_loc、seq_lens、block_table 被打包给 backend。" },
        { from: 4, to: 5, title: "跳读历史 KV", text: "Attention kernel 按 block table 读取历史 K/V。" },
        { from: 5, to: 6, title: "输出 logits", text: "Attention 输出进入后续 MLP / lm_head，得到下一个 token 分数。" }
      ]
    }
  };

  function mountMentalFlow() {
    const pageId = document.body.dataset.pageId;
    const model = flowModels[pageId];
    if (!model) return;
    const hero = document.querySelector(".topic-hero");
    if (!hero || document.querySelector(".mental-flow-panel")) return;
    const panel = document.createElement("section");
    panel.className = "panel mental-flow-panel";
    panel.innerHTML = [
      '<div class="mental-flow-head"><div><p class="eyebrow">Mental Model</p><h2>' + model.title + '</h2><p class="muted">' + model.description + '</p></div><div class="mental-flow-actions"><button type="button" data-flow-prev>上一步</button><button type="button" data-flow-next>下一步</button><button type="button" data-flow-reset>重置</button></div></div>',
      '<div class="mental-flow-track">' + model.nodes.map((node, index) => '<div class="mental-flow-node" data-flow-node="' + index + '">' + node + '</div>').join('') + '</div>',
      '<div class="mental-flow-status"><strong data-flow-title>准备开始</strong><span data-flow-progress>0 / ' + model.steps.length + '</span><p data-flow-text>点击下一步，观察数据或通信如何在模块间流转。</p></div>'
    ].join("");
    hero.insertAdjacentElement("afterend", panel);
    const nodes = Array.prototype.slice.call(panel.querySelectorAll("[data-flow-node]"));
    let index = -1;
    const title = panel.querySelector("[data-flow-title]");
    const text = panel.querySelector("[data-flow-text]");
    const progress = panel.querySelector("[data-flow-progress]");
    const prev = panel.querySelector("[data-flow-prev]");
    const next = panel.querySelector("[data-flow-next]");
    const reset = panel.querySelector("[data-flow-reset]");

    function render() {
      nodes.forEach((node) => node.classList.remove("is-active", "is-source", "is-target"));
      if (index >= 0) {
        const step = model.steps[index];
        nodes[step.from].classList.add("is-source");
        nodes[step.to].classList.add("is-target", "is-active");
        title.textContent = step.title;
        text.textContent = step.text;
      } else {
        title.textContent = "准备开始";
        text.textContent = "点击下一步，观察数据或通信如何在模块间流转。";
      }
      progress.textContent = (index + 1 < 0 ? 0 : index + 1) + " / " + model.steps.length;
      prev.disabled = index <= 0;
      next.disabled = index >= model.steps.length - 1;
    }

    prev.addEventListener("click", () => { if (index > 0) index -= 1; render(); });
    next.addEventListener("click", () => { if (index < model.steps.length - 1) index += 1; render(); });
    reset.addEventListener("click", () => { index = -1; render(); });
    render();
  }

  mountMentalFlow();

  const termDefs = {
    prefill: "一次性处理输入 prompt，生成首个 token 前的 KV Cache。",
    decode: "逐 token 生成阶段，每轮复用已有 KV Cache。",
    scheduler: "决定每轮哪些请求进入 batch、分配多少 token budget 的模块。",
    "kv cache": "保存历史 token 的 Key/Value 张量，避免重复计算。",
    pagedattention: "用块表把逻辑 token 映射到物理 KV block，降低显存碎片。",
    "block table": "记录每个请求逻辑块到物理 KV block 的映射表。",
    preemption: "资源不足时暂停或回收低优先级请求，稍后继续执行。"
  };

  document.querySelectorAll("[data-term]").forEach((term) => {
    const key = String(term.dataset.term || term.textContent || "").trim().toLowerCase();
    const text = term.dataset.tip || termDefs[key];
    if (!text) return;
    term.classList.add("term-tip");
    term.tabIndex = 0;
    term.setAttribute("role", "term");
    term.setAttribute("aria-label", text);
    term.setAttribute("data-tip", text);
  });

  function createSim(root, steps, renderFn) {
    if (!root) return null;
    let idx = -1;
    const title = root.querySelector("[data-sim-title]");
    const text = root.querySelector("[data-sim-text]");
    const progress = root.querySelector("[data-sim-progress]");
    const prev = root.querySelector("[data-sim-prev]");
    const next = root.querySelector("[data-sim-next]");
    const reset = root.querySelector("[data-sim-reset]");

    function render() {
      const step = idx >= 0 ? steps[idx] : null;
      renderFn(step, idx);
      if (title) title.textContent = step ? step.title : "初始状态";
      if (text) text.textContent = step ? step.text : steps.initText || "点击下一步开始模拟。";
      if (progress) progress.textContent = (idx + 1 < 0 ? 0 : idx + 1) + " / " + steps.length;
      if (prev) prev.disabled = idx <= 0;
      if (next) next.disabled = idx >= steps.length - 1;
    }

    if (prev) prev.addEventListener("click", () => { if (idx > 0) idx -= 1; render(); });
    if (next) next.addEventListener("click", () => { if (idx < steps.length - 1) idx += 1; render(); });
    if (reset) reset.addEventListener("click", () => { idx = -1; render(); });
    render();
    return { render, reset: () => { idx = -1; render(); } };
  }

  const schedSim = document.querySelector('[data-sim="scheduler"]');
  if (schedSim) {
    const wEl = schedSim.querySelector("#sim-waiting");
    const rEl = schedSim.querySelector("#sim-running");
    const fEl = schedSim.querySelector("#sim-finished");
    const qEls = schedSim.querySelectorAll(".sim-queue");

    function chip(req, isNew) {
      return '<span class="sim-req-chip' + (isNew ? ' is-new' : '') + '" data-req="' + req + '">' + req + '</span>';
    }
    function renderQueues(waiting, running, finished, activeQueue) {
      if (wEl) wEl.innerHTML = waiting.map((r) => chip(r, false)).join("") || '<span class="muted" style="font-size:12px">空</span>';
      if (rEl) rEl.innerHTML = running.map((r) => chip(r, false)).join("") || '<span class="muted" style="font-size:12px">空</span>';
      if (fEl) fEl.innerHTML = finished.map((r) => chip(r, false)).join("") || '<span class="muted" style="font-size:12px">空</span>';
      qEls.forEach((el) => el.classList.toggle("is-active", el.dataset.simQueue === activeQueue));
    }

    createSim(schedSim, [
      { title: "请求到达", text: "R1、R2、R3 进入 waiting 队列（FIFO 顺序）。", waiting: ["R1", "R2", "R3"], running: [], finished: [], activeQueue: "waiting" },
      { title: "调度 R1、R2", text: "schedule() Phase 2 从 waiting 取出 R1、R2，分配 KV blocks 后加入 running。", waiting: ["R3"], running: ["R1", "R2"], finished: [], activeQueue: "running" },
      { title: "执行并更新", text: "R1、R2 在 GPU 上执行 forward。update_from_output() 推进 num_computed_tokens。", waiting: ["R3"], running: ["R1", "R2"], finished: [], activeQueue: "running" },
      { title: "R1 完成", text: "R1 达到 max_tokens，状态变为 FINISHED，释放 KV blocks。R3 进入 running。", waiting: [], running: ["R2", "R3"], finished: ["R1"], activeQueue: "finished" },
      { title: "R2 被抢占", text: "显存不足，allocate_slots() 对 R2 失败。R2 被抢占：num_computed_tokens 归零，释放全部 KV blocks，prepend 到 waiting 队列头部。R3 仍在 running。", waiting: ["R2"], running: ["R3"], finished: ["R1"], activeQueue: "waiting" },
      { title: "R2 重新调度", text: "下一轮 schedule()：R3 继续执行，R2 从 waiting 头部被重新调度（prefix cache 可能恢复部分 KV blocks）。", waiting: [], running: ["R3", "R2"], finished: ["R1"], activeQueue: "running" }
    ], (step) => {
      if (step) renderQueues(step.waiting, step.running, step.finished, step.activeQueue);
      else renderQueues(["R1", "R2", "R3"], [], [], "waiting");
    });
  }

  const kvSim = document.querySelector('[data-sim="kv-cache"]');
  if (kvSim) {
    const poolEl = kvSim.querySelector("#sim-kv-pool");
    const tablesEl = kvSim.querySelector("#sim-kv-tables");

    function renderKV(blocks, tables, highlightBlocks) {
      if (poolEl) {
        poolEl.innerHTML = blocks.map((b) =>
          '<div class="sim-kv-block is-' + b.state + (highlightBlocks && highlightBlocks.indexOf(b.id) >= 0 ? ' is-highlight' : '') + '">' + b.id + '</div>'
        ).join("");
      }
      if (tablesEl) {
        tablesEl.innerHTML = tables.map((t) =>
          '<div class="sim-kv-table"><div class="sim-kv-table-title">' + t.name + '</div><div class="sim-kv-table-row">' +
          t.blocks.map((c) => '<span class="sim-kv-table-cell is-' + (c.shared ? 'shared' : 'unique') + '">' + c.id + '</span>').join("") +
          '</div></div>'
        ).join("");
      }
    }

    const allBlocks = [
      { id: "B0" }, { id: "B1" }, { id: "B2" }, { id: "B3" },
      { id: "B4" }, { id: "B5" }, { id: "B6" }, { id: "B7" }
    ];

    createSim(kvSim, [
      { title: "初始状态", text: "8 个 block 全部空闲。", blocks: allBlocks.map((b) => ({ ...b, state: "free" })), tables: [], highlight: [] },
      { title: "请求 A 分配 B0、B1", text: "请求 A 需要 2 个 block，从 free queue 取出 B0、B1。ref_cnt 各为 1。", blocks: allBlocks.map((b, i) => ({ ...b, state: i < 2 ? "used" : "free" })), tables: [{ name: "请求 A Block Table", blocks: [{ id: "B0", shared: false }, { id: "B1", shared: false }] }], highlight: ["B0", "B1"] },
      { title: "请求 A 写入 KV", text: "新 token 的 K/V 通过 slot_mapping 写入 B0、B1。block 填满后计算 hash 并缓存。", blocks: allBlocks.map((b, i) => ({ ...b, state: i < 2 ? "used" : "free" })), tables: [{ name: "请求 A Block Table", blocks: [{ id: "B0", shared: false }, { id: "B1", shared: false }] }], highlight: ["B0", "B1"] },
      { title: "请求 B 复用前缀 B0", text: "请求 B 与 A 有相同前缀，find_longest_cache_hit() 命中 B0。touch() 将 B0 的 ref_cnt 从 1 增至 2，B0 仍为 used。另分配 B2。", blocks: allBlocks.map((b, i) => ({ ...b, state: i < 3 ? "used" : "free" })), tables: [{ name: "请求 A", blocks: [{ id: "B0", shared: true }, { id: "B1", shared: false }] }, { name: "请求 B", blocks: [{ id: "B0", shared: true }, { id: "B2", shared: false }] }], highlight: ["B0", "B2"] },
      { title: "请求 A 结束释放", text: "A 释放 B0（ref_cnt 2→1，仍被 B 使用，保持 used）和 B1（ref_cnt 1→0，变为 cached，hash 保留可被复用）。", blocks: allBlocks.map((b, i) => ({ ...b, state: (i === 0 || i === 2) ? "used" : (i === 1 ? "cached" : "free") })), tables: [{ name: "请求 B", blocks: [{ id: "B0", shared: false }, { id: "B2", shared: false }] }], highlight: ["B0", "B1"] },
      { title: "请求 B 结束，B0 变 cached", text: "B 释放 B0（ref_cnt 1→0，hash 保留，变为 cached）和 B2（ref_cnt 1→0，变为 cached）。显存紧张时，LRU 驱逐会先清 B1 的 hash 再清 B0。", blocks: allBlocks.map((b, i) => ({ ...b, state: (i <= 2) ? "cached" : "free" })), tables: [], highlight: ["B0", "B2"] }
    ], (step) => {
      if (step) renderKV(step.blocks, step.tables, step.highlight);
      else renderKV(allBlocks.map((b) => ({ ...b, state: "free" })), [], []);
    });
  }

  const commSim = document.querySelector('[data-sim="comm"]');
  if (commSim) {
    const layers = commSim.querySelectorAll(".sim-comm-layer");
    const nodes = commSim.querySelectorAll(".sim-comm-node");

    createSim(commSim, [
      { title: "API 发送请求", text: "请求通过 ZMQ ROUTER→DEALER 跨进程发送到 EngineCore。", layer: "cross-proc", activeNodes: [0, 1] },
      { title: "input_thread 接收", text: "EngineCore 的 input_thread 从 ZMQ 读取，放入 input_queue。", layer: "intra-proc", activeNodes: [1, 3] },
      { title: "主线程调度执行", text: "主线程 busy loop 从 input_queue 取请求，调度后通过 ShmRingBuffer 发给 Worker。", layer: "intra-proc", activeNodes: [3, 4] },
      { title: "Worker 收到调度", text: "Worker 从共享内存读取 SchedulerOutput，准备 GPU 输入。", layer: "cross-proc", activeNodes: [1, 2] },
      { title: "NCCL 集合通信", text: "TP/SP/CP 策略通过 NCCL 在 GPU 间传递中间张量。", layer: "cross-gpu", activeNodes: [6, 7, 8] },
      { title: "结果回传 API", text: "ModelRunnerOutput 经 ShmRingBuffer 回到 EngineCore，再经 ZMQ PUSH→PULL 返回 API。", layer: "cross-proc", activeNodes: [2, 1, 0] }
    ], (step) => {
      layers.forEach((l) => l.classList.toggle("is-active", step && l.dataset.comm === step.layer));
      nodes.forEach((n, i) => n.classList.toggle("is-active", step && step.activeNodes.indexOf(i) >= 0));
    });
  }

  const paSim = document.querySelector('[data-sim="pa-rw"]');
  if (paSim) {
    const paths = paSim.querySelectorAll(".sim-pa-path");
    const paNodes = paSim.querySelectorAll(".sim-pa-node");

    createSim(paSim, [
      { title: "调度器分配 block", text: "Scheduler → KV Cache Manager 为请求分配物理 block，写入 block_table。", path: "write", activeNodes: [2] },
      { title: "生成 slot_mapping", text: "ModelRunner 根据 token 位置计算每个 token 的写入地址（block_id × block_size + slot_offset）。", path: "write", activeNodes: [0, 1] },
      { title: "Scatter 写入新 KV", text: "do_kv_cache_update() 用 slot_mapping 把新 K/V 散列写入物理 block 的对应位置。", path: "write", activeNodes: [1, 2] },
      { title: "构建 metadata", text: "打包 query_start_loc、seq_lens、block_table 给 Attention backend，准备读历史 KV。", path: "read", activeNodes: [4] },
      { title: "Paged read 历史 KV", text: "Attention kernel 按 block_table 从不连续的物理 block 中读取历史 K/V，计算注意力输出。", path: "read", activeNodes: [3, 4, 5] }
    ], (step) => {
      paths.forEach((p) => p.classList.toggle("is-active", step && p.dataset.paPath === step.path));
      paNodes.forEach((n, i) => n.classList.toggle("is-active", step && step.activeNodes.indexOf(i) >= 0));
    });
  }

  const lifecyclePlayer = document.getElementById("lifecycle-player");
  const lifecycleSteps = document.getElementById("lifecycle-steps");
  if (lifecyclePlayer && lifecycleSteps) {
    const stepItems = lifecycleSteps.querySelectorAll(".step-item");
    const total = stepItems.length;

    function renderLifecycle(idx) {
      stepItems.forEach((item, i) => {
        item.classList.toggle("step-item-active", i === idx);
        item.classList.toggle("step-item-done", i < idx);
      });
    }

    AnimationEngine.mountStepPlayer({
      root: lifecyclePlayer,
      steps: Array.from({ length: total }, (_, i) => {
        const body = stepItems[i].querySelector(".step-body");
        return {
          title: body ? body.querySelector("h3").textContent : "Step " + (i + 1),
          text: body ? body.querySelector("p").textContent : "",
          actions: [{ type: "highlight", index: i }]
        };
      }),
      onAction(action) {
        if (action.type === "highlight") renderLifecycle(action.index);
      },
      onReset() {
        renderLifecycle(-1);
      }
    });
  }
})();
