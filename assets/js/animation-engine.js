window.AnimationEngine = (function () {
  function createStepPlayer({ steps, onAction, onReset, onChange }) {
    let index = -1;

    function emit(step) {
      if (onChange) onChange({ step, index, total: steps.length });
    }

    return {
      next() {
        if (index + 1 >= steps.length) return null;
        index += 1;
        const step = steps[index];
        if (step.actions) step.actions.forEach(onAction);
        emit(step);
        return step;
      },
      prev() {
        if (index <= 0) return null;
        index -= 1;
        if (onReset) onReset();
        for (let i = 0; i <= index; i += 1) {
          if (steps[i].actions) steps[i].actions.forEach(onAction);
        }
        emit(steps[index]);
        return steps[index];
      },
      reset() {
        index = -1;
        if (onReset) onReset();
        emit(null);
      },
      current() {
        return steps[index] || null;
      },
      progress() {
        return { index, total: steps.length };
      },
    };
  }

  function mountStepPlayer({ root, steps, onAction, onReset }) {
    if (!root || !steps || !steps.length) return null;
    const title = root.querySelector("[data-step-title]");
    const text = root.querySelector("[data-step-text]");
    const progress = root.querySelector("[data-step-progress]");
    const prevBtn = root.querySelector("[data-step-prev]");
    const nextBtn = root.querySelector("[data-step-next]");
    const resetBtn = root.querySelector("[data-step-reset]");
    const player = createStepPlayer({
      steps,
      onAction,
      onReset,
      onChange({ step, index, total }) {
        if (title) title.textContent = step ? step.title || "Step " + (index + 1) : "准备开始";
        if (text) text.textContent = step ? step.text || step.description || "" : "点击下一步，逐帧理解流程。";
        if (progress) progress.textContent = index < 0 ? "0 / " + total : index + 1 + " / " + total;
        if (prevBtn) prevBtn.disabled = index <= 0;
        if (nextBtn) nextBtn.disabled = index + 1 >= total;
      },
    });

    if (prevBtn) prevBtn.addEventListener("click", player.prev);
    if (nextBtn) nextBtn.addEventListener("click", player.next);
    if (resetBtn) resetBtn.addEventListener("click", player.reset);
    player.reset();
    return player;
  }

  return { createStepPlayer, mountStepPlayer };
})();
