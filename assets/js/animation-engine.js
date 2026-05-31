window.AnimationEngine = (function () {
  function createStepPlayer({ steps, onAction, onReset }) {
    let index = -1;
    return {
      next() {
        if (index + 1 >= steps.length) return null;
        index += 1;
        const step = steps[index];
        step.actions.forEach(onAction);
        return step;
      },
      reset() {
        index = -1;
        if (onReset) onReset();
      },
      current() {
        return steps[index] || null;
      },
    };
  }
  return { createStepPlayer };
})();
