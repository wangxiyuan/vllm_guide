window.Diagrams = (function () {
  function renderPipeline(container, nodes) {
    container.innerHTML = nodes.map((node) => `<span>${node}</span>`).join("");
  }
  return { renderPipeline };
})();
