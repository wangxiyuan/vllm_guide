(() => {
  const body = document.body;
  body.dataset.builtAt = new Date().toISOString();

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
})();
