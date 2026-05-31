function main() {
  var data = window.SITE_DATA;
  if (!data) return;
  var container = document.getElementById("topics");
  var emptyEl = document.getElementById("topic-empty");
  var searchEl = document.getElementById("topic-search");
  var filterButtons = Array.prototype.slice.call(document.querySelectorAll(".filter-chip"));
  if (!container) return;

  var currentFilter = "all";
  var currentQuery = "";
  var categoryOrder = ["core", "distributed", "decoding", "optimization", "model", "reference"];
  var categoryLabels = {
    core: "核心引擎",
    distributed: "分布式策略",
    decoding: "解码优化",
    optimization: "性能优化",
    model: "模型结构",
    reference: "参考"
  };

  var categoryColors = {
    core: "#60a5fa",
    distributed: "#a78bfa",
    decoding: "#34d399",
    optimization: "#fbbf24",
    model: "#f87171",
    reference: "#94a3b8"
  };

  var categoryIcons = {
    core: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
    distributed: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="6" height="6" rx="1"/><rect x="16" y="2" width="6" height="6" rx="1"/><rect x="9" y="16" width="6" height="6" rx="1"/><path d="M5 8v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8"/><path d="M12 13v3"/></svg>',
    decoding: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    optimization: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></svg>',
    model: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
    reference: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>'
  };

  var activeStatuses = ["draft", "ready", "advanced"];
  var outlineStatuses = ["outline"];

  var grouped = {};
  data.topics.forEach(function (topic) {
    var cat = topic.category || "other";
    if (!grouped[cat]) grouped[cat] = [];
    grouped[cat].push(topic);
  });

  var totalCount = data.topics.length;
  var activeCount = data.topics.filter(function (t) { return activeStatuses.indexOf(t.status) >= 0; }).length;
  var outlineCount = data.topics.filter(function (t) { return outlineStatuses.indexOf(t.status) >= 0; }).length;

  var statsEl = document.getElementById("hero-stats");
  if (statsEl) {
    statsEl.innerHTML =
      '<div><span class="hero-stat-num">' + activeCount + '</span>专题已写好</div>' +
      '<div><span class="hero-stat-num">' + outlineCount + '</span>大纲已就绪</div>' +
      '<div><span class="hero-stat-num">' + (totalCount - activeCount - outlineCount) + '</span>规划中</div>';
  }

  function escapeHtml(value) {
    return String(value || "").replace(/[&<>"]/g, function (char) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[char];
    });
  }

  function getStatusGroup(status) {
    if (activeStatuses.indexOf(status) >= 0) return "ready";
    if (outlineStatuses.indexOf(status) >= 0) return "outline";
    return "todo";
  }

  function topicMatches(topic) {
    var statusGroup = getStatusGroup(topic.status);
    var queryText = [topic.title, topic.subtitle, topic.category, topic.level].concat(topic.tags || []).join(" ").toLowerCase();
    return (currentFilter === "all" || currentFilter === statusGroup) && (!currentQuery || queryText.indexOf(currentQuery) >= 0);
  }

  function renderTopics() {
    var html = "";
    var visibleCount = 0;
    categoryOrder.forEach(function (cat) {
      var topics = (grouped[cat] || []).filter(topicMatches);
      if (!topics.length) return;
      visibleCount += topics.length;
      var color = categoryColors[cat] || "#60a5fa";
      var activeInGroup = topics.filter(function (t) { return activeStatuses.indexOf(t.status) >= 0; }).length;
      var isCollapsed = cat !== "core" && !currentQuery && currentFilter === "all";
      var bodyId = "topic-group-" + cat;

      html += '<div class="card-group' + (isCollapsed ? ' card-group-collapsed' : '') + '" data-category="' + cat + '">';
      html += '<button class="card-group-header" type="button" style="background:' + color + '12;border:1px solid ' + color + '25" aria-expanded="' + (!isCollapsed) + '" aria-controls="' + bodyId + '">';
      html += '<div class="card-group-left">';
      html += '<div class="card-group-icon" style="color:' + color + '">' + (categoryIcons[cat] || '') + '</div>';
      html += '<h2 class="card-group-title" style="color:' + color + '">' + (categoryLabels[cat] || cat) + '</h2>';
      html += '</div>';
      html += '<div class="card-group-right">';
      html += '<div class="card-group-count">' + activeInGroup + '/' + topics.length + ' 可阅读</div>';
      html += '<svg class="card-group-chevron" viewBox="0 0 24 24" fill="none" stroke="' + color + '" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>';
      html += '</div>';
      html += '</button>';

      html += '<div class="card-group-body" id="' + bodyId + '">';
      topics.forEach(function (topic) {
        var isActive = activeStatuses.indexOf(topic.status) >= 0;
        var isOutline = outlineStatuses.indexOf(topic.status) >= 0;
        var cardClass = isActive ? "card-row" : (isOutline ? "card-row card-row-outline" : "card-row card-row-dim");
        html += '<a class="' + cardClass + '" href="' + escapeHtml(topic.href) + '">';
        html += '<div class="card-row-main">';
        html += '<div class="card-row-top">';
        html += '<span class="card-row-category" style="color:' + color + '">' + (categoryLabels[cat] || cat) + '</span>';
        if (topic.level) {
          var levelColors = { beginner: "#34d399", intermediate: "#60a5fa", advanced: "#a78bfa" };
          var levelLabels = { beginner: "入门", intermediate: "进阶", advanced: "深入" };
          html += '<span class="card-row-level" style="color:' + (levelColors[topic.level] || "#94a3b8") + '">' + (levelLabels[topic.level] || topic.level) + '</span>';
        }
        html += '</div>';
        html += '<h2>' + escapeHtml(topic.title) + '</h2>';
        html += '<p>' + escapeHtml(topic.subtitle) + '</p>';
        if (topic.tags && topic.tags.length > 0) {
          html += '<div class="card-row-tags">';
          topic.tags.forEach(function (tag) {
            html += '<span class="card-row-tag">' + escapeHtml(tag) + '</span>';
          });
          html += '</div>';
        }
        html += '</div>';
        html += '<div class="card-row-side">';
        html += '<span class="status status-' + topic.status + '">' + topic.status + '</span>';
        html += '<span class="muted">' + topic.readingMinutes + ' min</span>';
        html += '</div>';
        html += '</a>';
      });
      html += '</div>';
      html += '</div>';
    });

    container.innerHTML = html;
    if (emptyEl) emptyEl.hidden = visibleCount > 0;
  }

  container.addEventListener("click", function (event) {
    var header = event.target.closest(".card-group-header");
    if (!header) return;
    var group = header.parentElement;
    var isCollapsed = group.classList.toggle("card-group-collapsed");
    header.setAttribute("aria-expanded", String(!isCollapsed));
  });

  if (searchEl) {
    searchEl.addEventListener("input", function () {
      currentQuery = searchEl.value.trim().toLowerCase();
      renderTopics();
    });
  }

  filterButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      currentFilter = button.getAttribute("data-filter") || "all";
      filterButtons.forEach(function (item) { item.classList.toggle("is-active", item === button); });
      renderTopics();
    });
  });

  renderTopics();
}

main();
