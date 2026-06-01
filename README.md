# vLLM Visual Guide 站点框架

这个目录是 vLLM 技术可视化网站的**纯静态**项目骨架。没有构建步骤、没有运行时服务，可以直接部署到 GitHub Pages 等静态托管。

## 本地预览

直接双击 `index.html` 即可在浏览器打开。所有数据都已内联到 `assets/js/site-data.js`，无需 HTTP 服务。

如果你愿意，也可以起一个本地服务：

```bash
python -m http.server 8000 --directory .
```

## 部署到 GitHub Pages

1. 把 `guide/` 推到 GitHub 仓库。
2. 在 Settings → Pages 选择仓库的对应分支与目录（可选 `/guide`）。
3. Pages 会直接以静态站点的方式提供服务，无需任何构建命令。

## 目录结构

```
guide/
├── index.html
├── design.md
├── README.md
├── pages/
│   ├── core/                # 架构、调度等核心专题
│   ├── distributed/         # 进程通信、并行、KV 传输
│   ├── decoding/            # 投机解码、采样等解码专题
│   ├── optimizations/       # KV cache、prefix caching、量化等性能专题
│   ├── models/              # 各模型结构专题
│   └── reference/           # 术语表、填充指南等参考文档
├── scripts/
│   └── bundle-data.py       # 把 assets/data/*.json 打包到 assets/js/site-data.js
└── assets/
    ├── css/                 # 样式
    ├── js/                  # site-data.js、home.js 等
    ├── data/                # topics、learning-paths、code-map 等数据（编辑源）
    ├── templates/           # 复制即可创建的新页面模板
    └── images/
```

## 数据更新

数据的「单一信源」仍然是 `assets/data/*.json`。修改后请重新生成 `site-data.js`：

```bash
python scripts/bundle-data.py
```

## 新增专题

1. 复制 `assets/templates/topic-page.html`，放入合适的 `pages/<分类>/` 子目录。
2. 在 `assets/data/topics.json` 添加专题元信息。
3. 在 `assets/data/code-map.json` 添加源码索引（可选）。
4. 运行 `python scripts/bundle-data.py` 刷新数据。
5. 在专题页填充内容，并把状态从 `todo / outline` 升级到 `draft / ready`。

详细规则见 `pages/reference/contribution-guide.html`。
