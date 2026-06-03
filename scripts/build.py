#!/usr/bin/env python3
"""Build HTML pages from Markdown content files.

Usage:
    python scripts/build.py                  # build all
    python scripts/build.py architecture     # build single page
    python scripts/build.py --watch          # rebuild on change (requires watchdog)

Reads:   content/<category>/<id>.md
Writes:  pages/<category>/<id>.html
"""

import html as html_module
import json
import os
import re
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = ROOT / "content"
PAGES_DIR = ROOT / "pages"
ASSETS_DIR = ROOT / "assets"
TEMPLATES_DIR = ASSETS_DIR / "templates"

md_parser = MarkdownIt("commonmark", {"html": True, "typographer": False}).enable("table")

jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    keep_trailing_newline=True,
    autoescape=False,
)


def parse_frontmatter(text):
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        return {}, text
    meta = yaml.safe_load(m.group(1)) or {}
    return meta, text[m.end():]


def process_term_links(text):
    def repl(m):
        term = m.group(1)
        tip = m.group(2)
        return f'<span data-term="{term}" data-tip="{tip}">{term}</span>'
    return re.sub(r"\[([^\]]+)\]\(term:([^)]+)\)", repl, text)


def extract_code_block(lines, start):
    i = start
    if i < len(lines) and lines[i].strip() == "```":
        lang = ""
        i += 1
    elif i < len(lines) and re.match(r"^```\w*$", lines[i].strip()):
        lang = lines[i].strip()[3:]
        i += 1
    else:
        return "", i
    code_lines = []
    while i < len(lines) and lines[i].strip() != "```":
        code_lines.append(lines[i])
        i += 1
    if i < len(lines):
        i += 1
    return "\n".join(code_lines), i


def render_diagram_html(code_block):
    content = code_block.strip()
    if content.startswith("html\n"):
        content = content[5:]
    return content


def render_diagram_block(diagram_type, code_block):
    if diagram_type == "arch-html":
        return render_diagram_html(code_block)
    if diagram_type == "arch":
        return (
            '<div class="arch-diagram-ascii"><pre>'
            + html_module.escape(code_block.strip())
            + '</pre></div>'
        )
    return f'<pre class="diagram-raw">{html_module.escape(code_block)}</pre>'


def process_custom_blocks(raw_body):
    result = []
    diagram_desc_placeholders = {}
    i = 0
    lines = raw_body.split("\n")
    n = len(lines)

    while i < n:
        line = lines[i]

        m = re.match(r"^:::diagram-desc\s+(\S+)\s*$", line)
        if m:
            desc_id = m.group(1)
            i += 1
            desc_lines = []
            while i < n and not re.match(r"^:::\s*$", lines[i]):
                desc_lines.append(lines[i])
                i += 1
            if i < n:
                i += 1
            desc_text = "\n".join(desc_lines).strip()
            desc_html = md_parser.render(desc_text)
            desc_html = unescape_html_entities(desc_html)
            desc_html = process_term_links(desc_html)
            placeholder = f"DIAGRAMDESCPLACEHOLDER{desc_id}ENDPLACEHOLDER"
            diagram_desc_placeholders[placeholder] = desc_html
            result.append(placeholder)
            continue

        m = re.match(r"^:::diagram\s+(\S+)\s*$", line)
        if m:
            diagram_type = m.group(1)
            i += 1
            code_block, i = extract_code_block(lines, i)
            if i < n and not re.match(r"^:::\s*$", lines[i]):
                pass
            while i < n and not re.match(r"^:::\s*$", lines[i]):
                i += 1
            if i < n:
                i += 1
            result.append(render_diagram_block(diagram_type, code_block))
            continue

        m = re.match(r"^:::steps\s+id=(\S+)\s*$", line)
        if m:
            steps_id = m.group(1)
            i += 1
            steps = []
            current_step = None
            while i < n and not re.match(r"^:::\s*$", lines[i]):
                sm = re.match(r"^###\s+(\d+\.\s+.+)$", lines[i])
                if sm:
                    if current_step:
                        steps.append(current_step)
                    title_text = sm.group(1)
                    num_m = re.match(r"(\d+)\.\s+(.*)", title_text)
                    if num_m:
                        current_step = {
                            "num": int(num_m.group(1)),
                            "title": num_m.group(2),
                            "body_lines": [],
                        }
                    else:
                        current_step = {
                            "num": len(steps) + 1,
                            "title": title_text,
                            "body_lines": [],
                        }
                elif current_step is not None:
                    current_step["body_lines"].append(lines[i])
                i += 1
            if current_step:
                steps.append(current_step)
            if i < n:
                i += 1
            result.append(render_steps_block(steps_id, steps))
            continue

        m = re.match(r"^:::(\w+)", line)
        if m and not re.match(r"^:::\s*$", line):
            block_type = m.group(1)
            i += 1
            block_lines = []
            while i < n and not re.match(r"^:::\s*$", lines[i]):
                block_lines.append(lines[i])
                i += 1
            if i < n:
                i += 1
            result.append(
                f'<div class="custom-block custom-block-{block_type}">'
                + md_parser.render("\n".join(block_lines))
                + "</div>"
            )
            continue

        result.append(line)
        i += 1

    rendered = md_parser.render("\n".join(result))

    for placeholder, desc_html in diagram_desc_placeholders.items():
        rendered = rendered.replace(
            f"<p>{placeholder}</p>",
            f'<template class="diagram-desc" data-diagram-desc="{placeholder[len("DIAGRAMDESCPLACEHOLDER"):placeholder.index("ENDPLACEHOLDER")]}">{desc_html}</template>',
        )
        rendered = rendered.replace(
            placeholder,
            f'<template class="diagram-desc" data-diagram-desc="{placeholder[len("DIAGRAMDESCPLACEHOLDER"):placeholder.index("ENDPLACEHOLDER")]}">{desc_html}</template>',
        )

    return rendered


def render_step_ref(md_text):
    lines = md_text.strip().split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        m = re.match(r"^`([^`]+)`$", stripped)
        if m and ("/" in m.group(1) or m.group(1).startswith("vllm")):
            result.append(f'<code class="step-ref">{m.group(1)}</code>')
            continue
        result.append(line)
    return "\n".join(result)


def render_steps_block(steps_id, steps):
    total = len(steps)
    step_items = []
    for s in steps:
        body_md = render_step_ref("\n".join(s["body_lines"]))
        body_html = md_parser.render(body_md)
        body_html = process_term_links(body_html)
        body_html = re.sub(r"<p>(.*?)</p>", r"\1", body_html, flags=re.DOTALL)
        step_items.append(
            f'<div class="step-item" id="step-{s["num"]}">'
            f'<div class="step-num">{s["num"]}</div>'
            f'<div class="step-body">'
            f'<h3>{s["title"]}</h3>'
            f"{body_html}"
            f"</div></div>"
        )

    player_html = (
        f'<div class="step-player" id="{steps_id}">'
        '<div class="step-player-head">'
        '<h3 class="step-player-title" data-step-title>准备开始</h3>'
        f'<span class="step-player-progress" data-step-progress>0 / {total}</span>'
        "</div>"
        f'<p class="step-player-text" data-step-text>点击下一步，逐帧追踪流程。</p>'
        '<div class="step-player-actions">'
        '<button type="button" data-step-prev disabled>上一步</button>'
        '<button type="button" data-step-next>下一步</button>'
        '<button type="button" data-step-reset>重置</button>'
        "</div></div>"
    )

    list_html = (
        f'<div class="step-list" id="{steps_id}-steps">'
        + "\n".join(step_items)
        + "</div>"
    )

    return player_html + "\n" + list_html


def unescape_html_entities(text):
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    text = text.replace("&amp;", "&")
    text = text.replace("&quot;", '"')
    return text


def strip_custom_blocks_for_md(md_text):
    lines = md_text.split("\n")
    result = []
    i = 0
    n = len(lines)
    while i < n:
        m = re.match(r"^:::diagram\s+arch-html\s*$", lines[i])
        if m:
            i += 1
            if i < n and re.match(r"^```", lines[i].strip()):
                i += 1
                while i < n and lines[i].strip() != "```":
                    i += 1
                if i < n:
                    i += 1
            while i < n and not re.match(r"^:::\s*$", lines[i]):
                i += 1
            if i < n:
                i += 1
            continue

        m = re.match(r"^:::diagram-desc\s+\S+\s*$", lines[i])
        if m:
            i += 1
            while i < n and not re.match(r"^:::\s*$", lines[i]):
                result.append(lines[i])
                i += 1
            if i < n:
                i += 1
            continue

        m = re.match(r"^:::steps\s+id=\S+\s*$", lines[i])
        if m:
            i += 1
            while i < n and not re.match(r"^:::\s*$", lines[i]):
                result.append(lines[i])
                i += 1
            if i < n:
                i += 1
            continue

        m = re.match(r"^:::\w+", lines[i])
        if m and not re.match(r"^:::\s*$", lines[i]):
            i += 1
            while i < n and not re.match(r"^:::\s*$", lines[i]):
                result.append(lines[i])
                i += 1
            if i < n:
                i += 1
            continue

        result.append(lines[i])
        i += 1
    return "\n".join(result)


def build_page(md_path):
    rel = md_path.relative_to(CONTENT_DIR)
    category = rel.parts[0]
    stem = rel.stem

    raw = md_path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)

    page_id = meta.get("id", stem)
    title = meta.get("title", stem)
    category_meta = meta.get("category", category)
    level = meta.get("level", "beginner")
    status = meta.get("status", "todo")
    reading_minutes = meta.get("readingMinutes", 10)
    tags = meta.get("tags", [])
    hero_text_md = meta.get("heroText", "")
    code_refs = meta.get("codeRefs", [])

    body_html = process_custom_blocks(body)
    body_html = unescape_html_entities(body_html)
    body_html = process_term_links(body_html)

    hero_text_html = md_parser.renderInline(hero_text_md) if hero_text_md else ""
    hero_text_html = process_term_links(hero_text_html)

    tags_html = "".join(f'<span class="tag">{t}</span>' for t in tags)

    sections = split_sections(body_html)

    md_source = body.rstrip("\n")
    md_clean = strip_custom_blocks_for_md(md_source)
    md_clean_html = md_parser.render(md_clean)
    md_clean_html = unescape_html_entities(md_clean_html)
    md_clean_html = process_term_links(md_clean_html)

    out_dir = PAGES_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.html"

    depth = len(rel.parts) - 1
    prefix = "../" * (depth + 1)

    template = jinja_env.get_template("topic-page.html")
    html = template.render(
        page_id=page_id,
        title=title,
        category=category_meta,
        level=level,
        status=status,
        reading_minutes=reading_minutes,
        tags_html=tags_html,
        hero_text_html=hero_text_html,
        sections=sections,
        prefix=prefix,
        code_refs=code_refs,
        has_steps='class="step-player"' in body_html or 'class="step-list"' in body_html,
        md_source=md_source,
        md_clean_html=md_clean_html,
    )

    out_path.write_text(html, encoding="utf-8")
    print(f"  {rel} -> {out_path.relative_to(ROOT)}")
    return out_path


def split_sections(body_html):
    pat = re.compile(r"<h2>(.*?)</h2>", re.DOTALL)
    parts = pat.split(body_html)
    sections = []
    if parts and parts[0].strip():
        sections.append({"title": None, "html": parts[0]})
    i = 1
    while i < len(parts):
        sections.append({"title": parts[i], "html": parts[i + 1] if i + 1 < len(parts) else ""})
        i += 2
    return sections


def build_all(target=None):
    print("Building pages from content/ ...")
    md_files = sorted(CONTENT_DIR.rglob("*.md"))
    if target:
        md_files = [f for f in md_files if target in f.stem]
        if not md_files:
            print(f"  No content file matching '{target}'")
            return
    for f in md_files:
        build_page(f)
    print("Done.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--watch":
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class Handler(FileSystemEventHandler):
                def on_modified(self, event):
                    if event.src_path.endswith(".md"):
                        print(f"\nChange detected: {event.src_path}")
                        build_all()

            observer = Observer()
            observer.schedule(Handler(), str(CONTENT_DIR), recursive=True)
            observer.start()
            print("Watching content/ for changes... (Ctrl+C to stop)")
            build_all()
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()
        except ImportError:
            print("watchdog not installed. Run: pip install watchdog")
            sys.exit(1)
    else:
        build_all(args[0] if args else None)
