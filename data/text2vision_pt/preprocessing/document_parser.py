#!/usr/bin/env python3
r"""Parse mixed scientific text into safe, JSON-serializable render blocks.

Supported inputs:
- plain paragraphs
- inline math: \(...\), heuristic $...$
- display math: \[...\], $$...$$, equation/align environments
- [latex]...[/latex]
- common HTML paragraphs/tables
- Markdown pipe tables

The parser deliberately converts arbitrary HTML into safe text/table blocks instead
of injecting untrusted source HTML into Chromium.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any, Iterable

try:
    from bs4 import BeautifulSoup, NavigableString, Tag
except Exception:  # pragma: no cover
    BeautifulSoup = None
    NavigableString = Any
    Tag = Any

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
MATH_SUBJECTS = {
    "mathematics",
    "physics",
    "chemistry",
    "astronomy",
    "electricalengineering",
    "electrical engineering",
}

MATH_COMMAND_RE = re.compile(
    r"\\(?:frac|dfrac|tfrac|sqrt|sum|prod|int|iint|iiint|lim|log|ln|sin|cos|tan|"
    r"alpha|beta|gamma|delta|theta|lambda|mu|rho|sigma|omega|partial|nabla|cdot|"
    r"times|leq|geq|neq|approx|infty|mathrm|mathbf|mathit|text|begin|end)\b"
)
CURRENCY_RE = re.compile(
    r"^\s*[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?"
    r"(?:\s*(?:USD|EUR|GBP|dollars?|euros?|pounds?|million|billion|trillion))?\s*$",
    re.IGNORECASE,
)

DISPLAY_RE = re.compile(
    r"(?P<env>\\begin\{(?P<envname>equation\*?|align\*?|alignat\*?|gather\*?)\}.*?"
    r"\\end\{(?P=envname)\})"
    r"|(?P<bracket>\\\[.*?\\\])"
    r"|(?P<dollar>\$\$.*?\$\$)"
    r"|(?P<bbcode>\[latex\].*?\[/latex\])",
    re.IGNORECASE | re.DOTALL,
)

INLINE_RE = re.compile(
    r"(?P<bracket>\\\(.*?\\\))|(?P<dollar>(?<!\\)\$(?!\$).*?(?<!\\)\$)",
    re.DOTALL,
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？；;])\s+|\n+")
BARE_TEXT_COMMAND_RE = re.compile(r"\\(?:textbf|textit|emph|mathrm|mathbf|mathit)\{([^{}]*)\}")
BARE_SUBSCRIPT_RE = re.compile(r"\\textsubscript\{([^{}]*)\}")
BARE_SUPERSCRIPT_RE = re.compile(r"\\textsuperscript\{([^{}]*)\}")


def normalize_text(value: Any) -> str:
    if isinstance(value, list):
        value = "\n".join(str(x) for x in value)
    elif not isinstance(value, str):
        value = str(value)
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    value = CONTROL_RE.sub("", value)
    value = unicodedata.normalize("NFC", value)
    value = re.sub(r"[ \u00a0]+\n", "\n", value)
    value = re.sub(r"\n{4,}", "\n\n\n", value)
    return value.strip()


def looks_like_inline_math(content: str, subject: str | None = None) -> bool:
    content = content.strip()
    if not content or "\n" in content or len(content) > 256:
        return False
    if CURRENCY_RE.fullmatch(content):
        return False

    command = bool(MATH_COMMAND_RE.search(content))
    strong_symbol = any(ch in content for ch in ("_", "^", "=", "{", "}", "≤", "≥", "∫", "∑", "√"))
    operator_expression = bool(re.search(r"[A-Za-z0-9)\]}]\s*[+\-*/=<>]\s*[A-Za-z0-9({\[]", content))
    subscript_or_power = bool(re.search(r"[A-Za-z0-9][_^]\{?[A-Za-z0-9+\-]+\}?", content))
    subject_is_math = (subject or "").strip().lower() in MATH_SUBJECTS

    if subject_is_math:
        return command or strong_symbol or operator_expression or subscript_or_power
    return command or strong_symbol or operator_expression


def _strip_bare_visual_commands(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = BARE_TEXT_COMMAND_RE.sub(r"\1", text)
        text = BARE_SUBSCRIPT_RE.sub(r"\1", text)
        text = BARE_SUPERSCRIPT_RE.sub(r"\1", text)
    return text


def canonical_display(raw: str) -> tuple[str, str]:
    """Return (KaTeX source, canonical OCR target)."""
    value = raw.strip()
    low = value.lower()

    if value.startswith(r"\[") and value.endswith(r"\]"):
        inner = value[2:-2].strip()
        return inner, rf"\[{inner}\]"
    if value.startswith("$$") and value.endswith("$$"):
        inner = value[2:-2].strip()
        return inner, rf"\[{inner}\]"
    if low.startswith("[latex]") and low.endswith("[/latex]"):
        inner = value[7:-8].strip()
        return inner, rf"\[{inner}\]"

    env_match = re.match(r"\\begin\{([^{}]+)\}(.*?)\\end\{\1\}\s*$", value, re.DOTALL)
    if env_match:
        env_name = env_match.group(1).rstrip("*")
        body = env_match.group(2).strip()
        if env_name.startswith("align") or env_name == "gather":
            katex_source = rf"\begin{{aligned}}{body}\end{{aligned}}"
        else:
            katex_source = body
        return katex_source, rf"\[{body}\]"

    return value, rf"\[{value}\]"


def _single_dollar_positions(text: str) -> list[int]:
    positions: list[int] = []
    for index, char in enumerate(text):
        if char != "$":
            continue
        if index > 0 and text[index - 1] == "\\":
            continue
        if index > 0 and text[index - 1] == "$":
            continue
        if index + 1 < len(text) and text[index + 1] == "$":
            continue
        positions.append(index)
    return positions


def _inline_math_spans(text: str, subject: str | None) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []

    # Strong \(...\) delimiters.
    for match in re.finditer(r"\\\((.*?)\\\)", text, re.DOTALL):
        spans.append((match.start(), match.end(), match.group(1).strip()))

    # Single-dollar delimiters are ambiguous. If one candidate pair is rejected as
    # currency/prose, advance by one dollar instead of consuming both. This allows:
    # "costs $10, while the equation is $x=2$" to preserve $10 and still find $x=2$.
    positions = _single_dollar_positions(text)
    index = 0
    while index + 1 < len(positions):
        left, right = positions[index], positions[index + 1]
        inner = text[left + 1 : right].strip()
        if looks_like_inline_math(inner, subject):
            spans.append((left, right + 1, inner))
            index += 2
        else:
            index += 1

    spans.sort(key=lambda item: (item[0], item[1]))
    non_overlapping: list[tuple[int, int, str]] = []
    cursor = -1
    for span in spans:
        if span[0] >= cursor:
            non_overlapping.append(span)
            cursor = span[1]
    return non_overlapping


def parse_inline_parts(text: str, subject: str | None = None) -> list[dict[str, str]]:
    text = _strip_bare_visual_commands(text)
    spans = _inline_math_spans(text, subject)
    if not spans:
        return [{"kind": "text", "text": text}]

    parts: list[dict[str, str]] = []
    cursor = 0
    for start, end, inner in spans:
        if start > cursor:
            parts.append({"kind": "text", "text": text[cursor:start]})
        parts.append({"kind": "math_inline", "tex": inner, "target": rf"\({inner}\)"})
        cursor = end
    if cursor < len(text):
        parts.append({"kind": "text", "text": text[cursor:]})
    return parts


def split_long_paragraph(text: str, max_chars: int = 420) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    sentences = [x.strip() for x in SENTENCE_SPLIT_RE.split(text) if x.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(sentence), max_chars):
                chunks.append(sentence[start : start + max_chars])
            continue
        candidate = sentence if not current else current + " " + sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _paragraph_blocks(text: str, subject: str | None, max_block_chars: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for chunk in split_long_paragraph(paragraph, max_chars=max_block_chars):
            result.append({"kind": "paragraph", "parts": parse_inline_parts(chunk, subject)})
    return result


def parse_text_with_math(text: str, subject: str | None = None, max_block_chars: int = 420) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    cursor = 0
    for match in DISPLAY_RE.finditer(text):
        prefix = text[cursor : match.start()]
        blocks.extend(_paragraph_blocks(prefix, subject, max_block_chars))

        raw = match.group(0)
        if match.group("bbcode"):
            inner = raw[7:-8].strip()
            if "\n" not in inner and len(inner) <= 80:
                blocks.append(
                    {
                        "kind": "paragraph",
                        "parts": [{"kind": "math_inline", "tex": inner, "target": rf"\({inner}\)"}],
                    }
                )
            else:
                tex, target = canonical_display(raw)
                blocks.append({"kind": "math_display", "tex": tex, "target": target})
        else:
            tex, target = canonical_display(raw)
            blocks.append({"kind": "math_display", "tex": tex, "target": target})
        cursor = match.end()

    blocks.extend(_paragraph_blocks(text[cursor:], subject, max_block_chars))
    return blocks


def _parse_markdown_table(lines: list[str], start: int) -> tuple[dict[str, Any] | None, int]:
    if start + 1 >= len(lines) or "|" not in lines[start]:
        return None, start
    separator = lines[start + 1].strip()
    if not re.match(r"^\|?\s*:?-{3,}.*\|.*$", separator):
        return None, start

    rows: list[list[str]] = []
    index = start
    while index < len(lines) and "|" in lines[index] and lines[index].strip():
        line = lines[index].strip().strip("|")
        cells = [cell.strip() for cell in line.split("|")]
        if index != start + 1:
            rows.append(cells)
        index += 1
    return {"kind": "table", "rows": rows, "target": "\n".join("\t".join(r) for r in rows)}, index


def parse_markdown_and_math(text: str, subject: str | None, max_block_chars: int) -> list[dict[str, Any]]:
    lines = text.splitlines()
    blocks: list[dict[str, Any]] = []
    plain_buffer: list[str] = []

    def flush() -> None:
        if plain_buffer:
            blocks.extend(parse_text_with_math("\n".join(plain_buffer), subject, max_block_chars))
            plain_buffer.clear()

    index = 0
    while index < len(lines):
        table, next_index = _parse_markdown_table(lines, index)
        if table is not None:
            flush()
            blocks.append(table)
            index = next_index
        else:
            plain_buffer.append(lines[index])
            index += 1
    flush()
    return blocks


def parse_html_document(text: str, subject: str | None, max_block_chars: int) -> list[dict[str, Any]]:
    if BeautifulSoup is None:
        visible = re.sub(r"<[^>]+>", " ", text)
        return parse_markdown_and_math(html.unescape(visible), subject, max_block_chars)

    soup = BeautifulSoup(text, "html.parser")
    root = soup.body or soup
    blocks: list[dict[str, Any]] = []
    seen_tables: set[int] = set()

    for node in root.descendants:
        if isinstance(node, Tag) and node.name == "table":
            if id(node) in seen_tables:
                continue
            seen_tables.add(id(node))
            rows: list[list[str]] = []
            for tr in node.find_all("tr"):
                cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"], recursive=False)]
                if cells:
                    rows.append(cells)
            if rows:
                blocks.append(
                    {"kind": "table", "rows": rows, "target": "\n".join("\t".join(r) for r in rows)}
                )
        elif isinstance(node, Tag) and node.name in {"p", "h1", "h2", "h3", "h4", "li", "blockquote"}:
            if node.find_parent("table") is not None:
                continue
            value = node.get_text(" ", strip=True)
            if value:
                blocks.extend(parse_text_with_math(value, subject, max_block_chars))

    if not blocks:
        visible = soup.get_text("\n", strip=True)
        blocks = parse_markdown_and_math(visible, subject, max_block_chars)
    return blocks


def parse_document(text: Any, subject: str | None = None, max_block_chars: int = 420) -> list[dict[str, Any]]:
    value = normalize_text(text)
    if not value:
        return []
    if HTML_TAG_RE.search(value):
        return parse_html_document(value, subject, max_block_chars)
    return parse_markdown_and_math(value, subject, max_block_chars)


def block_target(block: dict[str, Any]) -> str:
    kind = block.get("kind")
    if kind == "paragraph":
        chunks: list[str] = []
        for part in block.get("parts", []):
            if part.get("kind") == "text":
                chunks.append(part.get("text", ""))
            elif part.get("kind") == "math_inline":
                chunks.append(part.get("target") or rf"\({part.get('tex', '')}\)")
        return "".join(chunks).strip()
    if kind == "math_display":
        return str(block.get("target") or rf"\[{block.get('tex', '')}\]").strip()
    if kind == "table":
        if block.get("target"):
            return str(block["target"]).strip()
        return "\n".join("\t".join(map(str, row)) for row in block.get("rows", [])).strip()
    return ""


def join_targets(blocks: Iterable[dict[str, Any]]) -> str:
    return "\n".join(value for block in blocks if (value := block_target(block))).strip()


def estimated_block_cost(block: dict[str, Any]) -> int:
    kind = block.get("kind")
    if kind == "paragraph":
        return max(1, len(block_target(block)))
    if kind == "math_display":
        return max(140, int(len(block.get("tex", "")) * 1.7))
    if kind == "table":
        rows = block.get("rows", [])
        return max(180, len(rows) * 85)
    return 50


def pack_pages(blocks: list[dict[str, Any]], page_budget: int = 950) -> list[list[dict[str, Any]]]:
    pages: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    cost = 0
    for block in blocks:
        block_cost = estimated_block_cost(block)
        if current and cost + block_cost > page_budget:
            pages.append(current)
            current = []
            cost = 0
        current.append(block)
        cost += block_cost
    if current:
        pages.append(current)
    return pages
