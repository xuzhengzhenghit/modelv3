#!/usr/bin/env python3
"""HTML sanitizer — extract safe text/table blocks, strip scripts and externals."""

from __future__ import annotations

import html as html_mod
import re
from typing import Any

try:
    from bs4 import BeautifulSoup, NavigableString, Tag
except Exception:  # pragma: no cover
    BeautifulSoup = None
    NavigableString = Any
    Tag = Any

# Tags we allow for content extraction
ALLOWED_TEXT_TAGS = {"p", "h1", "h2", "h3", "h4", "li", "blockquote", "div", "span", "article", "section", "pre", "code"}
ALLOWED_TABLE_TAGS = {"table", "thead", "tbody", "tr", "td", "th"}
ALLOWED_INLINE_TAGS = {"strong", "em", "b", "i", "sub", "sup", "code", "span"}
FORBIDDEN_TAGS = {"script", "style", "iframe", "object", "embed", "noscript", "svg", "math"}
FORBIDDEN_ATTRS = {"onload", "onclick", "onerror", "onmouseover", "onfocus", "style", "href", "src"}
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")


def strip_html_plain(text: str) -> str:
    """Fallback: strip all HTML tags, return unescaped visible text."""
    visible = re.sub(r"<[^>]+>", " ", text)
    return html_mod.unescape(visible).strip()


def extract_safe_blocks(html_text: str) -> list[dict[str, Any]]:
    """Extract paragraph and table blocks from HTML, stripping dangerous elements.

    Returns list of {"kind": "paragraph"|"table", "content": ...}.
    This is a lightweight alternative to the full document_parser for HTML-heavy sources.
    """
    if BeautifulSoup is None:
        raw_text = strip_html_plain(html_text)
        return [{"kind": "paragraph", "content": raw_text}] if raw_text else []

    soup = BeautifulSoup(html_text, "html.parser")

    # Remove forbidden tags entirely
    for tag_name in FORBIDDEN_TAGS:
        for node in soup.find_all(tag_name):
            node.decompose()

    # Strip forbidden attributes from all tags
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr.lower() in FORBIDDEN_ATTRS:
                del tag[attr]

    blocks: list[dict[str, Any]] = []
    seen_tables: set[int] = set()

    for node in soup.descendants:
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
                blocks.append({"kind": "table", "rows": rows})

        elif isinstance(node, Tag) and node.name in ALLOWED_TEXT_TAGS:
            if node.find_parent("table") is not None:
                continue
            value = node.get_text(" ", strip=True)
            if value:
                blocks.append({"kind": "paragraph", "content": value})

    if not blocks:
        raw_text = soup.get_text("\n", strip=True)
        if raw_text:
            blocks.append({"kind": "paragraph", "content": raw_text})

    return blocks


def has_html_risks(text: str) -> bool:
    """Check if text contains potentially dangerous HTML."""
    risky = re.compile(r"<(script|iframe|object|embed)\b", re.IGNORECASE)
    return bool(risky.search(text))


def contains_html(text: str) -> bool:
    """Check if text contains any HTML markup."""
    return bool(HTML_TAG_RE.search(text))
