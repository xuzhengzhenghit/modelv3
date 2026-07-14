#!/usr/bin/env python3
"""Document chunker — parse text into structured blocks, then split into training pages.

This is the core preprocessing module that upgrades the existing document_parser.
It ensures no cuts happen inside:
- English words
- Unicode multi-byte characters
- LaTeX inline math \\(...\\)
- LaTeX display math \\[...\\]
- LaTeX environments \\begin{align}...\\end{align}
- HTML/Markdown tables
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# ── Reuse the existing document_parser ──
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from document_parser import (  # noqa: E402
    block_target,
    estimated_block_cost,
    join_targets,
    parse_document,
)

# ── Sentence splitting that respects inline math ──
# Split on sentence boundaries that are NOT inside \\(...\\)
SENTENCE_END = re.compile(r"(?<=[.!?。！？；;])\s+")
WORD_CHAR_RE = re.compile(r"[\w']")
MATH_INLINE_RE = re.compile(r"\\\(.*?\\\)")


@dataclass
class PageRecord:
    """One training page / OCR unit."""

    id: str
    doc_id: str
    page_index: int
    subject: str = ""
    blocks: list[dict[str, Any]] = field(default_factory=list)
    target_text: str = ""
    has_math: bool = False
    has_table: bool = False
    flags: int = 0


def compute_flags(blocks: list[dict[str, Any]]) -> int:
    """Compute flags bitmask: 1=has_math, 2=has_table, 4=has_html."""
    flags = 0
    for block in blocks:
        kind = block.get("kind", "")
        if kind in ("math_inline", "math_display"):
            flags |= 1
        elif kind == "table":
            flags |= 2
        if block.get("from_html"):
            flags |= 4
    # Also check paragraph parts for inline math
    if not (flags & 1):
        for block in blocks:
            for part in block.get("parts", []):
                if part.get("kind") == "math_inline":
                    flags |= 1
                    break
            if flags & 1:
                break
    return flags


def pack_pages_with_guard(
    blocks: list[dict[str, Any]],
    page_budget: int = 950,
    max_block_chars: int = 420,
    subject: str | None = None,
) -> list[list[dict[str, Any]]]:
    """Pack blocks into pages, ensuring no cut inside protected spans.

    Uses ``pack_pages`` from document_parser with the estimated_block_cost heuristic.
    Falls back to single-block pages when a block alone exceeds budget.
    """
    # Split large paragraphs at sentence boundaries
    refined: list[dict[str, Any]] = []
    for block in blocks:
        kind = block.get("kind")
        if kind == "paragraph":
            text = block_target(block)
            if len(text) > max_block_chars:
                # Re-parse as smaller paragraphs
                from document_parser import parse_text_with_math
                sub_blocks = parse_text_with_math(text, subject, max_block_chars)
                refined.extend(sub_blocks)
                continue
        refined.append(block)

    from document_parser import pack_pages
    return pack_pages(refined, page_budget)


def document_to_pages(
    text: str,
    doc_id: str,
    subject: str | None = None,
    page_budget: int = 950,
    max_block_chars: int = 420,
) -> list[PageRecord]:
    """Parse one document and split into page records."""
    blocks = parse_document(text, subject, max_block_chars)
    if not blocks:
        return []

    page_blocks = pack_pages_with_guard(blocks, page_budget, max_block_chars, subject)
    records: list[PageRecord] = []
    for i, pblocks in enumerate(page_blocks):
        target = join_targets(pblocks)
        if not target:
            continue
        has_math = any(
            b.get("kind") in ("math_display", "math_inline")
            or any(p.get("kind") == "math_inline" for p in b.get("parts", []))
            for b in pblocks
        )
        has_table = any(b.get("kind") == "table" for b in pblocks)
        flags = compute_flags(pblocks)
        records.append(
            PageRecord(
                id=f"{doc_id}-p{i:05d}",
                doc_id=doc_id,
                page_index=i,
                subject=subject or "",
                blocks=pblocks,
                target_text=target,
                has_math=has_math,
                has_table=has_table,
                flags=flags,
            )
        )
    return records


def compact_record(record: PageRecord) -> dict[str, str | int]:
    """Convert PageRecord to compact JSONL format {i, t, s, f}."""
    return {
        "i": record.id,
        "t": record.target_text,
        "s": record.subject,
        "f": record.flags,
    }


def full_record(record: PageRecord) -> dict[str, Any]:
    """Convert PageRecord to full JSONL format."""
    return {
        "id": record.id,
        "doc_id": record.doc_id,
        "page_index": record.page_index,
        "subject": record.subject,
        "blocks": record.blocks,
        "target_text": record.target_text,
        "has_math": record.has_math,
        "has_table": record.has_table,
        "flags": record.flags,
    }
