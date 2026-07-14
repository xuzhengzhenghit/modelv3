#!/usr/bin/env python3
"""GZip-aware streaming JSON reader."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator

SUFFIX_MAP = {
    ".gz": "gzip",
    ".gzip": "gzip",
    ".jsonl": "plain",
    ".json": "plain",
}


def detect_format(filepath: str | Path) -> str:
    """Detect file compression format from suffix."""
    path = Path(filepath)
    suffixes = path.suffixes
    for suffix in suffixes:
        fmt = SUFFIX_MAP.get(suffix.lower())
        if fmt:
            return fmt
    return "plain"


def open_lines(filepath: str | Path) -> Iterator[str]:
    """Open a text file (plain or gzipped) and yield lines."""
    fmt = detect_format(filepath)
    if fmt == "gzip":
        with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as f:
            yield from f
    else:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            yield from f


def read_jsonl(filepath: str | Path) -> Iterator[dict[str, Any]]:
    """Stream JSON objects from a .jsonl or .jsonl.gz file."""
    for line_num, line in enumerate(open_lines(filepath), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {filepath}:{line_num}: {exc}") from exc


def read_json(filepath: str | Path) -> Any:
    """Read a single JSON file (plain or gzipped)."""
    fmt = detect_format(filepath)
    if fmt == "gzip":
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)


def count_lines_estimate(filepath: str | Path, sample_mb: float = 4.0) -> tuple[int, int]:
    """Estimate total lines by sampling first sample_mb of the file."""
    fmt = detect_format(filepath)
    if fmt == "gzip":
        # Can't seek in gzip, just count
        total = sum(1 for _ in open_lines(filepath))
        return total, total

    path = Path(filepath)
    file_size = path.stat().st_size
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        sample_bytes = int(sample_mb * 1024 * 1024)
        chunk = f.read(sample_bytes)
        sample_lines = chunk.count("\n")
        if len(chunk) < file_size:
            estimated = int(sample_lines * (file_size / len(chunk)))
            return sample_lines, estimated
        return sample_lines, sample_lines
