#!/usr/bin/env python3
"""Unicode normalization and text cleaning for scientific documents."""

from __future__ import annotations

import re
import unicodedata

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_NEWLINE_RE = re.compile(r"\n{4,}")
NBSP_LINE_RE = re.compile(r"[ \u00a0]+\n")
WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")


def normalize_text(value: str) -> str:
    """Normalize raw text: unicode, control chars, whitespace, newlines."""
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    value = CONTROL_RE.sub("", value)
    value = unicodedata.normalize("NFC", value)
    value = NBSP_LINE_RE.sub("\n", value)
    value = WHITESPACE_RE.sub(" ", value)
    value = MULTI_NEWLINE_RE.sub("\n\n\n", value)
    return value.strip()


def normalize_whitespace_only(text: str) -> str:
    """Minimal normalization: collapse whitespace without touching unicode."""
    text = WHITESPACE_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTI_NEWLINE_RE.sub("\n\n\n", text)
    return text.strip()


def has_unicode_issues(text: str) -> bool:
    """Check for common Unicode problems in the text."""
    for char in text:
        if ord(char) in (0xFFFD,):  # replacement character
            return True
        cat = unicodedata.category(char)
        if cat == "Co":  # private use
            return True
        if cat == "Cn":  # unassigned
            return True
    return False
