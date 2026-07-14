#!/usr/bin/env python3
"""Font registry — track Unicode coverage, select fonts for given text."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# ── Font classification ──
SERIF_FONTS = ["Times New Roman", "Georgia", "Noto Serif", "Liberation Serif"]
SANS_SERIF_FONTS = ["Arial", "Helvetica", "Noto Sans", "Liberation Sans"]
MONOSPACE_FONTS = ["Courier New", "Consolas", "Noto Sans Mono", "Liberation Mono", "DejaVu Sans Mono"]
CJK_FONTS = ["Noto Sans SC", "Noto Serif SC", "SimSun", "Microsoft YaHei"]
MATH_FONTS = ["KaTeX_Main", "KaTeX_Math"]  # rendered by KaTeX, not system fonts


@dataclass
class FontEntry:
    family: str
    category: str  # serif, sans_serif, monospace, cjk
    weight: float
    styles: list[str] = field(default_factory=lambda: ["normal"])


class FontRegistry:
    """Track available fonts and their Unicode coverage.

    In production, coverage data should come from system font inspection tools
    (fc-list, fontTools, etc.). This class provides the framework.
    """

    def __init__(self):
        self._fonts: dict[str, FontEntry] = {}
        self._register_defaults()

    def _register_defaults(self):
        for f in SERIF_FONTS:
            self._fonts[f.lower()] = FontEntry(f, "serif", 1.0)
        for f in SANS_SERIF_FONTS:
            self._fonts[f.lower()] = FontEntry(f, "sans_serif", 1.0)
        for f in MONOSPACE_FONTS:
            self._fonts[f.lower()] = FontEntry(f, "monospace", 1.0)
        for f in CJK_FONTS:
            self._fonts[f.lower()] = FontEntry(f, "cjk", 0.5)

    def get(self, family: str) -> FontEntry | None:
        return self._fonts.get(family.lower())

    def select_for_text(self, text: str, category: str, prefer: str | None = None) -> str:
        """Select a font from the given category that covers the text.

        For simplicity, returns the preferred font or first in category.
        Full Unicode coverage checking can be added later.
        """
        candidates = [f for f in self._fonts.values() if f.category == category]
        if not candidates:
            candidates = [self._fonts.get("arial", FontEntry("Arial", "sans_serif", 1.0))]

        if prefer:
            entry = self._fonts.get(prefer.lower())
            if entry and entry.category == category:
                return entry.family

        return candidates[0].family if candidates else "Arial"

    def needs_cjk_fallback(self, text: str) -> bool:
        """Check if text contains CJK characters."""
        cjk_pattern = re.compile(
            r"[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]"
        )
        return bool(cjk_pattern.search(text))

    def css_font_stack(self, text: str, category: str) -> str:
        """Build CSS font-family string with appropriate fallbacks."""
        primary = self.select_for_text(text, category)
        if self.needs_cjk_fallback(text):
            cjk_font = self.select_for_text(text, "cjk")
            return f'"{primary}", "{cjk_font}", serif'
        return f'"{primary}", serif'
