#!/usr/bin/env python3
"""Style sampler — generate randomized rendering specifications for each sample."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RenderSpec:
    """A complete rendering specification for one sample."""

    # Typography
    font_family: str = "Times New Roman"
    font_size: int = 26
    line_height: float = 1.4
    letter_spacing: float = 0.0
    text_align: str = "justify"
    font_weight: str = "normal"
    font_style: str = "normal"

    # Layout
    layout_style: str = "scientific_paper"
    max_content_width: int = 768
    top_margin: int = 32
    bottom_margin: int = 32
    left_margin: int = 32
    right_margin: int = 32

    # Difficulty / degradation
    difficulty: str = "clean"  # clean, mild, hard
    degradation_params: dict[str, Any] = field(default_factory=dict)

    # Color scheme
    text_color: str = "#000000"
    bg_color: str = "#FFFFFF"

    # Seed for reproducibility
    seed: int = 0

    # Formula rendering
    formula_size_mult: float = 1.0  # multiplier for KaTeX font size

    # Table styling
    table_border: str = "1px solid #333"
    table_font_size_pct: int = 90

    def css(self) -> str:
        """Generate CSS string for body/content element."""
        return f"""
            font-family: {self.font_family};
            font-size: {self.font_size}px;
            line-height: {self.line_height};
            letter-spacing: {self.letter_spacing}em;
            text-align: {self.text_align};
            font-weight: {self.font_weight};
            font-style: {self.font_style};
            color: {self.text_color};
            background: {self.bg_color};
            max-width: {self.max_content_width}px;
            padding: {self.top_margin}px {self.right_margin}px {self.bottom_margin}px {self.left_margin}px;
        """


class StyleSampler:
    """Generates randomized RenderSpecs using a reproducible seed."""

    _DEFAULTS = {
        "margins": {"top": [16, 64], "bottom": [16, 64], "left": [16, 64], "right": [16, 64]},
        "typography": {"font_size": [18, 34], "line_height": [1.2, 1.8], "letter_spacing": [0.0, 0.03], "text_align": ["justify", "left"]},
        "page_layouts": {
            "scientific_paper": {"weight": 0.30, "margin_mult": 1.3, "max_width": 800, "font_preference": "serif"},
            "textbook": {"weight": 0.20, "margin_mult": 1.2, "max_width": 750, "font_preference": "serif"},
            "news_web": {"weight": 0.15, "margin_mult": 0.8, "max_width": 700, "font_preference": "sans_serif"},
            "blog": {"weight": 0.10, "margin_mult": 1.0, "max_width": 720, "font_preference": "sans_serif"},
            "documentation": {"weight": 0.10, "margin_mult": 1.1, "max_width": 900, "font_preference": "monospace"},
            "terminal": {"weight": 0.05, "margin_mult": 0.6, "max_width": 640, "font_preference": "monospace"},
            "report": {"weight": 0.05, "margin_mult": 1.4, "max_width": 780, "font_preference": "serif"},
            "dense_text": {"weight": 0.05, "margin_mult": 0.5, "max_width": 960, "font_preference": "serif"},
        },
        "fonts": {
            "serif": ["Times New Roman", "Georgia", "Noto Serif"],
            "sans_serif": ["Arial", "Helvetica", "Noto Sans"],
            "monospace": ["Courier New", "Consolas", "Noto Sans Mono"],
        },
    }

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self._margins = cfg.get("margins") or self._DEFAULTS["margins"]
        self._typo = cfg.get("typography") or self._DEFAULTS["typography"]
        self._layouts = cfg.get("page_layouts") or self._DEFAULTS["page_layouts"]
        self._fonts = cfg.get("fonts") or self._DEFAULTS["fonts"]
        self._difficulty_weights = {"clean": 0.80, "mild": 0.18, "hard": 0.02}

    def set_curriculum(self, clean: float, mild: float, hard: float):
        self._difficulty_weights = {"clean": clean, "mild": mild, "hard": hard}

    def sample(self, sample_id: str, epoch: int, global_seed: int = 42) -> RenderSpec:
        """Generate a deterministic but varying RenderSpec."""
        seed_str = f"{sample_id}|{epoch}|{global_seed}"
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:16], 16) % (2**31)
        rng = random.Random(seed)

        # ── Pick layout style ──
        layout_names = list(self._layouts.keys())
        layout_weights = [self._layouts.get(k, {}).get("weight", 1.0) for k in layout_names]
        layout_style = rng.choices(layout_names, weights=layout_weights, k=1)[0]
        layout_cfg = self._layouts.get(layout_style, {})

        # ── Typography ──
        font_size = rng.randint(
            self._typo.get("font_size", [18, 34])[0],
            self._typo.get("font_size", [18, 34])[1],
        )
        line_height = round(
            rng.uniform(
                self._typo.get("line_height", [1.2, 1.8])[0],
                self._typo.get("line_height", [1.2, 1.8])[1],
            ),
            2,
        )
        letter_spacing = round(
            rng.uniform(
                self._typo.get("letter_spacing", [0.0, 0.03])[0],
                self._typo.get("letter_spacing", [0.0, 0.03])[1],
            ),
            3,
        )

        # ── Font selection ──
        font_pref = layout_cfg.get("font_preference", "serif")
        font_pool = self._fonts.get(font_pref, ["Times New Roman"])
        font_family = rng.choice(font_pool)

        # ── Font style (occasional italic/bold) ──
        font_style = rng.choices(["normal", "italic"], weights=[0.85, 0.15])[0]
        font_weight = rng.choices(["normal", "bold"], weights=[0.90, 0.10])[0]

        # ── Margins ──
        margin_mult = layout_cfg.get("margin_mult", 1.0)
        top_margin = int(rng.randint(*self._margins.get("top", [16, 64])) * margin_mult)
        bottom_margin = int(rng.randint(*self._margins.get("bottom", [16, 64])) * margin_mult)
        left_margin = int(rng.randint(*self._margins.get("left", [16, 64])) * margin_mult)
        right_margin = int(rng.randint(*self._margins.get("right", [16, 64])) * margin_mult)

        # ── Max content width ──
        max_content_width = layout_cfg.get("max_width", 768)

        # ── Difficulty ──
        difficulty = rng.choices(
            list(self._difficulty_weights.keys()),
            weights=list(self._difficulty_weights.values()),
            k=1,
        )[0]

        # ── Text align ──
        text_align = rng.choice(self._typo.get("text_align", ["justify", "left"]))

        return RenderSpec(
            font_family=font_family,
            font_size=font_size,
            line_height=line_height,
            letter_spacing=letter_spacing,
            text_align=text_align,
            font_weight=font_weight,
            font_style=font_style,
            layout_style=layout_style,
            max_content_width=max_content_width,
            top_margin=top_margin,
            bottom_margin=bottom_margin,
            left_margin=left_margin,
            right_margin=right_margin,
            difficulty=difficulty,
            seed=seed,
            formula_size_mult=1.0 + (font_size - 26) * 0.02,
            table_font_size_pct=max(70, min(100, int(font_size / 26 * 100))),
        )


VALIDATION_SPEC = RenderSpec(
    font_family="Times New Roman",
    font_size=26,
    line_height=1.4,
    layout_style="scientific_paper",
    max_content_width=768,
    top_margin=32,
    bottom_margin=32,
    left_margin=32,
    right_margin=32,
    difficulty="clean",
    text_align="justify",
    seed=42,
)
