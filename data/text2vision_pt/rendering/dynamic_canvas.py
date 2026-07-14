#!/usr/bin/env python3
"""Dynamic canvas calculator — determine optimal image dimensions from content size.

Algorithm:
    1. Browser renders content with spec → measure content_width × content_height
    2. Add margins: raw_w = content_w + left + right, raw_h = content_h + top + bottom
    3. Add random slack: extra_w ∈ {0, 0, 32, 64}, extra_h ∈ {0, 0, 32}
    4. Align to 32: width = ceil(raw_w / 32) * 32, height = ceil(raw_h / 32) * 32
    5. Clamp to [min_size, max_size]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class CanvasSpec:
    """Final canvas dimensions and derived visual token grid."""

    width: int
    height: int
    grid_w: int
    grid_h: int
    visual_tokens: int
    content_w: int
    content_h: int
    overflow: bool = False


class DynamicCanvas:
    """Compute optimal canvas dimensions for rendered content."""

    def __init__(
        self,
        min_width: int = 256,
        max_width: int = 1024,
        min_height: int = 64,
        max_height: int = 512,
        step: int = 32,
        extra_slack_patches: list[int] | None = None,
    ):
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.step = step
        self.extra_slack_patches = extra_slack_patches or [0, 0, 1, 2]
        self._patch_px = step  # one patch = step pixels

    def compute(
        self,
        content_width: int,
        content_height: int,
        top_margin: int,
        bottom_margin: int,
        left_margin: int,
        right_margin: int,
        extra_w: Optional[int] = None,
        extra_h: Optional[int] = None,
    ) -> CanvasSpec:
        """Compute canvas size from measured content dimensions.

        Args:
            content_width: Measured content width in pixels.
            content_height: Measured content height in pixels.
            top_margin, bottom_margin, left_margin, right_margin: Margins in pixels.
            extra_w, extra_h: Optional explicit extra slack. If None, chosen randomly.

        Returns:
            CanvasSpec with final dimensions and visual token counts.
        """
        import random

        # 1. Add margins
        raw_w = content_width + left_margin + right_margin
        raw_h = content_height + top_margin + bottom_margin

        # 2. Add random slack
        if extra_w is None:
            extra_w = random.choice(self.extra_slack_patches) * self._patch_px
        if extra_h is None:
            extra_h = random.choice([0, 0, 32])  # fewer vertical slack options

        raw_w += extra_w
        raw_h += extra_h

        # 3. Align to step
        width = math.ceil(raw_w / self.step) * self.step
        height = math.ceil(raw_h / self.step) * self.step

        # 4. Clamp
        overflow = False
        if width > self.max_width:
            width = self.max_width
            overflow = True
        if height > self.max_height:
            height = self.max_height
            overflow = True
        width = max(width, self.min_width)
        height = max(height, self.min_height)

        # 5. Compute grid
        grid_w = width // self.step
        grid_h = height // self.step
        visual_tokens = grid_w * grid_h

        return CanvasSpec(
            width=width,
            height=height,
            grid_w=grid_w,
            grid_h=grid_h,
            visual_tokens=visual_tokens,
            content_w=content_width,
            content_h=content_height,
            overflow=overflow,
        )

    def compute_from_measure(
        self,
        measured_element: dict[str, int],
        margins: dict[str, int],
        seed: int = 0,
    ) -> CanvasSpec:
        """Compute canvas from a browser measurement result dict."""
        import random
        rng = random.Random(seed)
        extra_w = rng.choice(self.extra_slack_patches) * self._patch_px
        extra_h = rng.choice([0, 0, 32])
        return self.compute(
            content_width=measured_element["width"],
            content_height=measured_element["height"],
            top_margin=margins["top"],
            bottom_margin=margins["bottom"],
            left_margin=margins["left"],
            right_margin=margins["right"],
            extra_w=extra_w,
            extra_h=extra_h,
        )

    def validate(self, width: int, height: int) -> bool:
        """Check if dimensions are valid (multiples of step, within bounds)."""
        if width % self.step != 0 or height % self.step != 0:
            return False
        if width < self.min_width or width > self.max_width:
            return False
        if height < self.min_height or height > self.max_height:
            return False
        return True
