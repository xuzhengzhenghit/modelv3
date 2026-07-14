#!/usr/bin/env python3
"""Visual token utilities — helpers for working with dynamic visual token sequences."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def compute_visual_tokens(image_height: int, image_width: int, patch_size: int = 32) -> int:
    """Compute number of visual tokens for given image dimensions."""
    return (image_height // patch_size) * (image_width // patch_size)


def compute_grid(image_height: int, image_width: int, patch_size: int = 32) -> Tuple[int, int]:
    """Compute (grid_h, grid_w) for given image dimensions."""
    return image_height // patch_size, image_width // patch_size


def build_vision_prefix(
    vision_start_id: int,
    image_pad_id: int,
    vision_end_id: int,
    visual_tokens: int,
) -> list[int]:
    """Build: [vision_start, image_pad × N, vision_end]"""
    return [vision_start_id] + [image_pad_id] * visual_tokens + [vision_end_id]
