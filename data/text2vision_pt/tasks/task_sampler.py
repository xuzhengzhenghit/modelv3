#!/usr/bin/env python3
"""Task system — defines Full OCR, Optical Continuation, Span Reconstruction, Text Replay."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(Enum):
    FULL_OCR = "full_ocr"
    OPTICAL_CONTINUATION = "optical_continuation"
    SPAN_RECONSTRUCTION = "span_reconstruction"
    TEXT_REPLAY = "text_replay"


@dataclass
class TaskSpec:
    """Specification for one training sample's task."""

    task_type: TaskType
    # Text segments for input construction
    prefix_text: str = ""       # Text before visual tokens (Task B, C)
    visual_text: str = ""       # Text rendered as image (Task A, B, C)
    suffix_text: str = ""       # Text after visual tokens (Task B)
    target_text: str = ""       # Text that model must predict
    # Metadata
    blocks: list[dict[str, Any]] = field(default_factory=list)
    is_text_only: bool = False

    @property
    def source_text(self) -> str:
        """Complete source text (prefix + visual + suffix)."""
        parts = [self.prefix_text, self.visual_text, self.suffix_text]
        return "".join(p for p in parts if p)


class TaskSampler:
    """Sample and construct training tasks from page records."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        span_min_tokens: int = 16,
        span_max_tokens: int = 256,
        span_distribution: str = "log_uniform",
    ):
        self._weights = weights or {
            "full_ocr": 0.60,
            "optical_continuation": 0.10,
            "span_reconstruction": 0.20,
            "text_replay": 0.10,
        }
        self._span_min = span_min_tokens
        self._span_max = span_max_tokens
        self._span_dist = span_distribution

    def set_weights(self, weights: dict[str, float]):
        self._weights = weights

    def sample_task(self, record: dict[str, Any], epoch: int, global_seed: int = 42) -> TaskSpec:
        """Choose and construct a task for this sample."""
        seed_str = f"{record.get('i', record.get('id', ''))}|task|{epoch}|{global_seed}"
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:16], 16) % (2**31)
        rng = random.Random(seed)

        task_types = list(self._weights.keys())
        weights = [self._weights[t] for t in task_types]
        task_name = rng.choices(task_types, weights=weights, k=1)[0]

        text = record.get("t", record.get("target_text", ""))
        blocks = record.get("blocks", [])

        if task_name == "full_ocr":
            return self._build_full_ocr(text, blocks)
        elif task_name == "optical_continuation":
            return self._build_optical_continuation(text, blocks, rng)
        elif task_name == "span_reconstruction":
            return self._build_span_reconstruction(text, blocks, rng)
        elif task_name == "text_replay":
            return self._build_text_replay(text, blocks)
        else:
            return self._build_full_ocr(text, blocks)

    def _build_full_ocr(self, text: str, blocks: list[dict[str, Any]]) -> TaskSpec:
        return TaskSpec(
            task_type=TaskType.FULL_OCR,
            visual_text=text,
            target_text=text,
            blocks=blocks,
        )

    def _build_optical_continuation(self, text: str, blocks: list[dict[str, Any]], rng: random.Random) -> TaskSpec:
        """Split text at a boundary, render middle as image, predict suffix."""
        if len(text) < 100:
            return self._build_full_ocr(text, blocks)  # fallback

        boundaries = _find_boundaries(text)
        if len(boundaries) < 3:
            return self._build_full_ocr(text, blocks)

        # Pick a start boundary (after some prefix)
        max_start = len(boundaries) - 2
        if max_start < 1:
            return self._build_full_ocr(text, blocks)

        start_idx = rng.randint(1, max_start)
        start_pos = boundaries[start_idx]

        # Pick an end boundary with reasonable span length
        span_chars = text[start_pos : boundaries[-1]]
        end_idx = start_idx + 1
        while end_idx < len(boundaries) and boundaries[end_idx] - start_pos < 300:
            end_idx += 1
        if end_idx >= len(boundaries):
            end_idx = len(boundaries) - 1

        end_pos = boundaries[end_idx]

        prefix = text[:start_pos].strip()
        visual = text[start_pos:end_pos].strip()
        suffix = text[end_pos:].strip()

        if not visual or len(visual) < 10:
            return self._build_full_ocr(text, blocks)
        if not suffix or len(suffix) < 10:
            # Predict the visual span instead
            return TaskSpec(
                task_type=TaskType.SPAN_RECONSTRUCTION,
                prefix_text=prefix,
                visual_text=visual,
                target_text=visual,
                blocks=blocks,
            )

        return TaskSpec(
            task_type=TaskType.OPTICAL_CONTINUATION,
            prefix_text=prefix,
            visual_text=visual,
            suffix_text=suffix,
            target_text=suffix,
            blocks=blocks,
        )

    def _build_span_reconstruction(self, text: str, blocks: list[dict[str, Any]], rng: random.Random) -> TaskSpec:
        """Render a span as image, model must decode it."""
        if len(text) < 60:
            return self._build_full_ocr(text, blocks)

        boundaries = _find_boundaries(text)
        if len(boundaries) < 3:
            return self._build_full_ocr(text, blocks)

        # Pick start/end to create a reasonable visual span
        max_tries = 10
        for _ in range(max_tries):
            si = rng.randint(0, len(boundaries) - 2)
            ei = rng.randint(si + 1, min(si + 5, len(boundaries) - 1))
            visual = text[boundaries[si] : boundaries[ei]].strip()
            if 15 < len(visual) < 500:
                prefix = text[: boundaries[si]].strip()
                suffix = text[boundaries[ei] :].strip()
                context = prefix if len(prefix) > len(suffix) else suffix
                return TaskSpec(
                    task_type=TaskType.SPAN_RECONSTRUCTION,
                    prefix_text=context,
                    visual_text=visual,
                    target_text=visual,
                    blocks=blocks,
                )

        # Fallback: first sentence as visual
        if boundaries:
            visual = text[: boundaries[1]].strip()
            suffix = text[boundaries[1] :].strip()
            return TaskSpec(
                task_type=TaskType.SPAN_RECONSTRUCTION,
                prefix_text=suffix,
                visual_text=visual,
                target_text=visual,
                blocks=blocks,
            )
        return self._build_full_ocr(text, blocks)

    def _build_text_replay(self, text: str, blocks: list[dict[str, Any]]) -> TaskSpec:
        return TaskSpec(
            task_type=TaskType.TEXT_REPLAY,
            target_text=text,
            blocks=blocks,
            is_text_only=True,
        )


def _find_boundaries(text: str) -> list[int]:
    """Find safe boundary positions in text (after periods, newlines, etc.)."""
    boundaries = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            boundaries.append(i + 1)
        elif ch in ".!?。！？" and i + 1 < len(text):
            next_ch = text[i + 1] if i + 1 < len(text) else ""
            if next_ch in " \n":
                boundaries.append(i + 1)
    if boundaries[-1] < len(text):
        boundaries.append(len(text))
    return sorted(set(boundaries))
