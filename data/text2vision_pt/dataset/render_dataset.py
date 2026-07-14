#!/usr/bin/env python3
"""RenderDataset — multi-task IterableDataset with online HTML rendering.

Supports: Full OCR (A), Optical Continuation (B), Text Replay (D).
Each DataLoader worker maintains a persistent Chromium + KaTeX instance.
"""

from __future__ import annotations

import glob as _glob
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

_T2V_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_T2V_DIR))

from rendering.html_ocr_renderer import (
    BrowserConfig, HtmlOCRRenderer, RenderConfig,
    RenderUnit, NeedsSplit, TooWide,
)
from tasks.task_sampler import TaskSampler, TaskType

logger = logging.getLogger(__name__)

# ── Default task weights (A: Full OCR, B: Optical Continuation, D: Text Replay) ──
DEFAULT_TASK_WEIGHTS = {
    "full_ocr": 0.60,
    "optical_continuation": 0.25,
    "text_replay": 0.15,
}

# ── Text-to-blocks parser ──
def _parse_text_to_blocks(text: str, subject: str | None = None) -> list[dict[str, Any]]:
    try:
        from preprocessing.document_parser import parse_document
        blocks = parse_document(text, subject, max_block_chars=420)
        for i, b in enumerate(blocks):
            b["id"] = f"b{i}"
        return blocks
    except Exception:
        return [{"id": "b0", "kind": "paragraph", "parts": [{"kind": "text", "text": text}]}]


class RenderDataset(IterableDataset):
    """Streams JSONL records → task-sampled → renders online → yields samples.

    Compatible with train_haina_cpt.py training loop.
    """

    def __init__(
        self,
        manifest_files: list[str],
        renderer: HtmlOCRRenderer | None = None,
        render_config: RenderConfig | None = None,
        browser_config: BrowserConfig | None = None,
        base_seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        max_samples: int = 0,
        task_weights: dict[str, float] | None = None,
    ):
        self._files = manifest_files
        self._renderer = renderer
        self._render_config = render_config
        self._browser_config = browser_config
        self._base_seed = base_seed
        self._rank = rank
        self._world_size = world_size
        self._max_samples = max_samples
        self._epoch = 0
        self._task_sampler = TaskSampler(weights=task_weights or DEFAULT_TASK_WEIGHTS)

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _get_renderer(self) -> HtmlOCRRenderer:
        worker = get_worker_info()
        wid = worker.id if worker else 0
        cache_attr = f"_renderer_w{wid}"
        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)
        if self._renderer is not None:
            r = self._renderer
        else:
            r = HtmlOCRRenderer(
                self._render_config or RenderConfig(output_mode="uint8"),
                self._browser_config or BrowserConfig(),
            )
        setattr(self, cache_attr, r)
        return r

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        wid = worker.id if worker else 0

        files = list(self._files)
        rng = random.Random(self._base_seed + self._epoch * 1000 + wid)
        rng.shuffle(files)

        renderer = self._get_renderer()
        count = 0

        for filepath in files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                rng.shuffle(lines)

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    sid = rec.get("i", rec.get("id", ""))
                    text = rec.get("t", rec.get("target_text", ""))
                    if not text or not sid:
                        continue

                    # ── Task sampling ──
                    task = self._task_sampler.sample_task(rec, self._epoch, self._base_seed)
                    is_text_only = task.is_text_only or task.task_type == TaskType.TEXT_REPLAY

                    # ── Text Replay: no rendering ──
                    if is_text_only:
                        yield {
                            "sample_id": sid,
                            "target_text": task.target_text,
                            "is_text_only": True,
                        }
                        count += 1
                        if self._max_samples > 0 and count >= self._max_samples:
                            return
                        continue

                    # ── Parse visual_text into blocks for rendering ──
                    subject = rec.get("s", rec.get("subject"))
                    blocks = _parse_text_to_blocks(task.visual_text, subject)
                    if not blocks:
                        continue

                    unit = RenderUnit(
                        sample_id=sid,
                        blocks=tuple(blocks),
                        target_text=task.target_text,
                        task_type=task.task_type.value,
                    )

                    seed = hash(sid) & 0x7FFFFFFF
                    try:
                        result = renderer.render_dynamic(unit, seed)
                    except (NeedsSplit, TooWide):
                        continue
                    except Exception:
                        continue

                    yield {
                        "sample_id": sid,
                        "pixel_values": result["pixel_values"],
                        "target_text": task.target_text,
                        "prefix_text": task.prefix_text,
                        "num_visual_tokens": result["num_visual_tokens"],
                        "image_grid_thw": result["image_grid_thw"],
                        "paper_size": result["paper_size"],
                        "task_type": task.task_type.value,
                    }

                    count += 1
                    if self._max_samples > 0 and count >= self._max_samples:
                        return
            except Exception:
                continue

    def __len__(self) -> int:
        if self._max_samples > 0:
            return self._max_samples
        return len(self._files) * 50000
