#!/usr/bin/env python3
"""Text2Vision-PT Dataset — IterableDataset with online rendering and dynamic canvas."""

from __future__ import annotations

import glob
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

# ── Dynamic resys.path for submodules ──
T2V_DIR = Path(__file__).resolve().parent.parent
if str(T2V_DIR) not in sys.path:
    sys.path.insert(0, str(T2V_DIR))

from rendering.style_sampler import RenderSpec, StyleSampler
from rendering.dynamic_canvas import CanvasSpec, DynamicCanvas
from rendering.html_builder import build_page_html, build_measure_html
from rendering.browser_renderer import BrowserConfig, HtmlOCRRenderer
from rendering.image_augment import ImageAugmenter
from tasks.task_sampler import TaskSampler, TaskSpec, TaskType


class Text2VisionDataset(IterableDataset):
    """Online-rendering dataset for Text2Vision-PT.

    Each worker maintains a persistent Chromium instance. Samples are streamed
    from JSONL shards, assigned tasks, rendered with dynamic canvas sizing,
    and returned as (pixel_values, task_spec) tuples.
    """

    def __init__(
        self,
        manifest_pattern: str,
        browser_cfg: BrowserConfig,
        style_sampler: StyleSampler | None = None,
        task_sampler: TaskSampler | None = None,
        canvas: DynamicCanvas | None = None,
        augmenter: ImageAugmenter | None = None,
        global_seed: int = 42,
        epoch: int = 0,
        max_samples: int = 0,
    ):
        self._manifest_pattern = manifest_pattern
        self._browser_cfg = browser_cfg
        self._style_sampler = style_sampler or StyleSampler()
        self._task_sampler = task_sampler or TaskSampler()
        self._canvas = canvas or DynamicCanvas()
        self._augmenter = augmenter or ImageAugmenter()
        self._global_seed = global_seed
        self._epoch = epoch
        self._max_samples = max_samples

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _worker_renderer(self) -> HtmlOCRRenderer:
        """Get or create per-worker renderer."""
        worker = get_worker_info()
        worker_id = worker.id if worker else 0

        # Cache on the worker process
        cache_attr = f"_renderer_{worker_id}"
        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)

        renderer = HtmlOCRRenderer(self._browser_cfg)
        setattr(self, cache_attr, renderer)
        return renderer

    def _list_files(self) -> list[str]:
        return sorted(glob.glob(self._manifest_pattern))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1

        files = self._list_files()
        if not files:
            return

        # Assign files to this worker
        worker_files = [f for i, f in enumerate(files) if i % num_workers == worker_id]
        if not worker_files:
            worker_files = files

        # Shuffle files
        seed_files = self._global_seed + self._epoch * 1000 + worker_id * 100
        rng = random.Random(seed_files)
        rng.shuffle(worker_files)

        renderer = self._worker_renderer()
        count = 0

        for filepath in worker_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # Shuffle lines within file
                seed_lines = self._global_seed + self._epoch * 1000 + worker_id * 100 + hash(filepath) % 10000
                rng = random.Random(seed_lines)
                rng.shuffle(lines)

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    result = self._process_sample(record, renderer)
                    if result is not None:
                        yield result
                        count += 1
                        if self._max_samples > 0 and count >= self._max_samples:
                            return
            except Exception:
                continue

    def _process_sample(self, record: dict[str, Any], renderer: HtmlOCRRenderer) -> dict[str, Any] | None:
        """Process one record: sample task + style → measure → render → augment."""
        sample_id = record.get("i", record.get("id", ""))
        text = record.get("t", record.get("target_text", ""))

        if not text or not sample_id:
            return None

        # 1. Sample task
        task = self._task_sampler.sample_task(record, self._epoch, self._global_seed)

        # 2. Sample style
        spec = self._style_sampler.sample(sample_id, self._epoch, self._global_seed)

        # 3. Build measure HTML and render it to get content dimensions
        blocks = task.blocks if task.blocks else _text_to_blocks(task.visual_text)
        measure_html = build_measure_html(
            blocks, spec, renderer._katex_css, renderer._katex_js
        )

        # 4. Measure content
        measure = renderer.measure_content(measure_html)
        content_w = measure.get("width", 600)
        content_h = measure.get("height", 200)

        # 5. Compute canvas
        canvas_spec = self._canvas.compute(
            content_width=content_w,
            content_height=content_h,
            top_margin=spec.top_margin,
            bottom_margin=spec.bottom_margin,
            left_margin=spec.left_margin,
            right_margin=spec.right_margin,
        )

        # 6. Build and render final HTML
        page_html = build_page_html(
            blocks, spec, canvas_spec.width, canvas_spec.height,
            renderer._katex_css, renderer._katex_js,
        )
        render_result = renderer.render(page_html, canvas_spec.width, canvas_spec.height)

        if not render_result.success:
            return None

        # 7. Apply augmentations
        if render_result.pixel_values is not None:
            pixel_values = self._augmenter.apply(
                render_result.pixel_values, spec.difficulty
            )
        else:
            return None

        # 8. Return sample dict
        return {
            "sample_id": sample_id,
            "pixel_values": pixel_values,
            "task_spec": task,
            "render_spec": spec,
            "canvas_spec": canvas_spec,
            "metadata": {
                "overflow": render_result.overflow,
                "kaTeX_errors": render_result.kaTeX_errors,
                "difficulty": spec.difficulty,
                "layout": spec.layout_style,
                "font_size": spec.font_size,
                "content_w": content_w,
                "content_h": content_h,
            },
        }

    def __len__(self) -> int:
        if self._max_samples > 0:
            return self._max_samples
        # Return a rough estimate
        try:
            files = self._list_files()
            return len(files) * 50000  # rough: ~50K samples per shard
        except Exception:
            return 1000000


def _text_to_blocks(text: str) -> list[dict[str, Any]]:
    """Simple fallback: wrap plain text in a paragraph block."""
    return [{"kind": "paragraph", "parts": [{"kind": "text", "text": text}]}]
