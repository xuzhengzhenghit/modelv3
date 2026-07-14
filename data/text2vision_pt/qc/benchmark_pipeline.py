#!/usr/bin/env python3
"""Benchmark the Text2Vision-PT pipeline end-to-end."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
T2V_DIR = SCRIPT_DIR.parent
if str(T2V_DIR) not in sys.path:
    sys.path.insert(0, str(T2V_DIR))

from rendering.browser_renderer import BrowserConfig, HtmlOCRRenderer
from rendering.style_sampler import StyleSampler
from rendering.dynamic_canvas import DynamicCanvas
from rendering.html_builder import build_page_html, build_measure_html
from tasks.task_sampler import TaskSampler


def benchmark(
    manifest_path: str,
    browser_path: str,
    katex_dist: str,
    num_samples: int = 100,
    warmup: int = 10,
):
    """Run pipeline benchmark and print statistics."""

    print("=" * 60)
    print("Text2Vision-PT Pipeline Benchmark")
    print(f"  samples: {num_samples}  warmup: {warmup}")
    print("=" * 60)

    browser_cfg = BrowserConfig(executable_path=browser_path, katex_dist=katex_dist)
    renderer = HtmlOCRRenderer(browser_cfg)
    style_sampler = StyleSampler()
    task_sampler = TaskSampler()
    canvas = DynamicCanvas()

    # Load samples
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print("[ERROR] No valid records found")
        return

    # Recycle records if not enough
    while len(records) < num_samples + warmup:
        records.extend(records)

    task_times: list[float] = []
    style_times: list[float] = []
    measure_times: list[float] = []
    canvas_times: list[float] = []
    build_times: list[float] = []
    render_times: list[float] = []
    total_times: list[float] = []

    overflow_count = 0
    error_count = 0
    visual_token_counts: list[int] = []
    sizes: list[tuple[int, int]] = []

    for idx, record in enumerate(records):
        if idx >= num_samples + warmup:
            break

        t0 = time.perf_counter()

        sample_id = record.get("i", record.get("id", str(idx)))
        text = record.get("t", record.get("target_text", ""))

        # Task sampling
        t1 = time.perf_counter()
        task = task_sampler.sample_task(record, 0)
        t2 = time.perf_counter()

        # Style sampling
        spec = style_sampler.sample(sample_id, 0)
        t3 = time.perf_counter()

        blocks = task.blocks if task.blocks else [{"kind": "paragraph", "parts": [{"kind": "text", "text": task.visual_text}]}]

        # Build measure HTML
        measure_html = build_measure_html(blocks, spec, renderer._katex_css, renderer._katex_js)
        t4 = time.perf_counter()

        # Measure content
        measure = renderer.measure_content(measure_html)
        t5 = time.perf_counter()

        # Compute canvas
        canvas_spec = canvas.compute(
            content_width=measure.get("width", 600),
            content_height=measure.get("height", 200),
            top_margin=spec.top_margin,
            bottom_margin=spec.bottom_margin,
            left_margin=spec.left_margin,
            right_margin=spec.right_margin,
        )
        t6 = time.perf_counter()

        # Build final HTML
        page_html = build_page_html(
            blocks, spec, canvas_spec.width, canvas_spec.height,
            renderer._katex_css, renderer._katex_js,
        )
        t7 = time.perf_counter()

        # Render
        result = renderer.render(page_html, canvas_spec.width, canvas_spec.height)
        t8 = time.perf_counter()

        if idx >= warmup:
            task_times.append((t2 - t1) * 1000)
            style_times.append((t3 - t2) * 1000)
            measure_times.append((t5 - t4) * 1000)
            canvas_times.append((t6 - t5) * 1000)
            build_times.append((t7 - t6) * 1000)
            render_times.append((t8 - t7) * 1000)
            total_times.append((t8 - t0) * 1000)

            visual_token_counts.append(canvas_spec.visual_tokens)
            sizes.append((canvas_spec.width, canvas_spec.height))

            if canvas_spec.overflow:
                overflow_count += 1
            if not result.success:
                error_count += 1

    renderer.shutdown()

    # Print results
    print(f"\n{'Stage':<25} {'Mean':>8} {'P50':>8} {'P95':>8}")
    print("-" * 52)
    for name, times in [
        ("task_sampling", task_times),
        ("style_sampling", style_times),
        ("measure_content", measure_times),
        ("canvas_compute", canvas_times),
        ("html_build", build_times),
        ("render_screenshot", render_times),
    ]:
        if times:
            print(f"{name:<25} {statistics.mean(times):7.1f}ms {statistics.median(times):7.1f}ms "
                  f"{_p95(times):7.1f}ms")

    print("-" * 52)
    if total_times:
        print(f"{'TOTAL':<25} {statistics.mean(total_times):7.1f}ms {statistics.median(total_times):7.1f}ms "
              f"{_p95(total_times):7.1f}ms")

    # Throughput
    if total_times:
        throughput = 1000 / statistics.mean(total_times)
        print(f"\nThroughput: {throughput:.2f} samples/s")

    # Visual token stats
    if visual_token_counts:
        print(f"\nVisual Tokens: mean={statistics.mean(visual_token_counts):.0f} "
              f"min={min(visual_token_counts)} max={max(visual_token_counts)}")

    # Size distribution
    if sizes:
        size_counter = {}
        for w, h in sizes:
            key = f"{w}×{h}"
            size_counter[key] = size_counter.get(key, 0) + 1
        print(f"Top sizes: {dict(sorted(size_counter.items(), key=lambda x: -x[1])[:5])}")

    print(f"\nOverflows: {overflow_count}  Errors: {error_count}")
    print(f"Visual tokens/s: {statistics.mean(visual_token_counts) * throughput:.0f}")


def _p95(data: list[float]) -> float:
    sor = sorted(data)
    idx = int(len(sor) * 0.95)
    return sor[min(idx, len(sor) - 1)]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Text2Vision-PT pipeline")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--browser-path", default="/root/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome")
    parser.add_argument("--katex-dist", default="/mnt/si001719bp3c/default/XJZ/modelv3/data/haina_html_render/node_modules/katex/dist")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    benchmark(
        manifest_path=args.manifest,
        browser_path=args.browser_path,
        katex_dist=args.katex_dist,
        num_samples=args.count,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
