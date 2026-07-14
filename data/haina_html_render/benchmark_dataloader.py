#!/usr/bin/env python3
"""Measure end-to-end online-render throughput with PyTorch DataLoader workers."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from html_ocr_dataset import HtmlRenderedOCRDataset, PreviewConfig
from html_ocr_renderer import BrowserConfig, HtmlOCRRenderer, RenderConfig


def collate_render_only(samples):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in samples]),
        "target_text": [x["target_text"] for x in samples],
        "render_meta": [x["render_meta"] for x in samples],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, action="append")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--browser-path", default=None)
    parser.add_argument("--katex-dist", default=None)
    parser.add_argument("--output-mode", choices=["uint8", "float01", "float11"], default="uint8")
    parser.add_argument("--preview-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    renderer = HtmlOCRRenderer(
        RenderConfig(output_mode=args.output_mode),
        BrowserConfig(executable_path=args.browser_path, katex_dist=args.katex_dist),
    )
    dataset = HtmlRenderedOCRDataset(
        args.manifest,
        renderer,
        preview=PreviewConfig(args.preview_dir, probability=0.02 if args.preview_dir else 0.0, max_per_worker=3),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
        pin_memory=True,
        collate_fn=collate_render_only,
        drop_last=True,
    )

    iterator = iter(loader)
    print("Warming up DataLoader/browser workers...")
    for _ in range(args.warmup_batches):
        next(iterator)

    batch_times = []
    sample_count = 0
    wall_start = time.perf_counter()
    for batch_index in range(args.batches):
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        elapsed = time.perf_counter() - start
        batch_times.append(elapsed)
        sample_count += len(batch["target_text"])
        if batch_index == 0 or (batch_index + 1) % 10 == 0:
            recent = batch_times[-10:]
            print(
                f"[{batch_index + 1:4d}/{args.batches}] shape={tuple(batch['pixel_values'].shape)} "
                f"batch_wait={elapsed * 1000:7.2f} ms recent={args.batch_size / statistics.fmean(recent):6.2f} pages/s"
            )

    wall = time.perf_counter() - wall_start
    print("\n========== dataloader summary ==========")
    print(f"workers           : {args.workers}")
    print(f"batch size        : {args.batch_size}")
    print(f"samples           : {sample_count}")
    print(f"wall time         : {wall:.3f} s")
    print(f"throughput        : {sample_count / wall:.2f} pages/s")
    print(f"mean batch wait   : {statistics.fmean(batch_times) * 1000:.2f} ms")
    print(f"median batch wait : {statistics.median(batch_times) * 1000:.2f} ms")
    print("========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
