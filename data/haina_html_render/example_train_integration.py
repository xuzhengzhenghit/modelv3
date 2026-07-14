#!/usr/bin/env python3
"""Reference integration for HainaOCR-NativePixel PT.

This is intentionally a wiring example rather than a replacement for your training
loop. Insert the Dataset/Collator construction and the uint8->bf16 transfer into
train_haina_cpt.py.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from haina_cpt_collator import HainaCPTCollator
from html_ocr_dataset import HtmlRenderedOCRDataset, PreviewConfig
from html_ocr_renderer import BrowserConfig, HtmlOCRRenderer, RenderConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, action="append")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--vision-start-id", required=True, type=int)
    parser.add_argument("--image-pad-id", required=True, type=int)
    parser.add_argument("--vision-end-id", required=True, type=int)
    parser.add_argument("--browser-path", default=None)
    parser.add_argument("--katex-dist", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--preview-dir", default=None)
    return parser.parse_args()


def build_loader(args, rank: int = 0, world_size: int = 1):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id; configure it explicitly")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id")

    # Keep images uint8 in CPU workers. Convert after asynchronous GPU transfer.
    renderer = HtmlOCRRenderer(
        RenderConfig(
            width=1024,
            height=512,
            patch_size=32,
            output_mode="uint8",
        ),
        BrowserConfig(
            executable_path=args.browser_path,
            katex_dist=args.katex_dist,
        ),
    )

    dataset = HtmlRenderedOCRDataset(
        args.manifest,
        renderer,
        preview=PreviewConfig(
            directory=args.preview_dir,
            probability=0.0001 if args.preview_dir else 0.0,
            max_per_worker=20,
        ),
        rank=rank,
    )

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )

    collator = HainaCPTCollator(
        tokenizer=tokenizer,
        vision_start_id=args.vision_start_id,
        image_pad_id=args.image_pad_id,
        vision_end_id=args.vision_end_id,
        eos_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id,
        max_length=args.max_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )
    return dataset, sampler, loader


def training_loop_example(model, optimizer, loader, dataset, sampler, epochs: int, device: torch.device):
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in loader:
            # Important: uint8->bf16 conversion occurs after transfer, not in renderer workers.
            pixel_values = batch.pop("pixel_values").to(device, non_blocking=True)
            pixel_values = pixel_values.to(dtype=torch.bfloat16).div_(255.0)

            tensor_keys = ("input_ids", "labels", "attention_mask", "image_grid_thw", "img_sample_indices")
            for key in tensor_keys:
                if key in batch:
                    batch[key] = batch[key].to(device, non_blocking=True)
            batch["pixel_values"] = pixel_values

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def main() -> int:
    args = parse_args()
    dataset, sampler, loader = build_loader(args)
    dataset.set_epoch(0)
    batch = next(iter(loader))
    print("input_ids       :", tuple(batch["input_ids"].shape), batch["input_ids"].dtype)
    print("labels          :", tuple(batch["labels"].shape), batch["labels"].dtype)
    print("attention_mask  :", tuple(batch["attention_mask"].shape))
    print("pixel_values    :", tuple(batch["pixel_values"].shape), batch["pixel_values"].dtype)
    print("image_grid_thw  :", batch["image_grid_thw"])
    print("image_pad count :", int((batch["input_ids"][0] == args.image_pad_id).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
