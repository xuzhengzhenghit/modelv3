#!/usr/bin/env python3
"""Overfitting test: text → HTML render → CPT training → verify loss drops to near zero.

Pipeline:
  100-page manifest → HtmlRenderedOCRDataset → HainaCPTCollator → random init model → train
"""

from __future__ import annotations

import math, sys, time
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

# ── Project paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # modelv3/
sys.path.insert(0, str(PROJECT_ROOT))

from hainaocr_nativepixel import (
    HainaOCRNativePixelConfig,
    HainaOCRNativePixelForConditionalGeneration,
)
from transformers import AutoTokenizer

from html_ocr_dataset import HtmlRenderedOCRDataset
from html_ocr_renderer import BrowserConfig, HtmlOCRRenderer, RenderConfig
from haina_cpt_collator import HainaCPTCollator

# ── Config ──
MANIFEST_PATH = "/tmp/overfit_100.jsonl"
OUTPUT_DIR = Path("/tmp/overfit_test_output")
BROWSER = "/root/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome"
KATEX = str(SCRIPT_DIR / "node_modules" / "katex" / "dist")
MODEL_DIR = PROJECT_ROOT / "hainaocr_nativepixel"
LLM_PATH = "/mnt/si001719kd1w/default/xjz/model/qwen3_0_6b"  # pretrained LLM + random CNN

BATCH_SIZE = 2
GRAD_ACCUM = 1
LR = 1e-3
MAX_STEPS = 500
LOG_INTERVAL = 10
MAX_LENGTH = 2048


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── Renderer + Dataset ──
    renderer = HtmlOCRRenderer(
        RenderConfig(output_mode="uint8"),
        BrowserConfig(executable_path=BROWSER, katex_dist=KATEX),
    )
    dataset = HtmlRenderedOCRDataset([MANIFEST_PATH], renderer)
    print(f"[data] {len(dataset)} samples")

    # ── Tokenizer ──
    tokenizer_path = "/mnt/si001719kd1w/default/xjz/model/qwen3_0_6b"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[tokenizer] pad_token_id={tokenizer.pad_token_id} eos_token_id={tokenizer.eos_token_id}")

    # ── Collator ──
    VISION_START, IMAGE_PAD, VISION_END = 151652, 151655, 151653
    collator = HainaCPTCollator(
        tokenizer=tokenizer,
        vision_start_id=VISION_START,
        image_pad_id=IMAGE_PAD,
        vision_end_id=VISION_END,
        eos_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id,
        max_length=MAX_LENGTH,
    )

    # ── DataLoader ──
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=collator,
    )

    # ── Model (random init) ──
    print("[model] building with pretrained LLM + random CNN...")
    config = HainaOCRNativePixelConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    config.use_liger_ce = False  # avoid liger kernel issue
    model = HainaOCRNativePixelForConditionalGeneration(config)
    # Load pretrained Qwen3-0.6B weights, CNN stays random
    if LLM_PATH and Path(LLM_PATH).exists():
        model.load_pretrained_components(LLM_PATH, dtype=torch.bfloat16)
        print(f"[model] loaded Qwen3-0.6B from {LLM_PATH}")
    # Train all params
    for p in model.parameters():
        p.requires_grad = True
    model = model.to(device=device, dtype=torch.bfloat16)
    model.train()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] total={total:,} trainable={trainable:,}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
    print(f"[optimizer] lr={LR}")

    # ── Training loop ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[train] max_steps={MAX_STEPS} log_interval={LOG_INTERVAL}")
    print("=" * 60)

    global_step = 0
    epoch = 0
    accum_loss = 0.0
    accum_count = 0

    while global_step < MAX_STEPS:
        epoch += 1
        dataset.set_epoch(epoch)
        for batch in loader:
            if global_step >= MAX_STEPS:
                break

            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16).div_(255.0)

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                use_cache=False,
            )
            loss = out.loss / GRAD_ACCUM
            loss.backward()

            accum_loss += float(loss.detach().cpu()) * GRAD_ACCUM
            accum_count += 1

            # optimizer step
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % LOG_INTERVAL == 0:
                avg = accum_loss / max(1, accum_count)
                print(f"[step {global_step:4d}/{MAX_STEPS}] loss={avg:.5f} grad_norm={grad_norm:.3f}")
                accum_loss = 0.0
                accum_count = 0

    print("=" * 60)
    print("[done] overfitting test complete")

    # ── Final loss on a few samples ──
    print("\n[final eval]")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 5:
                break
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16).div_(255.0)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                use_cache=False,
            )
            print(f"  eval loss={out.loss.item():.5f}")

    # ── Save ──
    from safetensors.torch import save_file
    state = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, str(OUTPUT_DIR / "overfit_model.safetensors"))
    print(f"\n[saved] {OUTPUT_DIR / 'overfit_model.safetensors'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
