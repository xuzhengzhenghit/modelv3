#!/usr/bin/env python3
"""RenderCollator — assembles rendered samples into training batches.

Compatible with train_haina_cpt.py — same output format as HainaCPTCollator.
Supports image samples (full OCR) and text-only samples (text mixing).
"""

from __future__ import annotations

from typing import Any

import torch


class RenderCollator:
    """Collates RenderDataset outputs into model inputs.

    Image sample:
        <vision_start> <image_pad>×N <vision_end> target_text <eos>
        labels: vision prefix = -100, target = loss

    Text-only sample:
        target_text <eos>
        labels: all tokens = loss
    """

    def __init__(
        self,
        tokenizer,
        vision_start_id: int = 151652,
        image_pad_id: int = 151655,
        vision_end_id: int = 151653,
        eos_id: int | None = None,
        pad_id: int | None = None,
        max_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.vision_start_id = vision_start_id
        self.image_pad_id = image_pad_id
        self.vision_end_id = vision_end_id
        self.eos_id = eos_id if eos_id is not None else tokenizer.eos_token_id
        self.pad_id = pad_id if pad_id is not None else tokenizer.pad_token_id
        self.max_length = max_length

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch_input_ids, batch_labels = [], []
        batch_pixel_values, batch_grid_thw = [], []

        for s in samples:
            if s.get("is_text_only"):
                ids, labs = self._build_text_only(s)
            else:
                ids, labs = self._build_image(s)

            # Truncate if needed
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
                labs = labs[:self.max_length]

            batch_input_ids.append(ids)
            batch_labels.append(labs)

            if not s.get("is_text_only"):
                batch_pixel_values.append(s["pixel_values"])
                _, gh, gw = s["image_grid_thw"]
                batch_grid_thw.append([1, gh, gw])

        # Pad
        max_len = max(len(seq) for seq in batch_input_ids)
        input_ids = []
        labels = []
        attention_mask = []
        for ids, labs in zip(batch_input_ids, batch_labels):
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad_len)
            labels.append(labs + [-100] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        if batch_pixel_values:
            out["pixel_values"] = _stack_images(batch_pixel_values)
            out["image_grid_thw"] = torch.tensor(batch_grid_thw, dtype=torch.long)

        return out

    def _build_image(self, s):
        nt = s.get("num_visual_tokens", 512)
        prefix = [self.vision_start_id] + [self.image_pad_id] * nt + [self.vision_end_id]
        target_ids = self.tokenizer.encode(s["target_text"], add_special_tokens=False)
        input_ids = prefix + target_ids + [self.eos_id]
        labels = [-100] * len(prefix) + target_ids + [self.eos_id]
        return input_ids, labels

    def _build_text_only(self, s):
        target_ids = self.tokenizer.encode(s["target_text"], add_special_tokens=False)
        input_ids = target_ids + [self.eos_id]
        labels = target_ids + [self.eos_id]
        return input_ids, labels


def _stack_images(images: list[torch.Tensor]) -> torch.Tensor:
    """Stack images of potentially different sizes by padding to max H,W."""
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    c = images[0].shape[0]
    batch = torch.zeros((len(images), c, max_h, max_w), dtype=images[0].dtype)
    for i, img in enumerate(images):
        _, h, w = img.shape
        batch[i, :, :h, :w] = img
    return batch
