#!/usr/bin/env python3
"""CPT collator for HainaOCR NativePixel visual-prefix pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class HainaCPTCollator:
    tokenizer: Any
    vision_start_id: int
    image_pad_id: int
    vision_end_id: int
    eos_id: int
    pad_id: int
    max_length: int = 4096
    include_metadata: bool = False

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Empty batch")

        encoded_inputs: list[list[int]] = []
        encoded_labels: list[list[int]] = []
        target_texts: list[str] = []
        img_sample_indices: list[int] = []

        for idx, sample in enumerate(samples):
            is_text_only = bool(sample.get("is_text_only", False))

            if is_text_only:
                target = str(sample["target_text"])
                target_ids = self.tokenizer.encode(target, add_special_tokens=False)
                available = self.max_length - 1  # reserve 1 for eos
                target_ids = list(target_ids[:available])
                input_ids = target_ids + [self.eos_id]
                labels = target_ids + [self.eos_id]  # all tokens contribute to loss
            else:
                visual_count = int(sample["num_visual_tokens"])
                prefix = [self.vision_start_id] + [self.image_pad_id] * visual_count + [self.vision_end_id]
                available = self.max_length - len(prefix) - 1
                if available <= 0:
                    raise ValueError(
                        f"max_length={self.max_length} is too small for visual prefix of {len(prefix)} tokens"
                    )
                target = str(sample["target_text"])
                target_ids = self.tokenizer.encode(target, add_special_tokens=False)
                target_ids = list(target_ids[:available])
                input_ids = prefix + target_ids + [self.eos_id]
                labels = [-100] * len(prefix) + target_ids + [self.eos_id]
                img_sample_indices.append(idx)

            encoded_inputs.append(input_ids)
            encoded_labels.append(labels)
            target_texts.append(target)

        batch_length = max(len(x) for x in encoded_inputs)
        input_ids = torch.full((len(samples), batch_length), self.pad_id, dtype=torch.long)
        labels = torch.full((len(samples), batch_length), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(samples), batch_length), dtype=torch.long)

        for row, (ids, row_labels) in enumerate(zip(encoded_inputs, encoded_labels)):
            length = len(ids)
            input_ids[row, :length] = torch.tensor(ids, dtype=torch.long)
            labels[row, :length] = torch.tensor(row_labels, dtype=torch.long)
            attention_mask[row, :length] = 1

        # ── pixel_values: only for image samples ──
        if img_sample_indices:
            pixel_values = torch.stack([samples[i]["pixel_values"] for i in img_sample_indices])
            image_grid_thw = torch.stack([samples[i]["image_grid_thw"] for i in img_sample_indices])
        else:
            pixel_values = torch.empty(0)
            image_grid_thw = torch.empty(0)

        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "img_sample_indices": img_sample_indices,
        }
        if self.include_metadata:
            batch["target_texts"] = target_texts
            batch["sample_ids"] = [sample.get("id") for sample in samples]
            batch["render_meta"] = [sample.get("render_meta") for sample in samples]
        return batch
