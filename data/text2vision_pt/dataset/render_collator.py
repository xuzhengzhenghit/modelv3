#!/usr/bin/env python3
"""RenderCollator — assembles multi-task rendered samples into training batches.

Task A (Full OCR):
    <vision_start> <image_pad>×N <vision_end> target_text <eos>
    labels: vision prefix = -100, target = loss

Task B (Optical Continuation):
    prefix_text <vision_start> <image_pad>×N <vision_end> suffix_text <eos>
    labels: prefix + vision prefix = -100, suffix = loss

Task D (Text Replay):
    target_text <eos>
    labels: all tokens = loss
"""

from __future__ import annotations

from typing import Any

import torch


class RenderCollator:
    """Collates RenderDataset outputs into model inputs."""

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
            task = s.get("task_type", "full_ocr")

            if s.get("is_text_only"):
                ids, labs = self._build_text_only(s)
            elif task == "optical_continuation":
                ids, labs = self._build_optical_continuation(s)
            else:
                ids, labs = self._build_full_ocr(s)

            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
                labs = labs[:self.max_length]

            batch_input_ids.append(ids)
            batch_labels.append(labs)

            if not s.get("is_text_only"):
                batch_pixel_values.append(s["pixel_values"])
                _, gh, gw = s["image_grid_thw"]
                batch_grid_thw.append([1, gh, gw])

        max_len = max(len(seq) for seq in batch_input_ids)
        input_ids, labels, attention_mask = [], [], []
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

    def _build_full_ocr(self, s: dict) -> tuple[list[int], list[int]]:
        # Task A: image → full text
        nt = s.get("num_visual_tokens", 512)
        vis_prefix = [self.vision_start_id] + [self.image_pad_id] * nt + [self.vision_end_id]
        tids = self.tokenizer.encode(s["target_text"], add_special_tokens=False)
        ids = vis_prefix + tids + [self.eos_id]
        labs = [-100] * len(vis_prefix) + tids + [self.eos_id]
        return ids, labs

    def _build_optical_continuation(self, s: dict) -> tuple[list[int], list[int]]:
        # Task B: prefix + image → predict suffix
        nt = s.get("num_visual_tokens", 512)
        vis_prefix = [self.vision_start_id] + [self.image_pad_id] * nt + [self.vision_end_id]

        prefix_text = s.get("prefix_text", "")
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False) if prefix_text else []

        suffix_ids = self.tokenizer.encode(s["target_text"], add_special_tokens=False)

        ids = prefix_ids + vis_prefix + suffix_ids + [self.eos_id]
        labs = [-100] * (len(prefix_ids) + len(vis_prefix)) + suffix_ids + [self.eos_id]
        return ids, labs

    def _build_text_only(self, s: dict) -> tuple[list[int], list[int]]:
        # Task D: pure text
        tids = self.tokenizer.encode(s["target_text"], add_special_tokens=False)
        ids = tids + [self.eos_id]
        labs = tids + [self.eos_id]
        return ids, labs


def _stack_images(images: list[torch.Tensor]) -> torch.Tensor:
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    c = images[0].shape[0]
    batch = torch.zeros((len(images), c, max_h, max_w), dtype=images[0].dtype)
    for i, img in enumerate(images):
        _, h, w = img.shape
        batch[i, :, :h, :w] = img
    return batch
