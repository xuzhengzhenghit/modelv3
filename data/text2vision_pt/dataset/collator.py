#!/usr/bin/env python3
"""Collator — multiprocessing-safe collator for HainaOCR CPT format."""

from __future__ import annotations

from typing import Any

import torch


class Text2VisionCollator:
    """Collates rendered samples into model inputs.

    Produces:
        - input_ids: [B, L] — vision prefix + text
        - labels: [B, L] — -100 for vision prefix, target ids for text
        - pixel_values: [B, 3, H, W] uint8
        - image_grid_thw: [B, 3] — [t=1, grid_h, grid_w]
        - attention_mask: [B, L]
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
        batch_pixel_values = []
        batch_image_grid_thw = []
        batch_input_ids = []
        batch_labels = []

        for sample in samples:
            task = sample["task_spec"]
            canvas = sample["canvas_spec"]
            pixel_values = sample["pixel_values"]

            # Build sequence
            if task.is_text_only:
                input_ids, labels = self._build_text_only(task)
            elif task.task_type.value == "full_ocr":
                input_ids, labels = self._build_full_ocr(task, canvas)
            elif task.task_type.value == "optical_continuation":
                input_ids, labels = self._build_optical_continuation(task, canvas)
            elif task.task_type.value == "span_reconstruction":
                input_ids, labels = self._build_span_reconstruction(task, canvas)
            else:
                input_ids, labels = self._build_full_ocr(task, canvas)

            # Truncate if needed
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                labels = labels[: self.max_length]

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_pixel_values.append(pixel_values)
            batch_image_grid_thw.append([1, canvas.grid_h, canvas.grid_w])

        # Pad sequences
        padded_ids, padded_labels, attention_mask = self._pad_sequences(
            batch_input_ids, batch_labels
        )

        # Stack pixel values — handle dynamic shapes via per-sample tensors
        # For bucket batching, all samples in a batch have the same canvas size
        first_pv = batch_pixel_values[0]
        all_same_shape = all(pv.shape == first_pv.shape for pv in batch_pixel_values)

        if all_same_shape:
            stacked_pv = torch.stack(batch_pixel_values)
        else:
            # Pad pixel values to max H/W
            stacked_pv = _pad_and_stack_images(batch_pixel_values)

        return {
            "input_ids": padded_ids,
            "labels": padded_labels,
            "attention_mask": attention_mask,
            "pixel_values": stacked_pv,
            "image_grid_thw": torch.tensor(batch_image_grid_thw, dtype=torch.long),
        }

    def _build_full_ocr(self, task, canvas) -> tuple[list[int], list[int]]:
        """Build: <vision_start> <image_pad>×N <vision_end> target <eos>"""
        vision_prefix = [self.vision_start_id] + [self.image_pad_id] * canvas.visual_tokens + [self.vision_end_id]
        target_ids = self.tokenizer.encode(task.target_text, add_special_tokens=False)
        input_ids = vision_prefix + target_ids + [self.eos_id]
        labels = [-100] * len(vision_prefix) + target_ids + [self.eos_id]
        return input_ids, labels

    def _build_optical_continuation(self, task, canvas) -> tuple[list[int], list[int]]:
        """Build: prefix <vision_start> <image_pad>×N <vision_end> suffix <eos>"""
        vision_prefix = [self.vision_start_id] + [self.image_pad_id] * canvas.visual_tokens + [self.vision_end_id]
        prefix_ids = self.tokenizer.encode(task.prefix_text, add_special_tokens=False) if task.prefix_text else []
        suffix_ids = self.tokenizer.encode(task.suffix_text, add_special_tokens=False) if task.suffix_text else []
        input_ids = prefix_ids + vision_prefix + suffix_ids + [self.eos_id]
        labels = [-100] * (len(prefix_ids) + len(vision_prefix)) + suffix_ids + [self.eos_id]
        return input_ids, labels

    def _build_span_reconstruction(self, task, canvas) -> tuple[list[int], list[int]]:
        """Build: context <vision_start> <image_pad>×N <vision_end> visual_text <eos>"""
        vision_prefix = [self.vision_start_id] + [self.image_pad_id] * canvas.visual_tokens + [self.vision_end_id]
        context_ids = self.tokenizer.encode(task.prefix_text, add_special_tokens=False) if task.prefix_text else []
        target_ids = self.tokenizer.encode(task.target_text, add_special_tokens=False)
        input_ids = context_ids + vision_prefix + target_ids + [self.eos_id]
        labels = [-100] * (len(context_ids) + len(vision_prefix)) + target_ids + [self.eos_id]
        return input_ids, labels

    def _build_text_only(self, task) -> tuple[list[int], list[int]]:
        """Build: target <eos> — all tokens participate in loss."""
        target_ids = self.tokenizer.encode(task.target_text, add_special_tokens=False)
        input_ids = target_ids + [self.eos_id]
        labels = target_ids + [self.eos_id]
        return input_ids, labels

    def _pad_sequences(self, input_ids_list, labels_list):
        max_len = max(len(seq) for seq in input_ids_list)
        padded_input_ids = []
        padded_labels = []
        attention_mask = []

        for input_ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(input_ids + [self.pad_id] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)
            attention_mask.append([1] * len(input_ids) + [0] * pad_len)

        return (
            torch.tensor(padded_input_ids, dtype=torch.long),
            torch.tensor(padded_labels, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )


def _pad_and_stack_images(images: list[torch.Tensor]) -> torch.Tensor:
    """Pad a list of [C, H, W] tensors to the same H and W, then stack."""
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    c = images[0].shape[0]
    batch = torch.zeros((len(images), c, max_h, max_w), dtype=images[0].dtype)
    for i, img in enumerate(images):
        _, h, w = img.shape
        batch[i, :, :h, :w] = img
    return batch
