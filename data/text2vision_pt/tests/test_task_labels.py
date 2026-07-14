#!/usr/bin/env python3
"""Test: verify collator produces correct labels for each task type."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import MagicMock
import torch

from tasks.task_sampler import TaskSpec, TaskType
from dataset.collator import Text2VisionCollator


def _make_mock_tokenizer():
    tok = MagicMock()
    tok.eos_token_id = 151645
    tok.pad_token_id = 151643

    def encode(text, **kwargs):
        if not text:
            return []
        return [ord(c) % 1000 for c in text[:50]]

    tok.encode = encode
    return tok


def _make_canvas_spec(grid_h=16, grid_w=32):
    return type("CanvasSpec", (), {"grid_h": grid_h, "grid_w": grid_w, "visual_tokens": grid_h * grid_w})()


def test_full_ocr_labels():
    collator = Text2VisionCollator(_make_mock_tokenizer())
    task = TaskSpec(
        task_type=TaskType.FULL_OCR,
        visual_text="Hello world",
        target_text="Hello world",
    )
    canvas = _make_canvas_spec(2, 2)  # 4 visual tokens

    sample = {
        "pixel_values": torch.zeros(3, 64, 128, dtype=torch.uint8),
        "task_spec": task,
        "canvas_spec": canvas,
    }
    batch = collator([sample])

    labels = batch["labels"][0]
    # First 6 tokens (1+4+1=vision prefix) should be -100
    assert labels[0] == -100  # vision_start
    assert labels[1] == -100  # image_pad
    assert labels[5] == -100  # vision_end
    # Last token (eos) should not be -100
    assert labels[-1] != -100


def test_text_replay_labels():
    collator = Text2VisionCollator(_make_mock_tokenizer())
    task = TaskSpec(
        task_type=TaskType.TEXT_REPLAY,
        target_text="Hello world",
        is_text_only=True,
    )
    canvas = _make_canvas_spec(2, 2)

    sample = {
        "pixel_values": torch.zeros(3, 64, 128, dtype=torch.uint8),
        "task_spec": task,
        "canvas_spec": canvas,
    }
    batch = collator([sample])

    labels = batch["labels"][0]
    # No -100 in text_only mode (all tokens participate)
    assert all(l != -100 for l in labels[: labels.tolist().index(collator.tokenizer.pad_token_id)] if collator.tokenizer.pad_token_id in labels)
