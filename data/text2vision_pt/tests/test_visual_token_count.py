#!/usr/bin/env python3
"""Test: visual token count must match between CNN output and input_ids."""

import pytest
import torch

# Add parent to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.visual_token_utils import (
    compute_visual_tokens,
    compute_grid,
    build_vision_prefix,
)
from model.dynamic_2d_position import validate_visual_token_alignment


def test_compute_visual_tokens():
    assert compute_visual_tokens(512, 1024) == 16 * 32  # 512
    assert compute_visual_tokens(256, 64) == 8 * 2  # 16
    assert compute_visual_tokens(768, 192) == 24 * 6  # 144


def test_build_vision_prefix():
    prefix = build_vision_prefix(151652, 151655, 151653, 512)
    assert len(prefix) == 514  # 1 + 512 + 1
    assert prefix[0] == 151652
    assert prefix[-1] == 151653
    assert all(t == 151655 for t in prefix[1:-1])


def test_validate_alignment_passes():
    cnn_output = torch.randn(512, 1024)
    input_ids = torch.tensor([[151655] * 512 + [1, 2, 3]])
    assert validate_visual_token_alignment(cnn_output, input_ids, 151655, 16, 32)


def test_validate_alignment_fails_cnn():
    cnn_output = torch.randn(511, 1024)
    input_ids = torch.tensor([[151655] * 512])
    with pytest.raises(ValueError):
        validate_visual_token_alignment(cnn_output, input_ids, 151655, 16, 32)


def test_validate_alignment_fails_ids():
    cnn_output = torch.randn(512, 1024)
    input_ids = torch.tensor([[151655] * 511])
    with pytest.raises(ValueError):
        validate_visual_token_alignment(cnn_output, input_ids, 151655, 16, 32)


def test_dynamic_2d_position():
    from model.dynamic_2d_position import Dynamic2DPositionEmbedding
    pos_embed = Dynamic2DPositionEmbedding(hidden_size=1024)
    # Test various grid sizes
    for h, w in [(2, 8), (4, 16), (8, 16), (16, 32), (6, 24), (2, 8)]:
        pos = pos_embed(h, w)
        assert pos.shape == (h * w, 1024)
