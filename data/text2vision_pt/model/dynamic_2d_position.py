#!/usr/bin/env python3
"""Dynamic 2D positional embedding — resolution-independent position encoding for visual tokens.

Produces position embeddings for arbitrary (H, W) visual token grids, enabling
the model to handle dynamic visual token counts instead of a fixed 512.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class Dynamic2DPositionEmbedding(nn.Module):
    """Resolution-independent 2D coordinate MLP positional embedding.

    For each visual token at grid position (x, y), computes:
        pos_{x,y} = MLP[x/(W-1), y/(H-1), log(H), log(W)]

    This is a drop-in upgrade from the existing PositionEmbedding2D that
    supports arbitrary grid sizes without retraining.
    """

    def __init__(self, hidden_size: int, use_log_scale: bool = True):
        super().__init__()
        input_dim = 4 if use_log_scale else 2
        self.use_log_scale = use_log_scale
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        grid_h: int,
        grid_w: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Generate 2D position embeddings for a grid of shape (grid_h, grid_w).

        Args:
            grid_h: Height of the visual token grid.
            grid_w: Width of the visual token grid.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Tensor of shape [grid_h * grid_w, hidden_size]
        """
        y = torch.arange(grid_h, device=device, dtype=torch.float32)
        x = torch.arange(grid_w, device=device, dtype=torch.float32)

        # Normalize to [0, 1]
        y_norm = y / max(grid_h - 1, 1)
        x_norm = x / max(grid_w - 1, 1)

        yy, xx = torch.meshgrid(y_norm, x_norm, indexing="ij")

        if self.use_log_scale:
            log_h = torch.full_like(yy, math.log(max(grid_h, 1)))
            log_w = torch.full_like(xx, math.log(max(grid_w, 1)))
            coords = torch.stack([
                xx.reshape(-1),
                yy.reshape(-1),
                log_h.reshape(-1),
                log_w.reshape(-1),
            ], dim=-1).to(dtype)
        else:
            coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).to(dtype)

        return self.mlp(coords)


class RowColumnPositionEmbedding(nn.Module):
    """Simpler alternative: separate row + column learned embeddings.

    Embeds each grid position as:
        pos_{i,j} = row_embed[i] + col_embed[j]
    """

    def __init__(self, hidden_size: int, max_grid_size: int = 64):
        super().__init__()
        self.row_embed = nn.Embedding(max_grid_size, hidden_size)
        self.col_embed = nn.Embedding(max_grid_size, hidden_size)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.row_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.col_embed.weight, std=0.02)

    def forward(self, grid_h: int, grid_w: int, device=None, dtype=None) -> torch.Tensor:
        rows = self.row_embed(torch.arange(grid_h, device=device))
        cols = self.col_embed(torch.arange(grid_w, device=device))
        # [grid_h, hidden] + [grid_w, hidden] → [grid_h, grid_w, hidden]
        pos = rows.unsqueeze(1) + cols.unsqueeze(0)
        return pos.reshape(grid_h * grid_w, -1).to(dtype)


def validate_visual_token_alignment(
    cnn_output: torch.Tensor,
    input_ids: torch.Tensor,
    image_pad_id: int,
    grid_h: int,
    grid_w: int,
) -> bool:
    """Hard assertion: visual token count in CNN output must match input_ids placeholders.

    Args:
        cnn_output: [B, N, D] or [N, D] from CNN.
        input_ids: [B, L] token IDs.
        image_pad_id: Token ID for <image_pad>.
        grid_h, grid_w: Expected grid dimensions.

    Returns:
        True if alignment is correct.

    Raises:
        ValueError: If visual token count mismatch is detected.
    """
    expected = grid_h * grid_w
    actual_cnn = cnn_output.shape[0] if cnn_output.ndim == 2 else cnn_output.shape[1]
    actual_ids = (input_ids == image_pad_id).sum().item()

    if actual_cnn != expected:
        raise ValueError(
            f"CNN visual token count mismatch: expected={expected}, got={actual_cnn}"
        )
    if actual_ids != expected:
        raise ValueError(
            f"input_ids image_pad count mismatch: expected={expected}, got={actual_ids}"
        )
    return True
