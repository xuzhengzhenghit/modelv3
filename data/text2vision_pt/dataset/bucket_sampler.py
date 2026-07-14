#!/usr/bin/env python3
"""Bucket batch sampler — groups samples with similar grid shapes into same batch.

When using dynamic canvas sizing, different samples produce different visual token
grids. Direct stacking fails. This sampler groups samples into buckets by grid shape
so all samples in a batch have identical (H, W).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


class BucketBatchSampler:
    """Reorders samples so each batch has uniform (grid_h, grid_w).

    Usage as a wrapper around the dataset iterator:
        sampler = BucketBatchSampler(batch_size=4)
        for batch in sampler.collate(dataset):
            ...
    """

    def __init__(self, batch_size: int = 4, drop_last: bool = False):
        self.batch_size = batch_size
        self.drop_last = drop_last

    def group_and_yield(self, samples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Group samples into batches by (grid_h, grid_w).

        Args:
            samples: List of raw dataset outputs.

        Returns:
            List of batches, each batch is a list of samples.
        """
        # Group by grid shape
        buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            canvas = sample.get("canvas_spec")
            if canvas is None:
                continue
            key = (canvas.grid_h, canvas.grid_w)
            buckets[key].append(sample)

        batches: list[list[dict[str, Any]]] = []
        for samples_in_bucket in buckets.values():
            random.shuffle(samples_in_bucket)
            for i in range(0, len(samples_in_bucket), self.batch_size):
                batch = samples_in_bucket[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        random.shuffle(batches)
        return batches
