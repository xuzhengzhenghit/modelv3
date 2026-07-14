#!/usr/bin/env python3
"""OCR image augmentation — apply scan/photograph style degradations."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


class ImageAugmenter:
    """Apply OCR-specific degradations to rendered text images.

    Degradations are applied directly to uint8 [0, 255] tensor images
    to simulate real-world scanning/photographing conditions.
    """

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def apply(self, image: torch.Tensor, difficulty: str = "clean") -> torch.Tensor:
        """Apply degradations based on difficulty level.

        Args:
            image: uint8 tensor [3, H, W] in range [0, 255].
            difficulty: "clean", "mild", or "hard".

        Returns:
            Augmented uint8 tensor [3, H, W].
        """
        if difficulty == "clean":
            return image  # no degradation

        # Work on float [0, 1] for easier math
        img = image.float() / 255.0

        if difficulty == "mild":
            img = self._maybe(self._gaussian_blur, img, p=0.3, sigma_max=0.8)
            img = self._maybe(self._jpeg_artifacts, img, p=0.2, quality_min=60)
            img = self._maybe(self._contrast_jitter, img, p=0.3, amount=0.08)
            img = self._maybe(self._brightness_jitter, img, p=0.2, amount=0.05)
            img = self._maybe(self._gaussian_noise, img, p=0.15, std=0.01)

        elif difficulty == "hard":
            img = self._maybe(self._gaussian_blur, img, p=0.6, sigma_max=1.5)
            img = self._maybe(self._motion_blur, img, p=0.2, kernel_max=7)
            img = self._maybe(self._jpeg_artifacts, img, p=0.5, quality_min=40)
            img = self._maybe(self._contrast_jitter, img, p=0.5, amount=0.15)
            img = self._maybe(self._brightness_jitter, img, p=0.4, amount=0.10)
            img = self._maybe(self._gaussian_noise, img, p=0.4, std=0.02)
            img = self._maybe(self._shadow, img, p=0.15)
            img = self._maybe(self._rotation, img, p=0.2, max_deg=2.0)

        # Clamp and convert back to uint8
        img = torch.clamp(img, 0.0, 1.0)
        return (img * 255.0).to(torch.uint8)

    def _maybe(self, fn, img, p, **kwargs):
        if self._rng.random() < p:
            return fn(img, **kwargs)
        return img

    @staticmethod
    def _gaussian_blur(img: torch.Tensor, sigma_max: float = 1.0) -> torch.Tensor:
        if sigma_max <= 0:
            return img
        import torch.nn.functional as F
        sigma = random.uniform(0.3, sigma_max)
        kernel_size = max(3, int(2 * int(3 * sigma) + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        # Simple approximation using average pooling as fallback
        if sigma < 0.6:
            return img
        # Use a small box blur approximation
        c, h, w = img.shape
        img_4d = img.unsqueeze(0)
        k = min(kernel_size, 7)
        pad = k // 2
        blurred = F.avg_pool2d(img_4d, kernel_size=k, stride=1, padding=pad)
        alpha = min(1.0, sigma / sigma_max)
        result = (1 - alpha) * img_4d + alpha * blurred
        return result.squeeze(0)

    @staticmethod
    def _motion_blur(img: torch.Tensor, kernel_max: int = 7) -> torch.Tensor:
        import torch.nn.functional as F
        k = random.randrange(3, kernel_max + 1, 2)
        c, h, w = img.shape
        img_4d = img.unsqueeze(0)
        # Horizontal motion blur
        pad = k // 2
        if random.random() < 0.5:
            blurred = F.avg_pool2d(img_4d, kernel_size=(1, k), stride=1, padding=(0, pad))
        else:
            blurred = F.avg_pool2d(img_4d, kernel_size=(k, 1), stride=1, padding=(pad, 0))
        return blurred.squeeze(0)

    @staticmethod
    def _jpeg_artifacts(img: torch.Tensor, quality_min: int = 40) -> torch.Tensor:
        quality = random.randint(quality_min, 85)
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        from io import BytesIO
        from PIL import Image
        pil_img = Image.fromarray(img_np)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        restored = Image.open(buf)
        restored_np = np.array(restored).astype(np.float32) / 255.0
        return torch.from_numpy(restored_np).permute(2, 0, 1)

    @staticmethod
    def _contrast_jitter(img: torch.Tensor, amount: float = 0.1) -> torch.Tensor:
        factor = 1.0 + random.uniform(-amount, amount)
        mean = img.mean(dim=(1, 2), keepdim=True)
        return torch.clamp((img - mean) * factor + mean, 0.0, 1.0)

    @staticmethod
    def _brightness_jitter(img: torch.Tensor, amount: float = 0.05) -> torch.Tensor:
        delta = random.uniform(-amount, amount)
        return torch.clamp(img + delta, 0.0, 1.0)

    @staticmethod
    def _gaussian_noise(img: torch.Tensor, std: float = 0.01) -> torch.Tensor:
        noise = torch.randn_like(img) * std
        return torch.clamp(img + noise, 0.0, 1.0)

    @staticmethod
    def _shadow(img: torch.Tensor, p: float = 0.15) -> torch.Tensor:
        """Simulate page shadow / vignette."""
        c, h, w = img.shape
        y = torch.linspace(0, 1, h, device=img.device).view(1, h, 1)
        x = torch.linspace(0, 1, w, device=img.device).view(1, 1, w)
        # Darker at edges
        vignette = 1.0 - 0.15 * ((y - 0.5)**2 + (x - 0.5)**2) * 4
        vignette = torch.clamp(vignette, 0.85, 1.0)
        return torch.clamp(img * vignette, 0.0, 1.0)

    @staticmethod
    def _rotation(img: torch.Tensor, max_deg: float = 2.0) -> torch.Tensor:
        angle = random.uniform(-max_deg, max_deg)
        import torchvision.transforms.functional as TF
        return TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
