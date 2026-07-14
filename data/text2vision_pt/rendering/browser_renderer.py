#!/usr/bin/env python3
"""Chromium renderer — persistent browser + KaTeX for online screenshot generation."""

from __future__ import annotations

import ctypes
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# Suppress playwright warnings
warnings.filterwarnings("ignore", message=".*URLError.*")

# ── Shared library for PIL access ──
_ctypes_cache: dict[str, Any] = {}


def _ensure_ctypes():
    """Load required ctypes libraries once."""
    if "cdll" not in _ctypes_cache:
        _ctypes_cache["cdll"] = ctypes.CDLL
    return _ctypes_cache


@dataclass
class BrowserConfig:
    executable_path: str = ""
    katex_dist: str = ""
    headless: bool = True
    timeout_ms: int = 5000
    block_external_urls: bool = True
    block_images: bool = True


@dataclass
class RenderResult:
    png_bytes: bytes = b""
    pixel_values: Optional[torch.Tensor] = None
    canvas_width: int = 0
    canvas_height: int = 0
    grid_w: int = 0
    grid_h: int = 0
    visual_tokens: int = 0
    target_text: str = ""
    content_width: int = 0
    content_height: int = 0
    overflow: bool = False
    kaTeX_errors: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: str = ""


class HtmlOCRRenderer:
    """Wraps the local html_ocr_renderer with dynamic canvas support."""

    def __init__(self, browser_config: BrowserConfig):
        self._cfg = browser_config
        self._renderer = None
        self._katex_css = ""
        self._katex_js = ""

        if self._cfg.katex_dist:
            kd = Path(self._cfg.katex_dist)
            css_path = kd / "katex.min.css"
            js_path = kd / "katex.min.js"
            if css_path.exists():
                self._katex_css = f"file://{css_path}"
            if js_path.exists():
                self._katex_js = f"file://{js_path}"

    def _get_renderer(self):
        if self._renderer is not None:
            return self._renderer

        from .html_ocr_renderer import BrowserConfig as OldBrowserConfig
        from .html_ocr_renderer import HtmlOCRRenderer as OldRenderer
        from .html_ocr_renderer import RenderConfig as OldRenderConfig

        old_render_cfg = OldRenderConfig(output_mode="uint8")
        old_browser_cfg = OldBrowserConfig(
            executable_path=self._cfg.executable_path,
            katex_dist=self._cfg.katex_dist,
            timeout_ms=self._cfg.timeout_ms,
        )
        self._renderer = OldRenderer(old_render_cfg, old_browser_cfg)
        return self._renderer

    def measure_content(
        self,
        html_content: str,
        viewport_width: int = 1200,
        viewport_height: int = 800,
    ) -> dict[str, int]:
        """Render HTML in a measurement viewport and return content dimensions.

        Uses JavaScript element.getBoundingClientRect() to get real content size.
        """
        try:
            renderer = self._get_renderer()
            # Render with a large viewport and extract the measurement
            result = renderer._screenshot_html(html_content, viewport_width, viewport_height)
            if result.get("success"):
                # Measurement is done via JS in the HTML itself
                measure_str = result.get("measure_result", "{}")
                measure = json.loads(measure_str)
                return {
                    "width": measure.get("width", viewport_width),
                    "height": measure.get("height", viewport_height),
                }
            return {"width": viewport_width, "height": viewport_height}
        except Exception:
            return {"width": viewport_width // 2, "height": viewport_height // 4}

    def render(
        self,
        html_content: str,
        width: int,
        height: int,
    ) -> RenderResult:
        """Render HTML to a PNG image of given dimensions.

        Args:
            html_content: Complete HTML page.
            width: Target canvas width.
            height: Target canvas height.

        Returns:
            RenderResult with png_bytes, pixel_values tensor, and metadata.
        """
        try:
            renderer = self._get_renderer()
            raw_result = renderer._screenshot_html(html_content, width, height)

            if not raw_result.get("success", False):
                return RenderResult(error=raw_result.get("error", "unknown"), success=False)

            png_bytes = raw_result.get("png_or_jpeg_bytes", b"")
            pixel_values = raw_result.get("pixel_values")

            if pixel_values is None and png_bytes:
                # Decode PNG to tensor
                from io import BytesIO
                from PIL import Image
                img = Image.open(BytesIO(png_bytes))
                img_arr = np.array(img)
                pixel_values = torch.from_numpy(img_arr).permute(2, 0, 1).to(torch.uint8)

            return RenderResult(
                png_bytes=png_bytes,
                pixel_values=pixel_values,
                canvas_width=width,
                canvas_height=height,
                grid_w=width // 32,
                grid_h=height // 32,
                visual_tokens=(width // 32) * (height // 32),
                overflow=raw_result.get("overflow", False),
                kaTeX_errors=raw_result.get("katex_errors", 0),
                success=True,
            )
        except Exception as exc:
            return RenderResult(error=str(exc), success=False)

    def shutdown(self):
        """Close browser resources."""
        if self._renderer is not None:
            try:
                self._renderer.shutdown()
            except Exception:
                pass
            self._renderer = None
