#!/usr/bin/env python3
"""Persistent Chromium renderer for online OCR pretraining.

One renderer object may be constructed in the main process, but Chromium is lazily
created inside each DataLoader worker. The browser/page stay alive for subsequent
samples. Images are returned from memory and are not written unless preview saving
is explicitly enabled.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from PIL import Image
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

from document_parser import join_targets

OutputMode = Literal["pil", "uint8", "float01", "float11"]

TEMPLATE_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #cfcfcf; }
body { font-family: var(--font-family); }
#paper {
  width: var(--paper-width);
  height: var(--paper-height);
  overflow: hidden;
  background: var(--paper-bg);
  color: var(--text-color);
  padding: var(--padding-y) var(--padding-x);
  font-size: var(--font-size);
  line-height: var(--line-height);
  letter-spacing: var(--letter-spacing);
}
#content { width: 100%; height: 100%; overflow: hidden; }
p { margin: 0 0 var(--paragraph-gap) 0; text-align: var(--text-align); }
.display-math { margin: 0.45em 0 0.65em 0; text-align: center; overflow: hidden; }
table { width: 100%; border-collapse: collapse; margin: 0.45em 0 0.65em; table-layout: fixed; }
td, th { border: var(--table-border) solid #333; padding: 0.22em 0.38em; overflow-wrap: anywhere; }
th { font-weight: 600; background: rgba(0,0,0,0.035); }
.katex-display { margin: 0 !important; overflow: hidden; }
</style>
</head>
<body>
<div id="paper"><div id="content"></div></div>
<script>
function applyStyle(style, fontSize) {
  const root = document.documentElement;
  const values = {
    '--paper-width': style.width + 'px',
    '--paper-height': style.height + 'px',
    '--paper-bg': style.background,
    '--text-color': style.color,
    '--padding-x': style.padding_x + 'px',
    '--padding-y': style.padding_y + 'px',
    '--font-size': fontSize + 'px',
    '--line-height': String(style.line_height),
    '--letter-spacing': style.letter_spacing + 'px',
    '--paragraph-gap': style.paragraph_gap + 'em',
    '--text-align': style.text_align,
    '--font-family': style.font_family,
    '--table-border': style.table_border + 'px'
  };
  for (const [key, value] of Object.entries(values)) root.style.setProperty(key, value);
}

function renderMath(node, tex, displayMode, state) {
  if (!window.katex) {
    node.textContent = displayMode ? `\\[${tex}\\]` : `\\(${tex}\\)`;
    state.katexMissing = true;
    return;
  }
  try {
    window.katex.render(tex, node, {
      displayMode,
      throwOnError: false,
      strict: 'ignore',
      trust: false,
      output: 'htmlAndMathml'
    });
  } catch (error) {
    state.katexErrors += 1;
    node.textContent = tex;
  }
}

function appendBlock(content, block, state) {
  if (block.kind === 'paragraph') {
    const p = document.createElement('p');
    for (const part of block.parts || []) {
      if (part.kind === 'text') {
        p.appendChild(document.createTextNode(part.text || ''));
      } else if (part.kind === 'math_inline') {
        const span = document.createElement('span');
        renderMath(span, part.tex || '', false, state);
        p.appendChild(span);
      }
    }
    content.appendChild(p);
  } else if (block.kind === 'math_display') {
    const div = document.createElement('div');
    div.className = 'display-math';
    renderMath(div, block.tex || '', true, state);
    content.appendChild(div);
  } else if (block.kind === 'table') {
    const table = document.createElement('table');
    for (let rowIndex = 0; rowIndex < (block.rows || []).length; rowIndex++) {
      const tr = document.createElement('tr');
      for (const value of block.rows[rowIndex]) {
        const cell = document.createElement(rowIndex === 0 ? 'th' : 'td');
        cell.textContent = String(value);
        tr.appendChild(cell);
      }
      table.appendChild(tr);
    }
    content.appendChild(table);
  }
}

function hasOverflow(content) {
  return content.scrollHeight > content.clientHeight + 1 || content.scrollWidth > content.clientWidth + 1;
}

window.renderBlocks = async function(payload) {
  const content = document.getElementById('content');
  let finalState = null;
  let finalFontSize = payload.style.font_size;

  for (let fontSize = payload.style.font_size; fontSize >= payload.style.min_font_size; fontSize -= 1) {
    applyStyle(payload.style, fontSize);
    content.replaceChildren();
    const state = {katexErrors: 0, katexMissing: false};
    for (const block of payload.blocks) appendBlock(content, block, state);
    if (document.fonts && document.fonts.ready) await document.fonts.ready;
    await new Promise(resolve => requestAnimationFrame(resolve));
    finalState = state;
    finalFontSize = fontSize;
    if (!hasOverflow(content)) break;
  }

  let usedCount = payload.blocks.length;
  while (usedCount > 1 && hasOverflow(content)) {
    content.lastElementChild.remove();
    usedCount -= 1;
  }
  await new Promise(resolve => requestAnimationFrame(resolve));

  return {
    usedCount,
    overflow: hasOverflow(content),
    scrollHeight: content.scrollHeight,
    clientHeight: content.clientHeight,
    finalFontSize,
    katexErrors: finalState ? finalState.katexErrors : 0,
    katexMissing: finalState ? finalState.katexMissing : true,
    visibleTextLength: content.innerText.length
  };
};
</script>
</body>
</html>
"""


@dataclasses.dataclass(frozen=True)
class RenderConfig:
    width: int = 1024
    height: int = 512
    patch_size: int = 32
    font_size_min: int = 21
    font_size_max: int = 29
    fit_min_font_size: int = 18
    padding_x_range: tuple[int, int] = (32, 58)
    padding_y_range: tuple[int, int] = (26, 46)
    line_height_range: tuple[float, float] = (1.30, 1.55)
    paragraph_gap_range: tuple[float, float] = (0.40, 0.70)
    letter_spacing_range: tuple[float, float] = (-0.05, 0.18)
    backgrounds: tuple[str, ...] = ("#ffffff", "#fcfcfa", "#faf9f4", "#f7f7f7")
    colors: tuple[str, ...] = ("#101010", "#161616", "#202020")
    font_families: tuple[str, ...] = (
        'Arial, "Noto Sans", "Noto Sans CJK SC", sans-serif',
        '"Times New Roman", "Noto Serif", "Noto Serif CJK SC", serif',
        'Georgia, "Noto Serif", "Noto Serif CJK SC", serif',
    )
    text_alignments: tuple[str, ...] = ("left", "left", "left", "justify")
    table_border_range: tuple[float, float] = (0.8, 1.2)
    screenshot_type: Literal["png", "jpeg"] = "png"
    jpeg_quality: int = 95
    output_mode: OutputMode = "uint8"

    def __post_init__(self) -> None:
        if self.width % self.patch_size or self.height % self.patch_size:
            raise ValueError("width and height must be multiples of patch_size")
        if self.fit_min_font_size > self.font_size_min:
            raise ValueError("fit_min_font_size must be <= font_size_min")

    @property
    def image_grid_thw(self) -> tuple[int, int, int]:
        return (1, self.height // self.patch_size, self.width // self.patch_size)

    @property
    def num_visual_tokens(self) -> int:
        return (self.width // self.patch_size) * (self.height // self.patch_size)


@dataclasses.dataclass(frozen=True)
class BrowserConfig:
    executable_path: str | None = None
    katex_dist: str | None = None
    headless: bool = True
    timeout_ms: int = 20_000


class HtmlOCRRenderer:
    def __init__(self, render_config: RenderConfig = RenderConfig(), browser_config: BrowserConfig = BrowserConfig()):
        self.config = render_config
        self.browser_config = browser_config
        self._pid: int | None = None
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._katex_dist = self._find_katex_dist(browser_config.katex_dist)
        self._browser_path = self._find_browser(browser_config.executable_path)

    @staticmethod
    def _find_browser(explicit: str | None) -> str | None:
        if explicit:
            path = Path(explicit)
            if not path.exists():
                raise FileNotFoundError(path)
            return str(path)
        env = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
        if env and Path(env).exists():
            return env
        for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
            found = shutil.which(name)
            if found:
                return found
        return None

    @staticmethod
    def _find_katex_dist(explicit: str | None) -> Path | None:
        candidates = []
        if explicit:
            candidates.append(Path(explicit))
        candidates.extend(
            [
                Path(__file__).resolve().parent / "node_modules" / "katex" / "dist",
                Path.cwd() / "node_modules" / "katex" / "dist",
            ]
        )
        for path in candidates:
            if (path / "katex.min.js").exists() and (path / "katex.min.css").exists():
                return path.resolve()
        return None

    def _ensure_page(self) -> Page:
        pid = os.getpid()
        if self._pid == pid and self._page is not None:
            return self._page
        self.close()

        self._playwright = sync_playwright().start()
        launch_kwargs: dict[str, Any] = {
            "headless": self.browser_config.headless,
            "args": ["--disable-dev-shm-usage", "--no-sandbox"],
        }
        if self._browser_path:
            launch_kwargs["executable_path"] = self._browser_path
        try:
            self._browser = self._playwright.chromium.launch(**launch_kwargs)
        except Exception as exc:
            self.close()
            raise RuntimeError(
                "Cannot launch Chromium. Run `playwright install chromium` or set "
                "PLAYWRIGHT_CHROMIUM_EXECUTABLE. Original error: " + str(exc)
            ) from exc

        self._context = self._browser.new_context(
            viewport={"width": self.config.width + 32, "height": self.config.height + 32},
            device_scale_factor=1,
        )
        self._page = self._context.new_page()
        self._page.set_default_timeout(self.browser_config.timeout_ms)
        self._page.set_content(TEMPLATE_HTML, wait_until="load")
        if self._katex_dist is not None:
            self._page.add_style_tag(path=str(self._katex_dist / "katex.min.css"))
            self._page.add_script_tag(path=str(self._katex_dist / "katex.min.js"))
        self._page.wait_for_function("typeof window.renderBlocks === 'function'")
        self._pid = pid
        return self._page

    def _random_style(self, rng: random.Random) -> dict[str, Any]:
        c = self.config
        return {
            "width": c.width,
            "height": c.height,
            "background": rng.choice(c.backgrounds),
            "color": rng.choice(c.colors),
            "padding_x": rng.randint(*c.padding_x_range),
            "padding_y": rng.randint(*c.padding_y_range),
            "font_size": rng.randint(c.font_size_min, c.font_size_max),
            "min_font_size": c.fit_min_font_size,
            "line_height": rng.uniform(*c.line_height_range),
            "paragraph_gap": rng.uniform(*c.paragraph_gap_range),
            "letter_spacing": rng.uniform(*c.letter_spacing_range),
            "text_align": rng.choice(c.text_alignments),
            "font_family": rng.choice(c.font_families),
            "table_border": rng.uniform(*c.table_border_range),
        }

    def _decode(self, screenshot: bytes) -> Image.Image | torch.Tensor:
        image = Image.open(io.BytesIO(screenshot)).convert("RGB")
        mode = self.config.output_mode
        if mode == "pil":
            return image
        array = np.asarray(image, dtype=np.uint8)
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()
        if mode == "uint8":
            return tensor
        tensor = tensor.float().div_(255.0)
        if mode == "float11":
            tensor = tensor.mul_(2.0).sub_(1.0)
        return tensor

    def render(self, blocks: Sequence[dict[str, Any]], seed: int) -> dict[str, Any]:
        if not blocks:
            raise ValueError("blocks cannot be empty")
        page = self._ensure_page()
        rng = random.Random(seed)
        style = self._random_style(rng)
        payload = {"blocks": list(blocks), "style": style}

        begin = time.perf_counter()
        try:
            t0 = time.perf_counter()
            info = page.evaluate("payload => window.renderBlocks(payload)", payload)
            t1 = time.perf_counter()
            screenshot_kwargs: dict[str, Any] = {
                "type": self.config.screenshot_type,
                "animations": "disabled",
            }
            if self.config.screenshot_type == "jpeg":
                screenshot_kwargs["quality"] = self.config.jpeg_quality
            screenshot = page.locator("#paper").screenshot(**screenshot_kwargs)
            t2 = time.perf_counter()
            pixel_values = self._decode(screenshot)
            t3 = time.perf_counter()
        except Exception:
            # A browser process may occasionally die under worker pressure. Restart once.
            self.close()
            page = self._ensure_page()
            info = page.evaluate("payload => window.renderBlocks(payload)", payload)
            screenshot = page.locator("#paper").screenshot(type=self.config.screenshot_type, animations="disabled")
            pixel_values = self._decode(screenshot)
            t0 = begin
            t1 = t2 = t3 = time.perf_counter()

        used_count = int(info["usedCount"])
        used_blocks = list(blocks[:used_count])
        target = join_targets(used_blocks)
        if not target:
            raise RuntimeError("Renderer produced an empty target")

        return {
            "pixel_values": pixel_values,
            "target_text": target,
            "image_grid_thw": torch.tensor(self.config.image_grid_thw, dtype=torch.long),
            "num_visual_tokens": self.config.num_visual_tokens,
            "png_or_jpeg_bytes": screenshot,
            "render_meta": {
                **info,
                "used_count": used_count,
                "input_block_count": len(blocks),
                "style": style,
                "katex_available": self._katex_dist is not None,
                "dom_ms": (t1 - t0) * 1000,
                "screenshot_ms": (t2 - t1) * 1000,
                "decode_ms": (t3 - t2) * 1000,
                "total_ms": (t3 - begin) * 1000,
            },
        }

    def close(self) -> None:
        for obj in (self._page, self._context, self._browser):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception:
            pass
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        for key in ("_page", "_context", "_browser", "_playwright"):
            state[key] = None
        state["_pid"] = None
        return state

    def __del__(self) -> None:  # pragma: no cover
        self.close()


def stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def save_preview(directory: str | Path, sample_id: str, result: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in sample_id)[:120]
    image_path = directory / f"{safe_id}.png"
    image_path.write_bytes(result["png_or_jpeg_bytes"])
    metadata = {
        "id": sample_id,
        "target_text": result["target_text"],
        "image_grid_thw": result["image_grid_thw"].tolist(),
        "num_visual_tokens": result["num_visual_tokens"],
        "render_meta": result["render_meta"],
        **(extra or {}),
    }
    (directory / f"{safe_id}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
