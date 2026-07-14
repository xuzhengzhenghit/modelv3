#!/usr/bin/env python3
"""Small feasibility/speed benchmark for HTML -> Chromium screenshot -> Tensor.

The benchmark intentionally keeps one Chromium page alive and replaces only the
DOM for each sample. It prints stage-level latency, throughput, output shapes,
and optionally saves a few preview PNG files.

Example:
    python benchmark_html_render.py --count 300 --warmup 20 --save-first 8

Dependencies:
    pip install playwright pillow numpy torch
    npm install katex

The script can use either Playwright's bundled Chromium or a system Chromium.
Set --browser-path /usr/bin/chromium when needed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from playwright.sync_api import Browser, Page, Playwright, sync_playwright


BASE_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: #d7d7d7; }
  body { font-family: Arial, "Noto Sans", "Noto Sans CJK SC", sans-serif; }
  #paper {
    width: var(--paper-width);
    height: var(--paper-height);
    overflow: hidden;
    background: var(--paper-bg);
    color: var(--text-color);
    padding: var(--padding-y) var(--padding-x);
    font-size: var(--font-size);
    line-height: var(--line-height);
  }
  #content { width: 100%; height: 100%; overflow: hidden; }
  p { margin: 0 0 0.55em 0; text-align: var(--text-align); }
  .display-math { margin: 0.50em 0 0.70em 0; text-align: center; }
  table { width: 100%; border-collapse: collapse; margin: 0.5em 0; table-layout: fixed; }
  td, th { border: 1px solid #333; padding: 0.25em 0.4em; overflow-wrap: anywhere; }
  th { font-weight: 600; }
  .katex-display { margin: 0 !important; overflow: hidden; }
</style>
</head>
<body>
<div id="paper"><div id="content"></div></div>
<script>
window.renderOne = async function(sample) {
  const root = document.documentElement;
  root.style.setProperty('--paper-width', sample.style.width + 'px');
  root.style.setProperty('--paper-height', sample.style.height + 'px');
  root.style.setProperty('--paper-bg', sample.style.background);
  root.style.setProperty('--text-color', sample.style.color);
  root.style.setProperty('--padding-x', sample.style.padding_x + 'px');
  root.style.setProperty('--padding-y', sample.style.padding_y + 'px');
  root.style.setProperty('--font-size', sample.style.font_size + 'px');
  root.style.setProperty('--line-height', String(sample.style.line_height));
  root.style.setProperty('--text-align', sample.style.text_align);

  const content = document.getElementById('content');
  content.replaceChildren();
  let katexErrors = 0;

  function renderMath(node, tex, displayMode) {
    if (!window.katex) {
      node.textContent = displayMode ? `\\[${tex}\\]` : `\\(${tex}\\)`;
      node.dataset.katexMissing = '1';
      return;
    }
    try {
      window.katex.render(tex, node, {
        displayMode: displayMode,
        throwOnError: false,
        strict: 'ignore',
        trust: false,
        output: 'htmlAndMathml'
      });
    } catch (error) {
      katexErrors += 1;
      node.textContent = tex;
      node.dataset.katexError = String(error);
    }
  }

  for (const block of sample.blocks) {
    if (block.kind === 'paragraph') {
      const p = document.createElement('p');
      for (const part of block.parts) {
        if (part.kind === 'text') {
          p.appendChild(document.createTextNode(part.text));
        } else if (part.kind === 'math_inline') {
          const span = document.createElement('span');
          renderMath(span, part.tex, false);
          p.appendChild(span);
        }
      }
      content.appendChild(p);
    } else if (block.kind === 'math_display') {
      const div = document.createElement('div');
      div.className = 'display-math';
      renderMath(div, block.tex, true);
      content.appendChild(div);
    } else if (block.kind === 'table') {
      const table = document.createElement('table');
      for (let r = 0; r < block.rows.length; r++) {
        const tr = document.createElement('tr');
        for (const value of block.rows[r]) {
          const cell = document.createElement(r === 0 ? 'th' : 'td');
          cell.textContent = value;
          tr.appendChild(cell);
        }
        table.appendChild(tr);
      }
      content.appendChild(table);
    }
  }

  if (document.fonts && document.fonts.ready) {
    await document.fonts.ready;
  }
  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));

  return {
    scrollHeight: content.scrollHeight,
    clientHeight: content.clientHeight,
    overflow: content.scrollHeight > content.clientHeight + 1,
    katexErrors,
    katexLoaded: Boolean(window.katex),
    textLength: content.innerText.length
  };
};
</script>
</body>
</html>
"""


@dataclass
class Timings:
    dom_ms: float
    screenshot_ms: float
    decode_ms: float
    tensor_ms: float
    total_ms: float
    png_bytes: int
    overflow: bool
    katex_errors: int


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def find_browser(explicit: str | None) -> str | None:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Browser executable does not exist: {path}")
        return str(path)
    env_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    if env_path and Path(env_path).exists():
        return env_path
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        found = shutil.which(name)
        if found:
            return found
    return None


def find_katex_dist(explicit: str | None) -> Path | None:
    candidates: list[Path] = []
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


def make_samples(width: int, height: int) -> list[dict[str, Any]]:
    common_style = {
        "width": width,
        "height": height,
        "background": "#ffffff",
        "color": "#111111",
        "padding_x": 42,
        "padding_y": 34,
        "font_size": 25,
        "line_height": 1.42,
        "text_align": "left",
    }
    return [
        {
            "name": "plain_science",
            "style": common_style,
            "blocks": [
                {
                    "kind": "paragraph",
                    "parts": [
                        {
                            "kind": "text",
                            "text": (
                                "What could be more mesmerizing than an eclipse that occurs right before "
                                "Halloween? Astronomers use precise measurements to study the geometry of "
                                "the Sun, Earth, and Moon. The same observations also help researchers "
                                "explain changes in brightness, orbital motion, and atmospheric scattering."
                            ),
                        }
                    ],
                },
                {
                    "kind": "paragraph",
                    "parts": [
                        {
                            "kind": "text",
                            "text": (
                                "This page is rendered entirely in memory. Chromium performs font shaping, "
                                "line breaking, layout, rasterization, and PNG encoding before the image is "
                                "decoded into a tensor for model training."
                            ),
                        }
                    ],
                },
            ],
        },
        {
            "name": "inline_math",
            "style": {**common_style, "font_size": 26},
            "blocks": [
                {
                    "kind": "paragraph",
                    "parts": [
                        {"kind": "text", "text": "For a small-signal model, the source resistance is "},
                        {"kind": "math_inline", "tex": r"r_s=1/g_m"},
                        {"kind": "text", "text": ", while a line can be represented by "},
                        {"kind": "math_inline", "tex": r"f(x)=\frac{2}{3}x+2"},
                        {"kind": "text", "text": ". Inline formulas remain part of the surrounding paragraph."},
                    ],
                },
                {
                    "kind": "paragraph",
                    "parts": [
                        {
                            "kind": "text",
                            "text": (
                                "The benchmark measures DOM construction and formula typesetting separately "
                                "from screenshot generation and tensor conversion."
                            ),
                        }
                    ],
                },
            ],
        },
        {
            "name": "display_math",
            "style": {**common_style, "font_size": 24},
            "blocks": [
                {
                    "kind": "paragraph",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "A continuum mechanics article may contain the following governing equation:",
                        }
                    ],
                },
                {
                    "kind": "math_display",
                    "tex": r"\rho \frac{\partial^2 u}{\partial t^2}=\nabla\cdot\sigma",
                },
                {
                    "kind": "paragraph",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Another page might describe a center-of-mass calculation:",
                        }
                    ],
                },
                {
                    "kind": "math_display",
                    "tex": r"x_{cm}=\frac{1}{m_{total}}\int x\,dm",
                },
            ],
        },
        {
            "name": "table",
            "style": {**common_style, "font_size": 23},
            "blocks": [
                {
                    "kind": "paragraph",
                    "parts": [{"kind": "text", "text": "A compact scientific results table:"}],
                },
                {
                    "kind": "table",
                    "rows": [
                        ["Method", "Accuracy", "Latency (ms)"],
                        ["Baseline", "91.4%", "18.7"],
                        ["Proposed", "94.8%", "21.2"],
                        ["Ablation", "92.1%", "19.6"],
                    ],
                },
                {
                    "kind": "paragraph",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "The table uses browser layout and can later be extended to rowspan and colspan.",
                        }
                    ],
                },
            ],
        },
    ]


def initialize_page(page: Page, katex_dist: Path | None) -> None:
    page.set_content(BASE_HTML, wait_until="load")
    if katex_dist is not None:
        page.add_style_tag(path=str(katex_dist / "katex.min.css"))
        page.add_script_tag(path=str(katex_dist / "katex.min.js"))
    page.wait_for_function("typeof window.renderOne === 'function'")


def render_once(
    page: Page,
    sample: dict[str, Any],
    tensor_mode: str,
    *,
    reinitialize: bool = False,
    katex_dist: Path | None = None,
) -> tuple[Timings, torch.Tensor, bytes, dict[str, Any]]:
    start = time.perf_counter()

    t0 = time.perf_counter()
    if reinitialize:
        initialize_page(page, katex_dist)
    info = page.evaluate("sample => window.renderOne(sample)", sample)
    t1 = time.perf_counter()

    png = page.locator("#paper").screenshot(type="png", animations="disabled")
    t2 = time.perf_counter()

    image = Image.open(io.BytesIO(png)).convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    t3 = time.perf_counter()

    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()
    if tensor_mode == 'float32':
        tensor = tensor.float().div_(255.0)
    t4 = time.perf_counter()

    timing = Timings(
        dom_ms=(t1 - t0) * 1000,
        screenshot_ms=(t2 - t1) * 1000,
        decode_ms=(t3 - t2) * 1000,
        tensor_ms=(t4 - t3) * 1000,
        total_ms=(t4 - start) * 1000,
        png_bytes=len(png),
        overflow=bool(info["overflow"]),
        katex_errors=int(info["katexErrors"]),
    )
    return timing, tensor, png, info


def summarize(timings: list[Timings], elapsed_s: float) -> None:
    def line(name: str, values: list[float]) -> None:
        print(
            f"  {name:<12} mean={statistics.fmean(values):8.2f} ms  "
            f"p50={percentile(values, 0.50):8.2f}  p95={percentile(values, 0.95):8.2f}  "
            f"max={max(values):8.2f}"
        )

    print("\n========== benchmark summary ==========")
    print(f"samples          : {len(timings)}")
    print(f"wall time        : {elapsed_s:.3f} s")
    print(f"throughput       : {len(timings) / elapsed_s:.2f} pages/s")
    print(f"average PNG size : {statistics.fmean(t.png_bytes for t in timings) / 1024:.1f} KiB")
    print(f"overflow pages   : {sum(t.overflow for t in timings)}")
    print(f"KaTeX errors     : {sum(t.katex_errors for t in timings)}")
    line("DOM/KaTeX", [t.dom_ms for t in timings])
    line("screenshot", [t.screenshot_ms for t in timings])
    line("PNG decode", [t.decode_ms for t in timings])
    line("to tensor", [t.tensor_ms for t in timings])
    line("total", [t.total_ms for t in timings])
    print("=======================================\n")


def launch_browser(playwright: Playwright, browser_path: str | None) -> Browser:
    kwargs: dict[str, Any] = {
        "headless": True,
        "args": ["--disable-dev-shm-usage", "--no-sandbox"],
    }
    if browser_path:
        kwargs["executable_path"] = browser_path
    try:
        return playwright.chromium.launch(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Chromium could not be launched. Run `playwright install chromium`, or pass "
            "`--browser-path /path/to/chromium`. Original error:\n" + str(exc)
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200, help="Measured pages")
    parser.add_argument("--warmup", type=int, default=20, help="Warm-up pages excluded from statistics")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--save-first", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_output"))
    parser.add_argument("--browser-path", type=str, default=None)
    parser.add_argument(
        "--katex-dist",
        type=str,
        default=None,
        help="Directory containing katex.min.js and katex.min.css; defaults to ./node_modules/katex/dist",
    )
    parser.add_argument(
        "--tensor-mode", choices=['uint8', 'float32'], default='uint8',
        help="uint8 is recommended for CPU rendering; convert after non-blocking GPU transfer",
    )
    parser.add_argument(
        "--compare-reload",
        action="store_true",
        help="Also benchmark the slow anti-pattern: reload template and KaTeX for every sample",
    )
    parser.add_argument(
        "--no-katex",
        action="store_true",
        help="Render formulas as literal source, useful for measuring browser-only speed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0 or args.warmup < 0:
        raise ValueError("--count must be positive and --warmup cannot be negative")
    if args.width % 32 or args.height % 32:
        print("WARNING: width/height are not multiples of HainaOCR's effective patch size 32.")

    browser_path = find_browser(args.browser_path)
    katex_dist = None if args.no_katex else find_katex_dist(args.katex_dist)

    print("========== configuration ==========")
    print(f"Python           : {sys.version.split()[0]}")
    print(f"PyTorch          : {torch.__version__}")
    print(f"canvas           : {args.width} x {args.height}")
    print(f"visual tokens@32 : {(args.width // 32) * (args.height // 32)}")
    print(f"browser          : {browser_path or 'Playwright bundled Chromium'}")
    print(f"KaTeX            : {katex_dist or 'disabled/not found'}")
    print(f"warmup/measured  : {args.warmup}/{args.count}")
    print(f"tensor mode      : {args.tensor_mode}")
    print("===================================")

    if not args.no_katex and katex_dist is None:
        print(
            "WARNING: KaTeX assets were not found. Formula source will be shown literally.\n"
            "Install locally with: npm install katex\n"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = make_samples(args.width, args.height)
    rng = random.Random(args.seed)

    with sync_playwright() as playwright:
        browser = launch_browser(playwright, browser_path)
        context = browser.new_context(
            viewport={"width": args.width + 32, "height": args.height + 32},
            device_scale_factor=1,
        )
        page = context.new_page()
        initialize_page(page, katex_dist)

        print("\nWarming up...")
        for i in range(args.warmup):
            sample = samples[i % len(samples)]
            render_once(page, sample, args.tensor_mode)

        print("Benchmarking...")
        timings: list[Timings] = []
        wall_start = time.perf_counter()
        for index in range(args.count):
            sample = rng.choice(samples)
            timing, tensor, png, info = render_once(page, sample, args.tensor_mode)
            timings.append(timing)

            if index < args.save_first:
                stem = f"{index:04d}_{sample['name']}"
                (args.output_dir / f"{stem}.png").write_bytes(png)
                metadata = {
                    "sample_name": sample["name"],
                    "tensor_shape": list(tensor.shape),
                    "tensor_dtype": str(tensor.dtype),
                    "tensor_min": float(tensor.min()),
                    "tensor_max": float(tensor.max()),
                    "render_info": info,
                    "timing_ms": timing.__dict__,
                }
                (args.output_dir / f"{stem}.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            completed = index + 1
            if completed == 1 or completed % args.progress_every == 0 or completed == args.count:
                recent = timings[-min(len(timings), args.progress_every) :]
                recent_ms = statistics.fmean(t.total_ms for t in recent)
                print(
                    f"[{completed:5d}/{args.count}] sample={sample['name']:<14} "
                    f"shape={tuple(tensor.shape)} total={timing.total_ms:7.2f} ms "
                    f"recent={1000.0 / recent_ms:6.2f} pages/s "
                    f"overflow={timing.overflow} katex_errors={timing.katex_errors}"
                )

        wall_elapsed = time.perf_counter() - wall_start
        summarize(timings, wall_elapsed)

        if args.compare_reload:
            compare_count = min(args.count, 30)
            print(f"Benchmarking reload-per-sample anti-pattern ({compare_count} pages)...")
            reload_timings: list[Timings] = []
            reload_start = time.perf_counter()
            for index in range(compare_count):
                sample = samples[index % len(samples)]
                timing, _, _, _ = render_once(
                    page, sample, args.tensor_mode, reinitialize=True, katex_dist=katex_dist
                )
                reload_timings.append(timing)
            summarize(reload_timings, time.perf_counter() - reload_start)

        context.close()
        browser.close()

    print(f"Preview output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
