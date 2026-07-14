#!/usr/bin/env python3
"""Render preview — use existing haina_html_render to generate and save preview images.

Usage:
  python qc/render_preview.py \
    --manifest /mnt/.../tmp/train/train-00000.jsonl \
    --out-dir /mnt/.../tmp/preview \
    --num 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HHTML_RENDER_DIR = SCRIPT_DIR.parent.parent / "haina_html_render"
sys.path.insert(0, str(HHTML_RENDER_DIR))

from html_ocr_renderer import BrowserConfig, HtmlOCRRenderer, RenderConfig


BROWSER = "/root/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome"
KATEX = "/mnt/si001719bp3c/default/XJZ/modelv3/data/haina_html_render/node_modules/katex/dist"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--browser-path", default=BROWSER)
    parser.add_argument("--katex-dist", default=KATEX)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading renderer ...")
    t0 = time.perf_counter()
    renderer = HtmlOCRRenderer(
        RenderConfig(output_mode="uint8"),
        BrowserConfig(executable_path=args.browser_path, katex_dist=args.katex_dist),
    )
    print(f"  init: {time.perf_counter() - t0:.1f}s")

    # Load records
    records = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("t"):  # compact format: {i, t, s, f}
                records.append(rec)
            if len(records) >= args.num:
                break

    print(f"\nRendering {len(records)} previews ...\n")

    results = []
    for idx, rec in enumerate(records):
        sample_id = rec.get("i", str(idx))
        text = rec.get("t", "")
        subject = rec.get("s", "")
        flags = rec.get("f", 0)

        # Build blocks from text — simple paragraph
        blocks = [{"kind": "paragraph", "parts": [{"kind": "text", "text": text}]}]

        # Render
        seed = hash(sample_id) & 0x7FFFFFFF
        try:
            result = renderer.render(blocks, seed)
        except Exception as exc:
            print(f"[{idx+1:2d}] SKIP {sample_id[:40]}: {exc}")
            continue

        png_bytes = result.get("png_or_jpeg_bytes", b"")
        if png_bytes is None:
            print(f"[{idx+1:2d}] SKIP {sample_id[:40]}: no png data")
            continue

        # Save PNG
        safe_id = sample_id.replace("/", "_").replace("\\", "_")
        png_path = out / f"{safe_id}.png"
        png_path.write_bytes(png_bytes)

        # Save JSON metadata
        pv = result.get("pixel_values")
        shape = list(pv.shape) if pv is not None else "?"

        meta = {
            "sample_id": sample_id,
            "subject": subject,
            "flags": flags,
            "target_text": result.get("target_text", "")[:300],
            "image_shape": shape,
            "visual_tokens": result.get("num_visual_tokens", "?"),
            "render_meta": result.get("render_meta", {}),
        }
        json_path = out / f"{safe_id}.json"
        json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        # Summary line
        size_kb = len(png_bytes) / 1024
        nt = result.get("num_visual_tokens", "?")
        print(f"[{idx+1:2d}] {sample_id[:50]:<52s} "
              f"{shape[2] if isinstance(shape, list) else '?'}×{shape[1] if isinstance(shape, list) and len(shape)>1 else '?'}  "
              f"{nt} tokens  {size_kb:.0f}KB")

        results.append(meta)

    try:
        renderer.close()
    except Exception:
        pass
    elapsed = time.perf_counter() - t0
    print(f"\nDone: {len(results)} previews in {elapsed:.1f}s")
    print(f"Output: {out}/")


if __name__ == "__main__":
    main()
