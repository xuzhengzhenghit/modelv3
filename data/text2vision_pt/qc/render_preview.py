#!/usr/bin/env python3
"""Render preview — generate and save sample images from JSONL manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
T2V_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(T2V_DIR))

from rendering.html_ocr_renderer import BrowserConfig, HtmlOCRRenderer, RenderConfig, RenderUnit, NeedsSplit, TooWide


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

    records = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("t"):
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

        blocks = [{"id": "b0", "kind": "paragraph", "parts": [{"kind": "text", "text": text}]}]
        unit = RenderUnit(sample_id=sample_id, blocks=tuple(blocks), target_text=text)

        seed = hash(sample_id) & 0x7FFFFFFF
        try:
            result = renderer.render_dynamic(unit, seed)
        except NeedsSplit as exc:
            print(f"[{idx+1:2d}] SPLIT {sample_id[:40]}: {exc}")
            continue
        except TooWide as exc:
            print(f"[{idx+1:2d}] TOOWIDE {sample_id[:40]}: {exc}")
            continue
        except Exception as exc:
            print(f"[{idx+1:2d}] ERROR {sample_id[:40]}: {exc}")
            continue

        png_bytes = result.get("png_or_jpeg_bytes", b"")
        if not png_bytes:
            print(f"[{idx+1:2d}] SKIP {sample_id[:40]}: no png data")
            continue

        safe_id = sample_id.replace("/", "_").replace("\\", "_")
        png_path = out / f"{safe_id}.png"
        png_path.write_bytes(png_bytes)

        meta = {
            "sample_id": sample_id,
            "subject": subject,
            "flags": flags,
            "target_text": result["target_text"],
            "paper_size": list(result["paper_size"]),
            "content_size": list(result["content_size"]),
            "visual_tokens": result["num_visual_tokens"],
            "image_grid_thw": list(result["image_grid_thw"]),
            "render_meta": result["render_meta"],
        }
        json_path = out / f"{safe_id}.json"
        json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        size_kb = len(png_bytes) / 1024
        nt = result["num_visual_tokens"]
        pw, ph = result["paper_size"]
        cw, ch = result["content_size"]
        oh = "O" if result["render_meta"].get("overflow") else ""
        print(f"[{idx+1:2d}] {sample_id[:50]:<52s} "
              f"paper={pw}×{ph:<4} content={ch}px  "
              f"{nt} tokens  {size_kb:.0f}KB {oh}")

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
