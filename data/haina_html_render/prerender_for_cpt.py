#!/usr/bin/env python3
"""Pre-render HTML pages to disk in Swift CPT format (images + JSONL).

Usage:
  python prerender_for_cpt.py --manifest /tmp/overfit_100.jsonl --out-dir /tmp/ocr_smoke_html
"""

import argparse, json, time
from pathlib import Path

from html_ocr_renderer import BrowserConfig, HtmlOCRRenderer, RenderConfig, stable_seed

BROWSER = "/root/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome"
KATEX = "/mnt/si001719bp3c/default/XJZ/modelv3/data/haina_html_render/node_modules/katex/dist"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    renderer = HtmlOCRRenderer(
        RenderConfig(output_mode="uint8"),
        BrowserConfig(executable_path=BROWSER, katex_dist=KATEX),
    )

    manifests = []
    with open(args.manifest) as f:
        for line in f:
            manifests.append(json.loads(line))
            if len(manifests) >= args.max_samples:
                break

    total = len(manifests)
    jsonl_path = out_dir / "train.jsonl"
    t0 = time.perf_counter()

    with open(jsonl_path, "w") as out:
        for idx, record in enumerate(manifests):
            sample_id = record.get("id", str(idx))
            blocks = record.get("blocks", [])

            seed = stable_seed(1234, 0, 0, 0, sample_id)
            result = renderer.render(blocks, seed)

            # Save image
            safe_id = sample_id.replace("/", "_").replace("\\", "_")
            img_name = f"{safe_id}.png"
            img_path = img_dir / img_name
            img_path.write_bytes(result["png_or_jpeg_bytes"])

            # Swift CPT JSONL format
            swift_record = {
                "messages": [
                    {"role": "user", "content": "<image>\nOutput the text in the image."},
                    {"role": "assistant", "content": result["target_text"]},
                ],
                "images": [str(img_dir / img_name)],
            }
            out.write(json.dumps(swift_record, ensure_ascii=False) + "\n")

            if (idx + 1) % 25 == 0:
                elapsed = time.perf_counter() - t0
                print(f"[{idx+1:3d}/{total}] {elapsed:.1f}s  {elapsed/(idx+1):.1f}s/img")

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {total} samples in {elapsed:.1f}s ({total/elapsed:.1f} pages/s)")
    print(f"Images: {img_dir}")
    print(f"JSONL:  {jsonl_path}")


if __name__ == "__main__":
    main()
