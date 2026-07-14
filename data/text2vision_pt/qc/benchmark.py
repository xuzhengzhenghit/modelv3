#!/usr/bin/env python3
"""Benchmark render_dynamic throughput."""

import json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rendering.html_ocr_renderer import (
    HtmlOCRRenderer, RenderConfig, BrowserConfig, RenderUnit, NeedsSplit, TooWide
)

BROWSER = "/root/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome"
KATEX = "/mnt/si001719bp3c/default/XJZ/modelv3/data/haina_html_render/node_modules/katex/dist"


def benchmark(manifest_path, num=200, warmup=20):
    print(f"Loading {num + warmup} samples ...")
    records = []
    with open(manifest_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("t"):
                records.append(r)
            if len(records) >= num + warmup:
                break

    while len(records) < num + warmup:
        records.extend(records)

    print(f"Creating renderer ...")
    renderer = HtmlOCRRenderer(
        RenderConfig(output_mode="uint8"),
        BrowserConfig(executable_path=BROWSER, katex_dist=KATEX),
    )

    layout_times, screenshot_times, decode_times, total_times = [], [], [], []
    tokens_list, sizes_list, splits, errors = [], [], 0, 0

    for idx, rec in enumerate(records):
        sid = rec["i"]
        text = rec["t"]
        blocks = [{"id": "b0", "kind": "paragraph", "parts": [{"kind": "text", "text": text}]}]
        unit = RenderUnit(sample_id=sid, blocks=tuple(blocks), target_text=text)
        seed = hash(sid) & 0x7FFFFFFF

        t0 = time.perf_counter()
        try:
            result = renderer.render_dynamic(unit, seed)
        except NeedsSplit:
            splits += 1
            continue
        except TooWide:
            errors += 1
            continue
        except Exception:
            errors += 1
            continue
        t1 = time.perf_counter()

        if idx >= warmup:
            meta = result["render_meta"]
            layout_times.append(meta["layout_ms"])
            screenshot_times.append(meta["screenshot_ms"])
            decode_times.append(meta["decode_ms"])
            total_times.append((t1 - t0) * 1000)
            tokens_list.append(result["num_visual_tokens"])
            sizes_list.append(result["paper_size"])

    renderer.close()

    def p(x): return f"{statistics.mean(x):.1f}" if x else "-"
    def p50(x): return f"{statistics.median(x):.1f}" if x else "-"
    def p95(x): return f"{_p95(x):.1f}" if x else "-"

    n = len(total_times)
    thr = 1000 / statistics.mean(total_times) if total_times else 0
    tok_thr = statistics.mean(tokens_list) * thr if tokens_list else 0

    print(f"\n{'='*60}")
    print(f"  Samples: {n} (warmup={warmup}, splits={splits}, errors={errors})")
    print(f"{'='*60}")
    print(f"{'Stage':<20} {'Mean':>8} {'P50':>8} {'P95':>8}")
    print(f"{'-'*44}")
    for name, data in [("layout", layout_times), ("screenshot", screenshot_times),
                        ("decode", decode_times), ("TOTAL (python)", total_times)]:
        print(f"{name:<20} {p(data):>7}ms {p50(data):>7}ms {p95(data):>7}ms")

    print(f"\n{'='*60}")
    print(f"  Throughput:        {thr:.1f} samples/s")
    print(f"  Visual tokens/s:   {tok_thr:.0f}")
    print(f"  Avg visual tokens: {statistics.mean(tokens_list):.0f} (min={min(tokens_list)} max={max(tokens_list)})")
    print(f"{'='*60}")

    # Size distribution
    counts = {}
    for s in sizes_list:
        k = f"{s[0]}×{s[1]}"
        counts[k] = counts.get(k, 0) + 1
    print(f"\nSize distribution:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * (v * 40 // n)
        print(f"  {k:<10} {v:>4} ({v*100/n:4.1f}%) {bar}")


def _p95(data): return sorted(data)[int(len(data) * 0.95)]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()
    benchmark(args.manifest, args.count, args.warmup)
