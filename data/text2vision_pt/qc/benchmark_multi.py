#!/usr/bin/env python3
"""Multi-worker rendering throughput benchmark.

Each worker runs a separate process with its own Chromium instance.
Workers read from separate slices of the manifest, render samples in parallel,
and report back throughput.
"""

import json, math, multiprocessing, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BROWSER = "/root/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome"
KATEX = "/mnt/si001719bp3c/default/XJZ/modelv3/data/haina_html_render/node_modules/katex/dist"


def worker_fn(records, worker_id, queue):
    """Render assigned records and push (total_ms, visual_tokens, paper_size, splits) to queue."""
    from rendering.html_ocr_renderer import (
        HtmlOCRRenderer, RenderConfig, BrowserConfig, RenderUnit, NeedsSplit, TooWide
    )
    renderer = HtmlOCRRenderer(
        RenderConfig(output_mode="uint8"),
        BrowserConfig(executable_path=BROWSER, katex_dist=KATEX),
    )
    splits = 0
    for rec in records:
        sid = rec["i"]
        text = rec["t"]
        blocks = [{"id": "b0", "kind": "paragraph", "parts": [{"kind": "text", "text": text}]}]
        unit = RenderUnit(sample_id=sid, blocks=tuple(blocks), target_text=text)
        seed = hash(sid) & 0x7FFFFFFF
        t0 = time.perf_counter()
        try:
            result = renderer.render_dynamic(unit, seed)
        except (NeedsSplit, TooWide):
            splits += 1
            continue
        except Exception:
            continue
        elapsed = (time.perf_counter() - t0) * 1000
        queue.put((elapsed, result["num_visual_tokens"], result["paper_size"], splits, worker_id))
    renderer.close()


def benchmark_multi(manifest_path, num_workers, total_samples=200, warmup_fraction=0.1):
    print(f"Loading {total_samples} samples ...")
    records = []
    with open(manifest_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("t"):
                records.append(r)
            if len(records) >= total_samples:
                break
    while len(records) < total_samples:
        records.extend(records)

    # Split records among workers
    per_worker = math.ceil(len(records) / num_workers)
    chunks = [records[i:i + per_worker] for i in range(0, len(records), num_workers)]

    print(f"  {len(records)} samples → {len(chunks)} workers × ~{per_worker} each")
    print(f"  Starting {num_workers} worker processes ...")

    # Collect results
    queue = multiprocessing.Queue()
    procs = []
    for wid in range(num_workers):
        p = multiprocessing.Process(target=worker_fn, args=(chunks[wid], wid, queue))
        p.start()
        procs.append(p)

    # Gather
    all_times, all_tokens, all_sizes, split_counts = [], [], [], {}
    t_start = time.perf_counter()
    done = 0
    while done < total_samples:
        try:
            elapsed, nt, size, sp, wid = queue.get(timeout=120)
            all_times.append(elapsed)
            all_tokens.append(nt)
            all_sizes.append(size)
            split_counts[wid] = sp
            done += 1
        except Exception:
            break

    wall = time.perf_counter() - t_start
    for p in procs:
        p.join(timeout=5)

    warmup = int(len(all_times) * warmup_fraction)
    valid_times = all_times[warmup:]
    valid_tokens = all_tokens[warmup:]
    n = len(valid_times)

    if n == 0:
        print("No results!"); return

    mean_ms = statistics.mean(valid_times)
    thr = 1000 / mean_ms * num_workers  # aggregate throughput
    tok_thr = statistics.mean(valid_tokens) * 1000 / mean_ms * num_workers

    print(f"\n{'='*60}")
    print(f"  Workers: {num_workers}  Samples: {n} (warmup={warmup})  Wall: {wall:.1f}s")
    print(f"  Splits: {sum(split_counts.values())}")
    print(f"{'='*60}")
    print(f"  Per-sample mean:     {mean_ms:.1f} ms  (p50={statistics.median(valid_times):.1f} p95={_p95(valid_times):.1f})")
    print(f"  Aggregate throughput: {thr:.1f} samples/s")
    print(f"  Visual tokens/s:      {tok_thr:.0f}")
    print(f"  Avg tokens/sample:    {statistics.mean(valid_tokens):.0f}")
    print(f"{'='*60}")

    # Size distribution
    counts = {}
    for s in all_sizes[warmup:]:
        k = f"{s[0]}×{s[1]}"
        counts[k] = counts.get(k, 0) + 1
    print(f"\n  Size distribution:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {k:<10} {v:>4} ({v*100//n:>3}%)")

    return thr


def _p95(data): return sorted(data)[int(len(data) * 0.95)]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--count", type=int, default=200)
    args = p.parse_args()

    results = {}
    for w in args.workers:
        print(f"\n{'#'*60}")
        print(f"# {w} workers")
        print(f"{'#'*60}")
        r = benchmark_multi(args.manifest, w, args.count)
        if r:
            results[w] = r

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  Scaling summary:")
        print(f"{'='*60}")
        base = list(results.values())[0]
        for w, t in sorted(results.items()):
            eff = t / (w * base) * 100 if base else 0
            print(f"  {w} workers: {t:.1f} samples/s  (efficiency: {eff:.0f}%)")
