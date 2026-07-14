#!/usr/bin/env bash
set -euo pipefail

BROWSER_PATH="${PLAYWRIGHT_CHROMIUM_EXECUTABLE:-/usr/bin/chromium}"
KATEX_DIST="${KATEX_DIST:-./node_modules/katex/dist}"

python benchmark_html_render.py \
  --count 50 \
  --warmup 5 \
  --browser-path "$BROWSER_PATH" \
  --katex-dist "$KATEX_DIST" \
  --tensor-mode uint8 \
  --save-first 6 \
  --compare-reload

python prepare_html_manifest.py \
  sample_source.jsonl \
  --output sample_manifest.jsonl \
  --page-budget 700

python benchmark_dataloader.py \
  --manifest sample_manifest.jsonl \
  --workers 2 \
  --batch-size 2 \
  --batches 20 \
  --warmup-batches 3 \
  --browser-path "$BROWSER_PATH" \
  --katex-dist "$KATEX_DIST" \
  --output-mode uint8 \
  --preview-dir quick_test_preview
