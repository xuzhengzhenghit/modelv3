#!/usr/bin/env python3
"""Re-package existing compact JSONL pages into Text2Vision-PT shards.

Input: onesci_cc_pages (compact: {i, t, s})
Output: text2vision_pt shards (compact: {i, t, s, f})

Adds `f` (flags bitmask): 1=has_math, 2=has_table, 4=has_html
Does SHA1 dedup and train/val/test split.

Usage:
  python repackage_shards.py \
    --input-glob '/data/onesci_cc_pages/pages-0000[1-3].jsonl' \
    --output-dir /tmp/t2v_test \
    --max-pages 50000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# ── Detect math, table, HTML content without full parse ──
LATEX_INLINE_RE = re.compile(r"\\\(.*?\\\)")
LATEX_DISPLAY_RE = re.compile(r"\\\[.*?\\\]")
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
TABLE_TSV_RE = re.compile(r"^.+\t.+$", re.MULTILINE)  # TSV-format tables


def compute_flags(text: str) -> int:
    """Fast flags computation from text alone.

    1 = has_math (\\( or \\[)
    2 = has_table (TSV lines or Markdown pipe tables)
    4 = has_html (<tag>)
    """
    flags = 0
    if LATEX_INLINE_RE.search(text) or LATEX_DISPLAY_RE.search(text):
        flags |= 1
    # Quick table check: tab-separated content spanning multiple lines
    tab_lines = [l for l in text.split("\n") if "\t" in l and len(l) > 10]
    if len(tab_lines) >= 2:
        flags |= 2
    # Pipe table
    if re.search(r"^\|.+\|.+\|$", text, re.MULTILINE):
        flags |= 2
    if HTML_TAG_RE.search(text):
        flags |= 4
    return flags


class ShardWriter:
    def __init__(self, directory: Path, max_size_mb: float, prefix: str):
        self._dir = directory
        self._max_bytes = int(max_size_mb * 1024 * 1024)
        self._prefix = prefix
        self._idx = 0
        self._current_bytes = 0
        self._handle = None

    def write(self, line: str):
        line_bytes = len(line.encode("utf-8"))
        if self._handle is None or self._current_bytes + line_bytes > self._max_bytes:
            if self._handle:
                self._handle.close()
            path = self._dir / f"{self._prefix}-{self._idx:05d}.jsonl"
            self._handle = open(path, "w", encoding="utf-8")
            self._current_bytes = 0
            self._idx += 1
        self._handle.write(line)
        self._current_bytes += line_bytes

    def close(self):
        if self._handle:
            self._handle.close()
            self._handle = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-size-mb", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.001)
    parser.add_argument("--test-ratio", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pages", type=int, default=0)
    args = parser.parse_args()

    import glob as glob_mod

    input_files = sorted(glob_mod.glob(args.input_glob))
    if not input_files:
        print(f"[ERROR] No files matched: {args.input_glob}")
        return 1

    out = Path(args.output_dir)
    for sub in ["train", "val", "test", "rejects"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    train_writer = ShardWriter(out / "train", args.shard_size_mb, "train")
    val_writer = ShardWriter(out / "val", args.shard_size_mb, "val")
    test_writer = ShardWriter(out / "test", args.shard_size_mb, "test")

    rng = random.Random(args.seed)
    seen_hashes: set[str] = set()

    total = 0
    valid = 0
    deduped = 0
    rejected = 0
    train_n = 0
    val_n = 0
    test_n = 0

    t0 = time.perf_counter()

    for filepath in input_files:
        print(f"  processing: {Path(filepath).name} ...")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                if args.max_pages > 0 and valid >= args.max_pages:
                    break

                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    rejected += 1
                    continue

                text = rec.get("t", "")
                if not text:
                    rejected += 1
                    continue

                # Dedup
                text_hash = hashlib.sha1(text.encode()).hexdigest()
                if text_hash in seen_hashes:
                    deduped += 1
                    continue
                seen_hashes.add(text_hash)

                # Compute flags
                flags = compute_flags(text)

                # Build output record
                out_rec = {
                    "i": rec.get("i", ""),
                    "t": text,
                    "s": rec.get("s", ""),
                    "f": flags,
                }

                # Train/val/test split
                r = rng.random()
                if r < args.test_ratio:
                    test_writer.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    test_n += 1
                elif r < args.test_ratio + args.val_ratio:
                    val_writer.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    val_n += 1
                else:
                    train_writer.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    train_n += 1
                valid += 1

                if valid % 50000 == 0 and valid > 0:
                    elapsed = time.perf_counter() - t0
                    print(f"    {valid:,} records processed ({valid / elapsed:.0f} rec/s)")

            if args.max_pages > 0 and valid >= args.max_pages:
                break

    train_writer.close()
    val_writer.close()
    test_writer.close()

    elapsed = time.perf_counter() - t0

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"  Input files:        {len(input_files)}")
    print(f"  Total lines:        {total:,}")
    print(f"  Valid pages:        {valid:,}")
    print(f"  Duplicates skipped: {deduped:,}")
    print(f"  Rejected:           {rejected:,}")
    print(f"  Train: {train_n:,}  Val: {val_n:,}  Test: {test_n:,}")
    print(f"  Time:  {elapsed:.1f}s  ({valid / elapsed:.0f} rec/s)")
    print(f"{'='*55}")

    # ── Show output ──
    print(f"\nOutput files:")
    for sub in ["train", "val", "test"]:
        d = out / sub
        files = sorted(d.glob("*.jsonl"))
        total_size = sum(f.stat().st_size for f in files)
        print(f"  {sub}/ ({len(files)} shards, {total_size / 1024 / 1024:.1f} MB)")

    # ── Sample records ──
    print(f"\nSample records:")
    first_shard = out / "train" / "train-00000.jsonl"
    if first_shard.exists():
        with open(first_shard) as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                rec = json.loads(line)
                flags_str = []
                if rec["f"] & 1:
                    flags_str.append("math")
                if rec["f"] & 2:
                    flags_str.append("table")
                if rec["f"] & 4:
                    flags_str.append("html")
                flag_label = "+".join(flags_str) if flags_str else "plain"
                print(f"  [{rec['s']}][{flag_label}] {rec['t'][:100]}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
