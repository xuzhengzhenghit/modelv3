#!/usr/bin/env python3
"""Build Text2Vision-PT manifest: raw text → JSONL shards.

This is the entry point for Stage 1 (Corpus Preprocessor).
It replaces the old prepare_html_manifest.py with:
    - streaming gzip input
    - Unicode normalization
    - HTML sanitization
    - Document parsing and chunking (reuses document_parser)
    - Sharded JSONL output (compact format: {i, t, s, f})
    - SHA1 dedup (optional)
    - Validation split

Usage:
  python build_manifest.py \
    --input-glob '/data/**/*.jsonl.gz' \
    --output-dir /data/text2vision_pt/train \
    --shard-size-mb 384 \
    --val-ratio 0.001
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
T2V_DIR = SCRIPT_DIR.parent

from .normalizer import normalize_text
from .html_sanitizer import contains_html, has_html_risks
from .gzip_reader import detect_format, open_lines, read_jsonl


def build_manifest(
    input_glob: str,
    output_dir: str,
    shard_size_mb: float = 384,
    val_ratio: float = 0.001,
    test_ratio: float = 0.0005,
    seed: int = 42,
    max_docs: int = 0,
    page_budget: int = 950,
    max_block_chars: int = 420,
    text_fields: list[str] | None = None,
    subject_fields: list[str] | None = None,
    dedup: bool = True,
    compact: bool = True,
) -> dict[str, Any]:
    """Build manifest shards from raw text data.

    Returns stats dict with counts, sizes, timing.
    """
    import glob
    import random
    import time

    from .document_chunker import compact_record, document_to_pages, full_record

    text_fields = text_fields or ["text", "content", "body", "abstract"]
    subject_fields = subject_fields or ["subject", "category", "discipline", "final_subjects"]

    t0 = time.perf_counter()

    # ── Setup output dirs ──
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val"
    test_dir = Path(output_dir) / "test"
    rejects_dir = Path(output_dir) / "rejects"
    for d in [train_dir, val_dir, test_dir, rejects_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    seen_hashes: set[str] = set()

    # ── Open shard writers ──
    train_writer = _ShardWriter(train_dir, shard_size_mb, "train")
    val_writer = _ShardWriter(val_dir, shard_size_mb, "val")
    test_writer = _ShardWriter(test_dir, shard_size_mb, "test")

    stats = {
        "docs_processed": 0,
        "pages_generated": 0,
        "pages_train": 0,
        "pages_val": 0,
        "pages_test": 0,
        "pages_rejected": 0,
        "dedup_skipped": 0,
        "parse_errors": 0,
        "html_risks": 0,
    }

    input_files = sorted(glob.glob(input_glob, recursive=True))
    if not input_files:
        print(f"[WARN] No files matched: {input_glob}")
        return stats

    print(f"[build_manifest] {len(input_files)} input files → {output_dir}")

    for file_idx, filepath in enumerate(input_files):
        try:
            for record in read_jsonl(filepath):
                if max_docs > 0 and stats["docs_processed"] >= max_docs:
                    break

                # ── Extract text ──
                raw_text = ""
                for field in text_fields:
                    if field in record:
                        val = record[field]
                        if isinstance(val, list):
                            raw_text = "\n".join(str(x) for x in val)
                        elif isinstance(val, str):
                            raw_text = val
                        if raw_text:
                            break

                if not raw_text:
                    continue

                # ── Extract subject ──
                subject = ""
                for field in subject_fields:
                    if field in record:
                        val = record[field]
                        if isinstance(val, list):
                            subject = str(val[0]) if val else ""
                        elif isinstance(val, str):
                            subject = val
                        if subject:
                            break

                # ── Extract doc_id ──
                doc_id = record.get("id", record.get("doc_id", ""))
                if not doc_id:
                    doc_id = hashlib.sha1(raw_text[:200].encode()).hexdigest()[:12]

                # ── Normalize ──
                text = normalize_text(raw_text)
                if not text:
                    continue

                # ── Dedup ──
                if dedup:
                    text_hash = hashlib.sha1(text.encode()).hexdigest()
                    if text_hash in seen_hashes:
                        stats["dedup_skipped"] += 1
                        continue
                    seen_hashes.add(text_hash)

                # ── Security check ──
                if contains_html(text) and has_html_risks(text):
                    stats["html_risks"] += 1
                    # Still process, but flag

                # ── Parse and chunk ──
                try:
                    pages = document_to_pages(
                        text, doc_id, subject, page_budget, max_block_chars
                    )
                except Exception:
                    stats["parse_errors"] += 1
                    continue

                stats["docs_processed"] += 1

                for page in pages:
                    # ── Split into train/val/test ──
                    rand_val = rng.random()
                    if rand_val < test_ratio:
                        writer = test_writer
                        stats["pages_test"] += 1
                    elif rand_val < test_ratio + val_ratio:
                        writer = val_writer
                        stats["pages_val"] += 1
                    else:
                        writer = train_writer
                        stats["pages_train"] += 1

                    rec = compact_record(page) if compact else full_record(page)
                    writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["pages_generated"] += 1

                if file_idx % 20 == 0 and file_idx > 0:
                    elapsed = time.perf_counter() - t0
                    rate = stats["docs_processed"] / max(elapsed, 0.001)
                    print(f"  [{file_idx}/{len(input_files)}] "
                          f"docs={stats['docs_processed']} "
                          f"pages={stats['pages_generated']} "
                          f"({rate:.0f} docs/s)")

        except Exception as exc:
            print(f"[ERROR] {filepath}: {exc}")
            continue

        if max_docs > 0 and stats["docs_processed"] >= max_docs:
            break

    # ── Finalize ──
    train_writer.close()
    val_writer.close()
    test_writer.close()

    elapsed = time.perf_counter() - t0
    stats["elapsed_seconds"] = elapsed
    stats["docs_per_second"] = stats["docs_processed"] / max(elapsed, 0.001)
    stats["pages_per_second"] = stats["pages_generated"] / max(elapsed, 0.001)

    print(f"\n[build_manifest] Done in {elapsed:.1f}s")
    print(f"  docs: {stats['docs_processed']}  pages: {stats['pages_generated']}")
    print(f"  train: {stats['pages_train']}  val: {stats['pages_val']}  test: {stats['pages_test']}")
    print(f"  dedup_skipped: {stats['dedup_skipped']}  parse_errors: {stats['parse_errors']}")
    print(f"  rate: {stats['pages_per_second']:.0f} pages/s")

    return stats


class _ShardWriter:
    def __init__(self, directory: Path, max_size_mb: float, prefix: str):
        self._dir = directory
        self._max_bytes = int(max_size_mb * 1024 * 1024)
        self._prefix = prefix
        self._idx = 0
        self._current_bytes = 0
        self._handle = None

    def _open_next(self):
        if self._handle:
            self._handle.close()
        path = self._dir / f"{self._prefix}-{self._idx:05d}.jsonl"
        self._handle = open(path, "w", encoding="utf-8")
        self._current_bytes = 0
        self._idx += 1

    def write(self, line: str):
        if self._handle is None or self._current_bytes + len(line) > self._max_bytes:
            self._open_next()
        self._handle.write(line)
        self._current_bytes += len(line.encode("utf-8"))

    def close(self):
        if self._handle:
            self._handle.close()
            self._handle = None


def main():
    parser = argparse.ArgumentParser(description="Build Text2Vision-PT manifest")
    parser.add_argument("--input-glob", required=True, help="Glob pattern for input JSONL/JSONL.GZ files")
    parser.add_argument("--output-dir", required=True, help="Output directory for shards")
    parser.add_argument("--shard-size-mb", type=int, default=384)
    parser.add_argument("--val-ratio", type=float, default=0.001)
    parser.add_argument("--test-ratio", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--page-budget", type=int, default=950)
    parser.add_argument("--max-block-chars", type=int, default=420)
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--full-format", action="store_true", help="Use full (non-compact) record format")

    args = parser.parse_args()
    build_manifest(
        input_glob=args.input_glob,
        output_dir=args.output_dir,
        shard_size_mb=args.shard_size_mb,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_docs=args.max_docs,
        page_budget=args.page_budget,
        max_block_chars=args.max_block_chars,
        dedup=not args.no_dedup,
        compact=not args.full_format,
    )


if __name__ == "__main__":
    main()
