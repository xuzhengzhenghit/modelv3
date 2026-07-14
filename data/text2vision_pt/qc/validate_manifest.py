#!/usr/bin/env python3
"""Validate processed JSONL manifest — check for common issues."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def validate_record(record: dict[str, Any]) -> list[str]:
    """Validate a single record, return list of issues (empty = OK)."""
    issues = []

    # Required fields (compact format)
    sample_id = record.get("i", record.get("id", ""))
    text = record.get("t", record.get("target_text", ""))

    if not sample_id:
        issues.append("missing id")
    if not text:
        issues.append("empty target_text")

    # Check for control characters
    if text:
        ctrl_count = sum(1 for c in text if ord(c) < 32 and c not in "\n\t")
        if ctrl_count > 0:
            issues.append(f"control_chars={ctrl_count}")

    # Check for unmatched LaTeX delimiters
    if text:
        paren_open = text.count(r"\(") - text.count(r"\|")  # approximate
        paren_close = text.count(r"\)")
        bracket_open = text.count(r"\[")
        bracket_close = text.count(r"\]")
        if paren_open != paren_close:
            issues.append(f"latex_paren_mismatch: {paren_open} vs {paren_close}")
        if bracket_open != bracket_close:
            issues.append(f"latex_bracket_mismatch: {bracket_open} vs {bracket_close}")

    return issues


def validate_file(
    filepath: str,
    max_samples: int = 0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Validate a JSONL manifest file.

    Returns stats dict with total, valid, invalid counts and issue distribution.
    """
    issues_counter: Counter = Counter()
    total = 0
    valid = 0
    invalid = 0
    empty_text = 0
    sample_lengths: list[int] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if max_samples > 0 and total >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[ERROR] {filepath}:{line_num} — JSON decode error: {exc}")
                invalid += 1
                continue

            total += 1
            issues = validate_record(record)

            if issues:
                invalid += 1
                for issue in issues:
                    issues_counter[issue] += 1
                    if "empty" in issue:
                        empty_text += 1
                if verbose:
                    sample_id = record.get("i", record.get("id", ""))
                    print(f"[WARN] {sample_id}: {issues}")
            else:
                valid += 1

            text = record.get("t", record.get("target_text", ""))
            sample_lengths.append(len(text))

    return {
        "filepath": filepath,
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "empty_text": empty_text,
        "issues": dict(issues_counter.most_common(20)),
        "avg_text_len": sum(sample_lengths) / max(len(sample_lengths), 1),
        "max_text_len": max(sample_lengths) if sample_lengths else 0,
        "min_text_len": min(sample_lengths) if sample_lengths else 0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate Text2Vision-PT manifest")
    parser.add_argument("manifest_pattern", help="Glob pattern or file path")
    parser.add_argument("--max", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import glob as glob_mod
    files = sorted(Path(p) for p in glob_mod.glob(args.manifest_pattern))
    if not files:
        files = [Path(args.manifest_pattern)]

    for filepath in files:
        if not filepath.exists():
            print(f"[SKIP] {filepath} not found")
            continue
        stats = validate_file(str(filepath), max_samples=args.max, verbose=args.verbose)
        print(f"\n{stats['filepath']}")
        print(f"  total={stats['total']} valid={stats['valid']} invalid={stats['invalid']}")
        print(f"  empty_text={stats['empty_text']}")
        print(f"  text_len: avg={stats['avg_text_len']:.0f} min={stats['min_text_len']} max={stats['max_text_len']}")
        if stats["issues"]:
            print(f"  top_issues: {stats['issues']}")


if __name__ == "__main__":
    main()
