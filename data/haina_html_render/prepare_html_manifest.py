#!/usr/bin/env python3
"""Convert JSON/JSONL scientific text into page-level HTML-render manifest JSONL."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from document_parser import join_targets, pack_pages, parse_document


def nested_get(record: Any, path: str) -> Any:
    current = record
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[WARN] {path}:{line_no}: invalid JSON: {exc}")
                    continue
                yield value if isinstance(value, dict) else {"text": value}
    else:
        value = json.loads(path.read_text("utf-8", errors="replace"))
        if isinstance(value, list):
            for item in value:
                yield item if isinstance(item, dict) else {"text": item}
        elif isinstance(value, dict):
            for key in ("data", "records", "items", "documents"):
                if isinstance(value.get(key), list):
                    for item in value[key]:
                        yield item if isinstance(item, dict) else {"text": item}
                    return
            yield value
        else:
            yield {"text": value}


def pick_field(record: dict[str, Any], fields: list[str]) -> Any:
    for field in fields:
        value = nested_get(record, field)
        if value is not None and str(value).strip():
            return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="JSON/JSONL paths or glob patterns")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-fields", default="text,content,body,abstract,document.text")
    parser.add_argument("--id-fields", default="id,doc_id,uuid")
    parser.add_argument("--subject-fields", default="subject,category,discipline,field")
    parser.add_argument("--page-budget", type=int, default=950)
    parser.add_argument("--max-block-chars", type=int, default=420)
    parser.add_argument("--min-target-chars", type=int, default=20)
    parser.add_argument("--max-docs", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text_fields = [x.strip() for x in args.text_fields.split(",") if x.strip()]
    id_fields = [x.strip() for x in args.id_fields.split(",") if x.strip()]
    subject_fields = [x.strip() for x in args.subject_fields.split(",") if x.strip()]

    paths: list[Path] = []
    for pattern in args.inputs:
        matches = [Path(x) for x in glob.glob(pattern, recursive=True)]
        paths.extend(matches or [Path(pattern)])
    paths = sorted({p.resolve() for p in paths if p.exists()})
    if not paths:
        raise FileNotFoundError("No input files matched")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    doc_counter = 0

    with args.output.open("w", encoding="utf-8") as output:
        for path in paths:
            print(f"[FILE] {path}")
            for record_index, record in enumerate(iter_records(path)):
                text = pick_field(record, text_fields)
                if text is None:
                    stats["missing_text"] += 1
                    continue
                subject = pick_field(record, subject_fields)
                raw_id = pick_field(record, id_fields)
                doc_id = str(raw_id or f"{path.stem}-{record_index:08d}")

                blocks = parse_document(text, str(subject) if subject is not None else None, args.max_block_chars)
                pages = pack_pages(blocks, args.page_budget)
                if not pages:
                    stats["empty"] += 1
                    continue

                doc_counter += 1
                for page_index, page_blocks in enumerate(pages):
                    target = join_targets(page_blocks)
                    if len(target) < args.min_target_chars:
                        stats["too_short_page"] += 1
                        continue
                    kinds = Counter(block["kind"] for block in page_blocks)
                    payload = {
                        "id": f"{doc_id}-p{page_index:05d}",
                        "doc_id": doc_id,
                        "page_index": page_index,
                        "subject": subject,
                        "blocks": page_blocks,
                        "target_text": target,
                        "has_math": bool(kinds["math_display"] or any(
                            part.get("kind") == "math_inline"
                            for block in page_blocks if block.get("kind") == "paragraph"
                            for part in block.get("parts", [])
                        )),
                        "has_table": bool(kinds["table"]),
                        "source_file": path.name,
                        "sha1": hashlib.sha1(target.encode("utf-8")).hexdigest(),
                    }
                    output.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    stats["pages"] += 1
                    stats["math_pages"] += int(payload["has_math"])
                    stats["table_pages"] += int(payload["has_table"])

                stats["documents"] += 1
                if args.max_docs and doc_counter >= args.max_docs:
                    break
            if args.max_docs and doc_counter >= args.max_docs:
                break

    print("\n========== manifest summary ==========")
    for key in sorted(stats):
        print(f"{key:<20}: {stats[key]}")
    print(f"output              : {args.output.resolve()}")
    print("======================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
