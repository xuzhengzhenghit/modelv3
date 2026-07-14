#!/usr/bin/env python3
"""PyTorch Dataset for page-level manifests rendered by persistent Chromium workers."""

from __future__ import annotations

import bisect
import json
import multiprocessing as mp
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from html_ocr_renderer import HtmlOCRRenderer, save_preview, stable_seed

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None


class IndexedJsonlFile:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.path.suffix.lower() in {".gz", ".bz2", ".xz"}:
            raise ValueError("Use uncompressed JSONL for random access")
        self.index_path = self.path.with_suffix(self.path.suffix + ".offsets.npy")
        self.meta_path = self.path.with_suffix(self.path.suffix + ".offsets.meta.json")
        self.lock_path = self.path.with_suffix(self.path.suffix + ".offsets.lock")
        self._handles: dict[int, Any] = {}
        self._ensure_index()
        self.offsets = np.load(self.index_path, mmap_mode="r")

    def _signature(self) -> dict[str, int]:
        stat = self.path.stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _is_current(self) -> bool:
        try:
            return self.index_path.exists() and json.loads(self.meta_path.read_text("utf-8")) == self._signature()
        except Exception:
            return False

    def _ensure_index(self) -> None:
        if self._is_current():
            return
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self._is_current():
                return
            offsets: list[int] = []
            with self.path.open("rb") as handle:
                while True:
                    position = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if line.strip():
                        offsets.append(position)
            with tempfile.NamedTemporaryFile(dir=self.index_path.parent, suffix=".npy", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            np.save(tmp_path, np.asarray(offsets, dtype=np.uint64))
            os.replace(tmp_path, self.index_path)
            temp_meta = self.meta_path.with_suffix(self.meta_path.suffix + ".tmp")
            temp_meta.write_text(json.dumps(self._signature()), "utf-8")
            os.replace(temp_meta, self.meta_path)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        pid = os.getpid()
        handle = self._handles.get(pid)
        if handle is None or handle.closed:
            handle = self.path.open("rb")
            self._handles[pid] = handle
        handle.seek(int(self.offsets[index]))
        value = json.loads(handle.readline())
        if not isinstance(value, dict):
            raise TypeError(f"Manifest row must be an object, got {type(value)}")
        return value

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state


class IndexedJsonlCorpus:
    def __init__(self, paths: Sequence[str | Path]):
        self.files = [IndexedJsonlFile(path) for path in paths]
        if not self.files:
            raise ValueError("At least one manifest is required")
        self.cumulative: list[int] = []
        total = 0
        for file in self.files:
            total += len(file)
            self.cumulative.append(total)

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        file_index = bisect.bisect_right(self.cumulative, index)
        previous = self.cumulative[file_index - 1] if file_index else 0
        return self.files[file_index][index - previous]


@dataclass(frozen=True)
class PreviewConfig:
    directory: str | None = None
    probability: float = 0.0
    max_per_worker: int = 20


class HtmlRenderedOCRDataset(Dataset):
    def __init__(
        self,
        manifest_paths: Sequence[str | Path],
        renderer: HtmlOCRRenderer,
        preview: PreviewConfig = PreviewConfig(),
        base_seed: int = 1234,
        rank: int = 0,
    ):
        self.corpus = IndexedJsonlCorpus(manifest_paths)
        self.renderer = renderer
        self.preview = preview
        self.base_seed = base_seed
        self.rank = rank
        self._epoch = mp.Value("q", 0)
        self._preview_counts: dict[tuple[int, int], int] = {}

    def __len__(self) -> int:
        return len(self.corpus)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def _maybe_preview(self, sample_id: str, result: dict[str, Any], record: dict[str, Any], rng: random.Random) -> None:
        if not self.preview.directory or self.preview.probability <= 0:
            return
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        key = (os.getpid(), worker_id)
        count = self._preview_counts.get(key, 0)
        if count >= self.preview.max_per_worker or rng.random() >= self.preview.probability:
            return
        save_preview(
            self.preview.directory,
            f"e{self._epoch.value:03d}_r{self.rank:02d}_w{worker_id:02d}_{sample_id}",
            result,
            {"manifest_target": record.get("target_text"), "source_file": record.get("source_file")},
        )
        self._preview_counts[key] = count + 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.corpus[index]
        sample_id = str(record.get("id", index))
        epoch = int(self._epoch.value)
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        seed = stable_seed(self.base_seed, epoch, self.rank, worker_id, sample_id)
        rng = random.Random(seed)

        blocks = record.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            # Compact format: parse the "t" field into blocks on-the-fly
            text = str(record.get("t") or record.get("text") or "")
            if not text.strip():
                raise ValueError(f"Manifest sample {sample_id} has no blocks and no text")
            from document_parser import parse_document
            subject = record.get("s") or record.get("subject")
            blocks = parse_document(text, str(subject) if subject else None, max_block_chars=420)
        if not isinstance(blocks, list) or not blocks:
            raise ValueError(f"Manifest sample {sample_id} has no blocks")

        result = self.renderer.render(blocks, seed)
        self._maybe_preview(sample_id, result, record, rng)
        result.pop("png_or_jpeg_bytes", None)
        result.update(
            {
                "id": sample_id,
                "doc_id": record.get("doc_id"),
                "subject": record.get("subject"),
                "has_math": bool(record.get("has_math")),
                "has_table": bool(record.get("has_table")),
            }
        )
        return result
