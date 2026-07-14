#!/usr/bin/env python3
"""Prefetch queue — decouples rendering from GPU training.

The renderer workers populate a prefetch queue so GPU training doesn't block on
Chromium screenshot latency. Samples are kept as uint8 tensors in CPU memory.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Iterator, Optional


class PrefetchQueue:
    """Thread-safe prefetch queue for rendered samples."""

    def __init__(self, maxsize: int = 64):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stopped = threading.Event()

    def put(self, item: Any, timeout: Optional[float] = None):
        """Put a sample into the queue, blocking if full."""
        if self._stopped.is_set():
            return False
        try:
            self._queue.put(item, timeout=timeout)
            return True
        except queue.Full:
            return False

    def get(self, timeout: Optional[float] = None) -> Any:
        """Get a sample from the queue, blocking if empty."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._stopped.set()

    @property
    def qsize(self) -> int:
        try:
            return self._queue.qsize()
        except Exception:
            return 0

    def __len__(self) -> int:
        return self.qsize


class PrefetchLoader:
    """Wraps a dataset iterator with a prefetch thread.

    The render thread fetches samples ahead and puts them in the queue.
    The training loop reads from the queue without blocking on rendering.
    """

    def __init__(self, dataset_iter: Iterator, prefetch_size: int = 8):
        self._iter = dataset_iter
        self._queue = PrefetchQueue(maxsize=prefetch_size)
        self._thread: Optional[threading.Thread] = None
        self._prefetch_size = prefetch_size

    def _fetch_loop(self):
        try:
            for sample in self._iter:
                if self._queue._stopped.is_set():
                    break
                self._queue.put(sample)
        except Exception:
            pass
        finally:
            self._queue.put(None)  # Sentinel

    def start(self):
        self._thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._queue.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __iter__(self) -> Iterator[Any]:
        self.start()
        try:
            while True:
                item = self._queue.get(timeout=30.0)
                if item is None:
                    break
                yield item
        finally:
            self.stop()

    @property
    def queue_depth(self) -> int:
        return len(self._queue)
