"""Rate limiting extension point.

Prototype hook only — not enforced by default. Wire middleware in main.py when
ready for production traffic control.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict


class InMemoryRateLimiter:
    """Simple sliding-window limiter suitable as a FastAPI dependency later."""

    def __init__(self, *, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.time()
        bucket = self._hits[key]
        cutoff = now - self.window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            return False
        bucket.append(now)
        return True
