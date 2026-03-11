"""
RouteIQ — LRU response cache.

Caches LLM responses keyed by (model, messages_hash).
Identical prompts skip the API call entirely → $0 cost, ~0ms latency.

Thread-safe via threading.Lock.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ResponseCache:
    """In-memory LRU cache for LLM responses."""

    def __init__(self, max_size: int = 256, ttl_seconds: float = 3600.0) -> None:
        self._max_size = max(1, max_size)
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(messages: list[dict], model: str | None = None) -> str:
        """Deterministic hash of messages + optional model."""
        blob = json.dumps({"m": messages, "model": model}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, messages: list[dict], model: str | None = None) -> Optional[dict]:
        """Return cached response or None."""
        key = self._make_key(messages, model)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            # Check TTL
            if time.time() - entry["ts"] > self._ttl:
                del self._store[key]
                self._misses += 1
                logger.debug("Cache EXPIRED for key %s", key[:12])
                return None

            # Move to end (LRU)
            self._store.move_to_end(key)
            self._hits += 1
            logger.debug("Cache HIT for key %s", key[:12])
            return entry["response"]

    def put(self, messages: list[dict], response: dict, model: str | None = None) -> None:
        """Store a response in cache."""
        key = self._make_key(messages, model)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = {"response": response, "ts": time.time()}
                return

            self._store[key] = {"response": response, "ts": time.time()}

            # Evict oldest if over capacity
            while len(self._store) > self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("Cache EVICT key %s", evicted_key[:12])

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
                "ttl_seconds": self._ttl,
            }
