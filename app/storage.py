"""
RouteIQ — Atomic file storage helpers.

Thread-safe read/write for JSON and JSONL files.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

_locks: dict[str, Lock] = {}
_locks_guard = Lock()


def _get_lock(path: str) -> Lock:
    key = str(Path(path).absolute())
    with _locks_guard:
        if key not in _locks:
            _locks[key] = Lock()
        return _locks[key]


def _ensure_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.exception("Failed to create dirs for %s: %s", path, e)


def atomic_read(path: str) -> dict:
    """Read JSON file atomically. Returns {} on any failure."""
    p = Path(path)
    with _get_lock(path):
        if not p.exists():
            return {}
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {"_data": data}
        except Exception as e:
            logger.exception("atomic_read failed for %s: %s", p, e)
            return {}


def atomic_write(path: str, data: dict) -> None:
    """Write dict to JSON file atomically (write-to-tmp + replace)."""
    p = Path(path)
    _ensure_parent(p)
    tmp = Path(str(p) + ".tmp")
    with _get_lock(path):
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp), str(p))
        except Exception as e:
            logger.exception("atomic_write failed for %s: %s", p, e)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass


def append_jsonl(path: str, record: dict) -> None:
    """Append a single JSON record to a JSONL file."""
    p = Path(path)
    _ensure_parent(p)
    with _get_lock(path):
        try:
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.exception("append_jsonl failed for %s: %s", p, e)


def rotate_if_needed(path: str, max_mb: float = 10.0) -> None:
    """Rotate file if it exceeds max_mb."""
    p = Path(path)
    if not p.exists():
        return
    with _get_lock(path):
        try:
            size_mb = p.stat().st_size / (1024 * 1024)
            if size_mb <= max_mb:
                return
            old = Path(str(p) + ".old")
            old.unlink(missing_ok=True)
            os.replace(str(p), str(old))
            logger.info("Rotated %s (%.1fMB)", p, size_mb)
        except Exception as e:
            logger.exception("rotate failed for %s: %s", p, e)
