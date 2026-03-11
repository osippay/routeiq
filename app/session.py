"""
RouteIQ — Session manager.

Pins a model for multi-turn conversations so that the user doesn't bounce
between models mid-thread (which breaks context and degrades quality).

Sessions expire after `ttl_seconds` of inactivity.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)

DEFAULT_TTL = 1800  # 30 minutes


@dataclass
class Session:
    session_id: str
    model_name: str
    task_type: str
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    request_count: int = 0

    def touch(self) -> None:
        self.last_used = time.time()
        self.request_count += 1


class SessionManager:
    """Manages model pinning per conversation session."""

    def __init__(self, ttl_seconds: float = DEFAULT_TTL, max_sessions: int = 1024) -> None:
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()

    def get(self, session_id: str) -> Session | None:
        """Get active session or None if expired / missing."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if time.time() - session.last_used > self._ttl:
                del self._sessions[session_id]
                logger.debug("Session expired: %s", session_id)
                return None
            return session

    def pin(self, session_id: str, model_name: str, task_type: str) -> Session:
        """Pin a model to a session. Creates or updates."""
        with self._lock:
            existing = self._sessions.get(session_id)
            if existing and time.time() - existing.last_used <= self._ttl:
                existing.touch()
                return existing

            session = Session(
                session_id=session_id,
                model_name=model_name,
                task_type=task_type,
            )
            self._sessions[session_id] = session

            # Evict oldest if over limit
            if len(self._sessions) > self._max:
                oldest_id = min(self._sessions, key=lambda k: self._sessions[k].last_used)
                del self._sessions[oldest_id]

            logger.debug("Session pinned: %s → %s", session_id, model_name)
            return session

    def remove(self, session_id: str) -> None:
        """Remove a session."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup(self) -> int:
        """Remove all expired sessions. Returns count removed."""
        now = time.time()
        with self._lock:
            expired = [sid for sid, s in self._sessions.items()
                       if now - s.last_used > self._ttl]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)

    def stats(self) -> dict:
        """Active session stats."""
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": self._max,
                "ttl_seconds": self._ttl,
            }
