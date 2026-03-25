"""
SessionStore — thread-safe in-memory store with TTL expiry.
Swap the backend (Redis, DB) by replacing get/save/delete.
"""

import threading
from datetime import datetime, timedelta
from typing import Optional

from models import SessionState
from config import settings


class SessionStore:
    def __init__(self, max_sessions: int = 500):
        self._store: dict[str, tuple[SessionState, datetime]] = {}
        self._lock = threading.Lock()
        self._max = max_sessions
        self._ttl = timedelta(minutes=settings.session_ttl_minutes)

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            entry = self._store.get(session_id)
            if not entry:
                return None
            session, created_at = entry
            if datetime.utcnow() - created_at > self._ttl:
                del self._store[session_id]
                return None
            return session

    def save(self, session_id: str, session: SessionState) -> None:
        with self._lock:
            # Evict oldest if at capacity
            if session_id not in self._store and len(self._store) >= self._max:
                oldest_key = min(self._store, key=lambda k: self._store[k][1])
                del self._store[oldest_key]
            # Preserve original creation time if updating
            created_at = self._store.get(session_id, (None, datetime.utcnow()))[1]
            self._store[session_id] = (session, created_at)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    def count(self) -> int:
        with self._lock:
            return len(self._store)

    def purge_expired(self) -> int:
        """Call periodically to free memory. Returns number purged."""
        cutoff = datetime.utcnow() - self._ttl
        with self._lock:
            expired = [k for k, (_, ts) in self._store.items() if ts < cutoff]
            for k in expired:
                del self._store[k]
        return len(expired)
