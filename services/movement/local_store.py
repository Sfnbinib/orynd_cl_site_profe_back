"""In-memory movement signal store.

Phase 1+ will swap this for a SQLite-backed store under ``~/.orynd/movement.db``
so signals survive restarts and feed federated training. The interface stays
the same so callers don't need to change.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Optional
from uuid import UUID

from orynd_core.services.movement.signals import MovementSignal


class MovementStore:
    def __init__(self) -> None:
        self._signals: list[MovementSignal] = []
        self._by_session: dict[UUID, list[MovementSignal]] = defaultdict(list)
        self._by_user: dict[UUID, list[MovementSignal]] = defaultdict(list)
        self._lock = threading.Lock()

    def record(self, signal: MovementSignal) -> MovementSignal:
        with self._lock:
            self._signals.append(signal)
            self._by_session[signal.session_id].append(signal)
            if signal.user_id is not None:
                self._by_user[signal.user_id].append(signal)
        return signal

    def list_session(self, session_id: UUID, limit: int = 200) -> list[MovementSignal]:
        return list(self._by_session.get(session_id, []))[-limit:]

    def list_user(self, user_id: UUID, limit: int = 500) -> list[MovementSignal]:
        return list(self._by_user.get(user_id, []))[-limit:]

    def all_signals(self) -> list[MovementSignal]:
        return list(self._signals)

    def clear(self) -> None:
        with self._lock:
            self._signals.clear()
            self._by_session.clear()
            self._by_user.clear()

    def session_count(self) -> int:
        return len(self._by_session)

    def total(self) -> int:
        return len(self._signals)


_store: Optional[MovementStore] = None
_lock = threading.Lock()


def get_movement_store() -> MovementStore:
    global _store
    if _store is not None:
        return _store
    with _lock:
        if _store is None:
            _store = MovementStore()
        return _store


def reset_movement_store() -> None:
    global _store
    with _lock:
        _store = None


__all__ = ["MovementStore", "get_movement_store", "reset_movement_store"]
