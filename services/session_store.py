"""
In-memory session store.
Maps session_id → list of Candidate dicts.
No external dependencies — works without Supabase.
Phase 4: swap with Supabase-backed store.
"""

from __future__ import annotations
from typing import Any

_store: dict[str, list[dict]] = {}
_selected: dict[str, dict] = {}  # session_id → the candidate the user picked


def set_selected(session_id: str, candidate: dict | None) -> None:
    """Remember the candidate the user selected (or clear with None)."""
    if candidate is None:
        _selected.pop(session_id, None)
    else:
        _selected[session_id] = dict(candidate)


def get_selected(session_id: str) -> dict | None:
    """Return the user's selected candidate for this session, or None."""
    return _selected.get(session_id)


def set_candidates(session_id: str, candidates: list[Any]) -> None:
    """Store candidate list for a session. Replaces previous value."""
    _store[session_id] = [
        c.model_dump() if hasattr(c, "model_dump") else dict(c)
        for c in candidates
    ]


def get_candidates(session_id: str) -> list[dict]:
    """Return candidates for session_id, or [] if not found."""
    return _store.get(session_id, [])


def get_candidate(session_id: str, index: int) -> dict | None:
    """Return single candidate by index, or None."""
    candidates = get_candidates(session_id)
    if 0 <= index < len(candidates):
        return candidates[index]
    return None


def clear(session_id: str) -> None:
    """Remove session."""
    _store.pop(session_id, None)
