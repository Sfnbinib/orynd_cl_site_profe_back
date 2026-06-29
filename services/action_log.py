"""Action Log — records every tool call from the workspace agent loop.

This is the "Practice Store" feed: every tool invocation (search, build, mesh, etc.)
is written here so the Learning Engine (#42) can compare practice vs theory.

Architecture:
  workspace.py tool-loop
    → action_log.write(tool_name, params, result, session_id)
    → stored in _log (in-memory ring buffer, max 500 entries)
    → also published on event_bus as "action.recorded" so any subscriber
      (Learning Engine, Telemetry, future Supabase sink) can react async

In Phase 1: in-memory only (no DB). A Supabase sink is added in Phase 2
by subscribing to "action.recorded" and writing to the action_events table.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any

from orynd_core.services.event_bus import bus

_MAX_ENTRIES = 500
_log: deque[dict] = deque(maxlen=_MAX_ENTRIES)


async def write(
    tool_name: str,
    params: dict[str, Any],
    result: Any,
    session_id: str = "anonymous",
) -> None:
    """Record a tool call. Non-blocking — errors are swallowed so the main loop never breaks."""
    entry = {
        "tool": tool_name,
        "params": params,
        "result": result if isinstance(result, dict) else {"raw": str(result)[:500]},
        "session_id": session_id,
        "ts": time.time(),
    }
    _log.append(entry)

    # Publish async — Learning Engine and other subscribers receive this
    try:
        await bus.publish("action.recorded", entry)
    except Exception:
        pass


def recent(limit: int = 50, session_id: str | None = None) -> list[dict]:
    """Return recent entries, optionally filtered by session."""
    entries = list(_log)
    if session_id:
        entries = [e for e in entries if e["session_id"] == session_id]
    return entries[-limit:]


def clear(session_id: str | None = None) -> int:
    """Clear log entries. Returns number removed."""
    global _log
    if session_id is None:
        count = len(_log)
        _log.clear()
        return count
    before = len(_log)
    _log = deque(
        (e for e in _log if e["session_id"] != session_id),
        maxlen=_MAX_ENTRIES,
    )
    return before - len(_log)
