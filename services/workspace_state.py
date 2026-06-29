"""Workspace State — shared backend runStore per workspace_id.

Every chat surface (bottom chat, left agent panel, Jarvis) that belongs
to the same workspace shares this state. When bottom chat finds candidates
→ left panel sees them. When a model is built → all surfaces get model_ready.

Phase 1: in-memory. Phase 2: Supabase row + realtime subscription.

Wire-in points:
  routers/chat.py     → update() after candidates / model_ready events
  routers/multi_context.py → update() after /chip
  GET /context/workspace/{workspace_id} → Visual Orchestrator reads current state
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from orynd_core.services.event_bus import bus

_DEFAULT_STATE: dict[str, Any] = {
    "selected_model": None,
    "candidates": [],
    "active_tasks": [],
    "context_chips": [],
    "last_tool": None,
    "updated_at": 0.0,
}

# workspace_id → state dict
_states: dict[str, dict[str, Any]] = defaultdict(lambda: dict(_DEFAULT_STATE))


async def update(workspace_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    """Merge patch into workspace state, publish workspace.event on bus."""
    state = _states[workspace_id]

    # List fields: append items rather than replace
    for list_key in ("context_chips", "active_tasks"):
        if list_key in patch and isinstance(patch[list_key], list):
            existing = state.get(list_key) or []
            state[list_key] = (existing + patch.pop(list_key))[-20:]  # keep last 20

    state.update(patch)
    state["workspace_id"] = workspace_id
    state["updated_at"] = time.time()

    await bus.publish("workspace.event", {
        "workspace_id": workspace_id,
        "patch": patch,
        "state": dict(state),
    })

    return dict(state)


def get(workspace_id: str) -> dict[str, Any]:
    s = dict(_states[workspace_id])
    s["workspace_id"] = workspace_id
    return s


def clear(workspace_id: str) -> None:
    _states.pop(workspace_id, None)
