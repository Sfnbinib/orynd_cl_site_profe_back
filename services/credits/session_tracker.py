"""Session credits tracker — in-memory cost accumulator per session.

Phase 1: in-memory only (no Supabase). Tracks cost of every tool call
in the current process. A Supabase sink is added in Phase 2 by subscribing
to "credits.consumed" on the event_bus.

Wire-in point:
  workspace.py _execute_tool() → session_tracker.record(tool_name, tool_input, session_id)
  chat.py _stream() → session_tracker.get_session(session_id) → emit credits_update event
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from orynd_core.services.credits.pricing import PricingError, quote_action
from orynd_core.services.event_bus import bus

# tool_name (workspace) → pricing action name
_TOOL_TO_ACTION: dict[str, str] = {
    "search_models": "deep_search",
    "analyze_image": "vision_analyze",
    "get_fabrication": "fabricate",
    "deep_research": "deep_research",
    "analyze_mesh": "mesh_analyze",
    "build_3d_model": "cad_execute",
    "select_model": None,   # free — no charge
}

# session_id → {total_cost, tool_calls: [{tool, cost, ts}]}
_sessions: dict[str, dict[str, Any]] = defaultdict(
    lambda: {"total_cost": 0, "tool_calls": []}
)


def _params_for_tool(tool_name: str, tool_input: dict) -> dict:
    """Extract pricing-relevant params from tool input."""
    if tool_name == "build_3d_model":
        ops = tool_input.get("operations") or []
        return {"operation_count": len(ops)}
    if tool_name == "deep_research":
        depth = int(tool_input.get("depth", 2) or 2)
        return {"phase_count": depth + 2}  # depth maps roughly to phases
    if tool_name == "analyze_mesh":
        return {}  # file size unknown at this point
    return {}


async def record(tool_name: str, tool_input: dict, session_id: str = "anonymous") -> int:
    """Record a tool call, compute its cost, publish credits.consumed. Returns cost."""
    action = _TOOL_TO_ACTION.get(tool_name)
    if action is None:
        return 0  # free tool

    try:
        params = _params_for_tool(tool_name, tool_input)
        quote = quote_action(action, params)
        cost = quote.cost
    except PricingError:
        cost = 1  # unknown action → minimal cost

    session = _sessions[session_id]
    session["total_cost"] += cost
    session["tool_calls"].append({"tool": tool_name, "cost": cost})

    # Publish so SSE and future Supabase sink can react
    await bus.publish("credits.consumed", {
        "session_id": session_id,
        "tool": tool_name,
        "cost": cost,
        "session_total": session["total_cost"],
    })

    return cost


def get_session(session_id: str) -> dict[str, Any]:
    """Return current session cost summary."""
    s = _sessions.get(session_id, {"total_cost": 0, "tool_calls": []})
    return {
        "session_cost": s["total_cost"],
        "tool_calls": len(s["tool_calls"]),
        "breakdown": s["tool_calls"][-5:],  # last 5
    }


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
