"""
MemoryAgent — session context store. No LLM required.

Loads conversation history into ctx at pipeline start,
saves updated history after pipeline completes.
Runs first in every pipeline (load) and last (save) optionally.
"""

from __future__ import annotations
import time
from collections import defaultdict
from dataclasses import dataclass, field

from orynd_core.agents.base import AgentContext, AgentResult, BaseAgent

_MAX_HISTORY = 20

# In-memory store: session_id → session data
# Phase 5: replace with Supabase persistent store
_store: dict[str, dict] = defaultdict(lambda: {"history": [], "profile": {}, "last_ts": 0})


@dataclass
class SessionData:
    history: list[dict] = field(default_factory=list)   # last N turns
    profile: dict = field(default_factory=dict)          # user preferences
    last_ts: float = 0.0


def load_session(session_id: str) -> SessionData:
    raw = _store[session_id]
    return SessionData(
        history=raw["history"],
        profile=raw["profile"],
        last_ts=raw["last_ts"],
    )


def save_session(session_id: str, data: SessionData) -> None:
    _store[session_id] = {
        "history": data.history[-_MAX_HISTORY:],
        "profile": data.profile,
        "last_ts": data.last_ts,
    }


class MemoryAgent(BaseAgent):
    """
    Load mode (default): pulls session history into ctx.extra["history"].
    Save mode: appends current turn to session history.
    """

    name = "memory_agent"

    def __init__(self, mode: str = "load") -> None:
        super().__init__(provider=None)
        assert mode in ("load", "save"), "mode must be 'load' or 'save'"
        self.mode = mode

    async def run_logic(self, ctx: AgentContext) -> AgentResult:
        session = load_session(ctx.session_id)

        if self.mode == "load":
            ctx.extra["history"] = session.history
            ctx.extra["user_profile"] = session.profile
            return AgentResult.success(self.name, {"loaded_turns": len(session.history)})

        # save mode — append current turn with full context
        turn: dict = {"ts": time.time()}
        if ctx.raw_text:
            turn["query"] = ctx.raw_text
        if ctx.intent:
            turn["intent"] = ctx.intent
        if ctx.candidates:
            turn["candidate_count"] = len(ctx.candidates)
            # Save top 3 candidate names for context
            turn["candidate_names"] = [
                (c.get("name") if isinstance(c, dict) else c.name)
                for c in ctx.candidates[:3]
            ]
        if ctx.selected:
            turn["selected_id"] = ctx.selected.get("id")
            turn["selected_name"] = ctx.selected.get("name")
        # Save workspace response (truncated)
        workspace_resp = ctx.extra.get("workspace_response", "")
        if workspace_resp:
            turn["workspace_response"] = workspace_resp[:500]
        # Save fabrication method if used
        fab = ctx.extra.get("fabrication", {})
        if fab:
            turn["fabrication_method"] = fab.get("recommended_method")
            turn["fabrication_material"] = fab.get("material")
        # Save CAD result if model was built
        cad = ctx.extra.get("cad", {})
        if cad:
            turn["cad_built"] = True
            turn["cad_dry_run"] = cad.get("dry_run", False)
            turn["cad_ops"] = cad.get("operations_executed", 0)
        # Save tool calls for context
        tools_used = ctx.extra.get("tool_calls", [])
        if tools_used:
            turn["tools_used"] = tools_used

        session.history.append(turn)
        session.last_ts = turn["ts"]
        save_session(ctx.session_id, session)

        return AgentResult.success(self.name, {"saved_turn": turn})
