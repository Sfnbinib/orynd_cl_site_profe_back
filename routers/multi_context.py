"""/context/* — multi-context UI glue (slash commands + scene chips).

Per MULTI_CONTEXT_UI.md. Bottom chat ↔ scene canvas ↔ left agent panel.

* POST /context/slash        — parse a chat input, return routing target
* GET  /context/slash/help   — list available slash commands
* POST /context/chip         — selected 3D object → context chip for chat
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from orynd_core.services.event_bus import bus
from orynd_core.services.multi_context.slash import help_text, parse_slash

router = APIRouter(prefix="/context", tags=["multi-context"])


@router.post("/slash")
async def slash(payload: dict = Body(...)) -> dict[str, Any]:
    """Parse chat input. If slash command → routing target for the handler."""
    text = payload.get("text", "")
    if not isinstance(text, str):
        raise HTTPException(status_code=422, detail="'text' must be string")
    result = parse_slash(text)
    return result.to_dict()


@router.get("/slash/help")
async def slash_help() -> list[dict]:
    return help_text()


@router.post("/chip")
async def context_chip(payload: dict = Body(...)) -> dict[str, Any]:
    """User selected a 3D object → emit a context chip into the bottom chat.

    Body: {session_id, workspace_id?, object_id, object_type?, label?, metadata?}
    The chip is published on the event bus so the chat panel can render it,
    and returned so the UI can optimistically show it.
    """
    object_id = payload.get("object_id")
    if not object_id:
        raise HTTPException(status_code=422, detail="missing object_id")
    session_id = str(payload.get("session_id", "default"))
    workspace_id = str(payload.get("workspace_id") or session_id)
    chip = {
        "chip_id": f"chip_{object_id}_{int(time.time() * 1000)}",
        "session_id": session_id,
        "workspace_id": workspace_id,
        "object_id": str(object_id),
        "object_type": payload.get("object_type", "unknown"),
        "label": payload.get("label") or str(object_id),
        "metadata": dict(payload.get("metadata", {}) or {}),
        "created_at": time.time(),
    }
    await bus.publish("context.chip", chip)

    # Wire #5: persist chip into shared workspace state
    try:
        from orynd_core.services import workspace_state
        await workspace_state.update(workspace_id, {"context_chips": [chip]})
    except Exception:
        pass

    return {"chip": chip}


@router.get("/workspace/{workspace_id}")
async def get_workspace_state(workspace_id: str) -> dict[str, Any]:
    """Return current shared workspace state for this workspace_id.

    Used by Visual Orchestrator and any surface that needs to sync on connect.
    """
    from orynd_core.services import workspace_state
    return workspace_state.get(workspace_id)
