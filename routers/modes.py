"""/modes/* — read/write the operation mode."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body, HTTPException

from orynd_core.services.modes import (
    OperationMode,
    clear_session_mode,
    get_mode,
    set_mode,
)

router = APIRouter(prefix="/modes", tags=["modes"])


@router.get("")
async def get_current_mode(session_id: Optional[str] = None) -> dict:
    return {"mode": get_mode(session_id).value, "session_id": session_id}


@router.post("")
async def set_current_mode(payload: dict = Body(...)) -> dict:
    raw = payload.get("mode")
    session_id = payload.get("session_id")
    if not raw:
        raise HTTPException(status_code=422, detail="missing 'mode'")
    try:
        mode = OperationMode(raw)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"unknown mode {raw!r}; expected one of {[m.value for m in OperationMode]}",
        )
    await set_mode(mode, session_id=session_id)
    return {"mode": mode.value, "session_id": session_id}


@router.delete("/session/{session_id}")
async def clear_session(session_id: str) -> dict:
    clear_session_mode(session_id)
    return {"cleared": session_id}


@router.get("/options")
async def list_modes() -> list[dict]:
    return [
        {"value": OperationMode.PLAN.value, "label": "Plan", "description": "Preview every action; never executes."},
        {"value": OperationMode.AUTO.value, "label": "Auto", "description": "Default — only ask for high-permission actions."},
        {"value": OperationMode.ASK_PERMISSION.value, "label": "Ask Permission", "description": "Confirm every action."},
        {"value": OperationMode.BYPASS.value, "label": "Bypass", "description": "Autonomous; never asks."},
    ]
