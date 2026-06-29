"""/attachment/* — axes attachment (hybrid manual UX).

Founder ask (blueprint 99 + HYBRID_WORKFLOW): click axis → right-click →
⌘1..⌘9 hints → pick → adaptive fit → CoreOps. Tab repeats last action.

Endpoints:
  GET  /attachment/catalog              — list seed parts (optional category)
  POST /attachment/suggest              — axis → ranked candidates (⌘1..⌘9 hints)
  POST /attachment/apply                — chosen part + axis → CoreOps attach op
  GET  /attachment/last/{session_id}    — last applied action (for Tab repeat)
  POST /attachment/repeat/{session_id}  — repeat last action on a new axis (Tab)
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Query

from orynd_core.services.attachment.catalog import all_parts, get_part, parts_by_category
from orynd_core.services.attachment.matcher import Axis, match_axis

router = APIRouter(prefix="/attachment", tags=["attachment"])

# Per-session "last action" memory for Tab repeat (in-process, demo).
_last_action: dict[str, dict] = {}
_lock = threading.Lock()


def _axis_from_payload(p: dict) -> Axis:
    try:
        origin = tuple(float(v) for v in p["origin"])
        direction = tuple(float(v) for v in p["direction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"bad axis: {exc}")
    if len(origin) != 3 or len(direction) != 3:
        raise HTTPException(status_code=422, detail="origin/direction must be 3-tuples")
    diameter = p.get("diameter")
    length = p.get("length")
    return Axis(
        origin=origin,
        direction=direction,
        diameter=float(diameter) if diameter is not None else None,
        length=float(length) if length is not None else None,
    )


@router.get("/catalog")
async def list_catalog(category: Optional[str] = None) -> list[dict]:
    parts = parts_by_category(category) if category else all_parts()
    return [p.to_dict() for p in parts]


@router.post("/suggest")
async def suggest(payload: dict = Body(...)) -> dict[str, Any]:
    """Axis → ranked candidates. Top-9 → ⌘1..⌘9 hint slots in UI."""
    axis = _axis_from_payload(payload.get("axis", {}))
    intent = payload.get("intent")
    category = payload.get("category")
    candidates = match_axis(axis, intent=intent, category=category, k=9)
    return {
        "axis": {
            "origin": list(axis.origin),
            "direction": list(axis.normalised_direction()),
            "diameter": axis.diameter,
            "length": axis.length,
        },
        "intent": intent,
        "candidates": [c.to_dict() for c in candidates],
        # UI hint mapping: hotkey ⌘1..⌘9
        "hints": [
            {"hotkey": f"cmd+{i + 1}", "part_id": c.part.part_id, "name": c.part.name}
            for i, c in enumerate(candidates)
        ],
    }


@router.post("/apply")
async def apply(payload: dict = Body(...)) -> dict[str, Any]:
    """Apply chosen part to axis → CoreOps attach operation.

    Body: {session_id, axis, part_id, adaptation?: {scale, rotation_deg}}
    """
    session_id = str(payload.get("session_id", "attach"))
    axis = _axis_from_payload(payload.get("axis", {}))
    part_id = payload.get("part_id")
    if not part_id:
        raise HTTPException(status_code=422, detail="missing part_id")
    part = get_part(str(part_id))
    if part is None:
        raise HTTPException(status_code=404, detail=f"unknown part: {part_id}")

    adaptation = payload.get("adaptation", {}) or {}
    scale = float(adaptation.get("scale", 1.0))
    rotation_deg = float(adaptation.get("rotation_deg", 0.0))

    # Adaptive-fit parameters (scale primitive dims)
    params = {
        k: (v * scale if isinstance(v, (int, float)) else v)
        for k, v in part.default_parameters.items()
    }

    coreops_op = {
        "op_id": f"attach_{part.part_id}",
        "type": "attach",
        "primitive_type": part.primitive_type,
        "parameters": params,
        "axis": {
            "origin": list(axis.origin),
            "direction": list(axis.normalised_direction()),
        },
        "rotation_deg": rotation_deg,
        "part_id": part.part_id,
        "source": "manual_attach",
    }

    action = {
        "part_id": part.part_id,
        "category": part.category,
        "adaptation": {"scale": scale, "rotation_deg": rotation_deg},
        "coreops_op": coreops_op,
    }
    with _lock:
        _last_action[session_id] = action

    return {
        "session_id": session_id,
        "applied": action,
        "tab_repeat_available": True,
    }


@router.get("/last/{session_id}")
async def get_last(session_id: str) -> dict[str, Any]:
    action = _last_action.get(session_id)
    if action is None:
        return {"session_id": session_id, "last_action": None}
    return {"session_id": session_id, "last_action": action}


@router.post("/repeat/{session_id}")
async def repeat_last(session_id: str, payload: dict = Body(...)) -> dict[str, Any]:
    """Tab repeat — apply the last action's part to a NEW axis.

    Founder UX: выбрал 2 объекта → ⌘1 (соединить по оси) → на след раз
    жмёшь Tab → повторяется последнее действие на новой оси.
    """
    last = _last_action.get(session_id)
    if last is None:
        raise HTTPException(status_code=404, detail="no last action to repeat")
    axis = _axis_from_payload(payload.get("axis", {}))
    part = get_part(last["part_id"])
    if part is None:
        raise HTTPException(status_code=410, detail="last part no longer in catalog")

    scale = float(last["adaptation"]["scale"])
    rotation_deg = float(last["adaptation"]["rotation_deg"])
    params = {
        k: (v * scale if isinstance(v, (int, float)) else v)
        for k, v in part.default_parameters.items()
    }
    coreops_op = {
        "op_id": f"attach_{part.part_id}_repeat",
        "type": "attach",
        "primitive_type": part.primitive_type,
        "parameters": params,
        "axis": {
            "origin": list(axis.origin),
            "direction": list(axis.normalised_direction()),
        },
        "rotation_deg": rotation_deg,
        "part_id": part.part_id,
        "source": "tab_repeat",
    }
    return {
        "session_id": session_id,
        "repeated": last["part_id"],
        "coreops_op": coreops_op,
    }
