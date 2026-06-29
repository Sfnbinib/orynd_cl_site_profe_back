"""/learning/* — Learning Engine (Phase 11, algorithm path).

Theory-vs-practice comparison + suggestion generation per blueprint 42:
weights 0.4 semantic / 0.4 structural / 0.2 outcome, suggestions
rate-limited to 3 per 10 minutes per session.

* POST /learning/suggest            — compare an action against theory → hints
* GET  /learning/recent-comparisons — debug: last comparisons
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from orynd_core.auth import UserContext, optional_user

from orynd_core.services.learning.engine import (
    TheoryPattern,
    compare,
    generate_suggestions,
    recent_comparisons,
    record_comparison,
)
from orynd_core.services.learning.theory_seed import SEED_THEORIES

router = APIRouter(prefix="/learning", tags=["learning"])


def _parse_theories(raw: list[dict]) -> list[TheoryPattern]:
    out: list[TheoryPattern] = []
    for i, t in enumerate(raw):
        out.append(TheoryPattern(
            pattern_id=str(t.get("pattern_id", f"custom-{i}")),
            action_type=str(t.get("action_type", "*")),
            text=str(t.get("text", "")),
            recommended_params=dict(t.get("recommended_params", {}) or {}),
            expected_outcome=dict(t.get("expected_outcome", {}) or {}),
        ))
    return out


@router.post("/suggest")
async def suggest(
    payload: dict = Body(...),
    user: UserContext | None = Depends(optional_user),
) -> dict[str, Any]:
    action_type = str(payload.get("action_type", "")).strip()
    if not action_type:
        raise HTTPException(status_code=422, detail="missing 'action_type'")
    action_params = dict(payload.get("action_params", {}) or {})
    practice_text = str(payload.get("text", "") or f"{action_type} {action_params}")
    result = payload.get("result")
    session_id = str(payload.get("session_id") or (str(user.id) if user else "default"))

    theories = _parse_theories(payload.get("theories", []) or []) or SEED_THEORIES

    comparison = compare(
        action_type=action_type,
        action_params=action_params,
        practice_text=practice_text,
        theories=theories,
        result=result if isinstance(result, dict) else None,
    )
    suggestions = generate_suggestions(comparison, session_id=session_id)

    record_comparison({
        "action_type": action_type,
        "match_score": comparison.match_score,
        "gap_areas": comparison.gap_areas,
        "reasoning": comparison.reasoning,
        "computed_at": time.time(),
    })

    return {
        "comparison": comparison.to_dict(),
        "suggestions": suggestions,
        "session_id": session_id,
    }


@router.get("/recent-comparisons")
async def get_recent(limit: int = Query(20, ge=1, le=50)) -> dict[str, Any]:
    return {"comparisons": recent_comparisons(limit)}


@router.get("/action-log")
async def get_action_log(
    limit: int = Query(50, ge=1, le=200),
    session_id: str | None = Query(None),
) -> dict[str, Any]:
    """Debug: recent action_log entries (Practice Store feed)."""
    from orynd_core.services.action_log import recent
    return {"entries": recent(limit=limit, session_id=session_id), "count": limit}
