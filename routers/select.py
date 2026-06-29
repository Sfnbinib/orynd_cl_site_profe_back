"""
POST /select — pick a candidate and verify the STL URL.

Pipeline:
  SelectorAgent (HEAD verify STL, fallback to source_url)

Returns verified download URL or model page URL as fallback.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from orynd_core.models.schemas import SelectRequest, SelectResponse
from orynd_core.services import session_store
from orynd_core.agents.base import AgentContext
from orynd_core.agents.selector import SelectorAgent
from orynd_core.agents.orchestrator import Pipeline

router = APIRouter()


@router.post("/select", response_model=SelectResponse)
async def select(req: SelectRequest) -> SelectResponse:
    candidates = session_store.get_candidates(req.session_id)
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No candidates for session '{req.session_id}'. Run /search first.",
        )

    if req.index >= len(candidates):
        raise HTTPException(
            status_code=422,
            detail=f"Index {req.index} out of range (0–{len(candidates) - 1}).",
        )

    ctx = AgentContext(
        session_id=req.session_id,
        candidates=candidates,
        extra={"select_index": req.index},
    )

    pipeline = Pipeline([SelectorAgent()])
    pipeline_result = await pipeline.run(ctx)

    if not pipeline_result.ok or ctx.selected is None:
        err = pipeline_result.last.error if pipeline_result.last else "SelectorAgent failed"
        raise HTTPException(status_code=500, detail=err)

    last = pipeline_result.last
    verified = last.data.get("verified", False) if last else False

    return SelectResponse(
        url=ctx.stl_url or ctx.selected.get("source_url", ""),
        name=ctx.selected.get("name", ""),
        source=ctx.selected.get("source", ""),
        source_url=ctx.selected.get("source_url", ""),
        verified=verified,
    )
