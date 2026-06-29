"""
POST /fabricate — run FabricationAgent on a selected candidate.

Returns fabrication pack: method, material, infill, orientation, notes.
"""
from __future__ import annotations
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from orynd_core.services import session_store
from orynd_core.agents.base import AgentContext
from orynd_core.agents.fabrication import FabricationAgent
from orynd_core.agents.orchestrator import Pipeline

router = APIRouter()


class FabricateRequest(BaseModel):
    session_id: str
    index: int = 0


@router.post("/fabricate")
async def fabricate(req: FabricateRequest) -> dict:
    candidates = session_store.get_candidates(req.session_id)
    if not candidates:
        raise HTTPException(404, detail=f"No candidates for session '{req.session_id}'")

    index = req.index if req.index < len(candidates) else 0
    candidate = candidates[index]

    key = os.getenv("ANTHROPIC_API_KEY", "")
    provider = None
    if key:
        from orynd_core.services.llm.claude import ClaudeProvider
        provider = ClaudeProvider(api_key=key)

    ctx = AgentContext(
        session_id=req.session_id,
        selected=candidate,
    )

    await Pipeline([FabricationAgent(provider=provider)]).run(ctx)
    return ctx.extra.get("fabrication", {})
