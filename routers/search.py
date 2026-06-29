"""
POST /search — main search endpoint.

Pipeline:
  MemoryAgent(load) → VisionAgent → IntentAgent → RetrievalAgent → MemoryAgent(save)

Falls back gracefully if ANTHROPIC_API_KEY not set (algorithm mode).
"""
from __future__ import annotations
import os
import uuid

from fastapi import APIRouter
from orynd_core.models.schemas import SearchRequest, SearchResponse, Intent, Candidate
from orynd_core.services import session_store
from orynd_core.agents.base import AgentContext
from orynd_core.agents.intent import IntentAgent
from orynd_core.agents.memory import MemoryAgent
from orynd_core.agents.retrieval import RetrievalAgent
from orynd_core.agents.vision import VisionAgent
from orynd_core.agents.orchestrator import Pipeline

router = APIRouter()


def _make_provider():
    """Return ClaudeProvider if API key available, else None (algorithm fallback)."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if key:
        from orynd_core.services.llm.claude import ClaudeProvider
        return ClaudeProvider(api_key=key)
    return None


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    session_id = req.session_id or str(uuid.uuid4())
    provider = _make_provider()

    # Build shared context
    ctx = AgentContext(
        session_id=session_id,
        user_id=getattr(req, "user_id", None),
        raw_text=req.query,
        image_b64=getattr(req, "image_b64", None),
        image_caption=getattr(req, "caption", None),
        platform=getattr(req, "platform", "desktop"),
    )

    # Full search pipeline
    pipeline = Pipeline([
        MemoryAgent(mode="load"),        # load session history → ctx.extra["history"]
        VisionAgent(provider=provider),  # image → ctx.extra["vision"] + populates raw_text if empty
        IntentAgent(provider=provider),  # text/image → ctx.intent
        RetrievalAgent(limit=5),         # ctx.intent → ctx.candidates (tiered parallel search)
        MemoryAgent(mode="save"),        # persist turn to session history
    ])
    await pipeline.run(ctx)

    # Normalise candidates to Pydantic models
    candidates_out: list[Candidate] = []
    for c in ctx.candidates:
        if isinstance(c, Candidate):
            candidates_out.append(c)
        elif isinstance(c, dict):
            candidates_out.append(Candidate(**c))

    # Cache for /select
    session_store.set_candidates(
        session_id,
        [c.model_dump() for c in candidates_out],
    )

    intent_data = ctx.intent
    keywords_raw = intent_data.get("keywords", [])
    keywords_str = " ".join(keywords_raw) if isinstance(keywords_raw, list) else str(keywords_raw)

    intent = Intent(
        raw=req.query,
        keywords=keywords_str,
        action="search",
    )

    return SearchResponse(
        session_id=session_id,
        candidates=candidates_out,
        intent=intent,
        action="show" if candidates_out else "ideas",
        extra={
            "intent_parsed": intent_data,
            "llm_active": provider is not None,
            "vision": ctx.extra.get("vision"),
        },
    )
