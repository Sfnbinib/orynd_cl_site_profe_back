"""/research — Deep Research (Phase 10, real).

Runs DeepResearchAgent (5-phase parallel pipeline: collect → filter →
synthesize → orchestrate → recommend). Model selection goes through
model_router (local-first: Ollama → Claude fallback; algorithm path when
neither is available).

The finished synthesis is pushed to the Knowledge Library as an article
(topic auto-created from the research topic slug).

* POST /research        — full research {topic, depth?: 1..3}
* POST /research/light  — depth=1 shortcut (slash command target)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional
from uuid import NAMESPACE_DNS, uuid4, uuid5

# Deterministic system author for auto-generated research artifacts
SYSTEM_AUTHOR_ID = uuid5(NAMESPACE_DNS, "research.orynd.system")

from fastapi import APIRouter, Body, Depends, HTTPException

from orynd_core.auth import UserContext, optional_user

from orynd_core.agents.base import AgentContext
from orynd_core.agents.research import DeepResearchAgent

log = logging.getLogger(__name__)
router = APIRouter(prefix="/research", tags=["research"])


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9а-яё]+", "-", text.lower()).strip("-")
    return slug[:64] or "research"


async def _push_article_to_library(topic_text: str, research: dict) -> Optional[str]:
    """Write the synthesis into the Library; returns article id or None."""
    if not _research_has_sources(research):
        return None
    try:
        from orynd_core.services.library.schemas import Article, Topic
        from orynd_core.services.library.storage_factory import get_storage_backend

        backend = get_storage_backend()
        slug = _slugify(topic_text)
        topic = await backend.find_topic_by_slug(slug)
        if topic is None:
            topic = await backend.upsert_topic(
                Topic(
                    slug=slug,
                    title=topic_text,
                    description="Auto-created by Deep Research",
                    created_by=SYSTEM_AUTHOR_ID,
                )
            )

        knowledge_map = research.get("knowledge_map", {})
        body_lines = [
            f"# {topic_text}",
            "",
            research.get("recommendations", "") or "",
            "",
            "## Open-source solutions",
            *[f"- {s}" for s in research.get("open_source", [])],
            "",
            "## Knowledge gaps",
            *[f"- {g}" for g in research.get("gaps", [])],
            "",
            "## Build from scratch",
            *[f"- {b}" for b in research.get("build_from_scratch", [])],
        ]
        article = Article(
            topic_id=topic.id,
            title=f"Research: {topic_text}",
            abstract=str(knowledge_map.get("summary", ""))[:500],
            body_md="\n".join(body_lines),
            authored_by="llm",
            llm_model="model_router",
            quality_score=float(research.get("confidence", 0.0) or 0.0),
        )
        saved = await backend.upsert_article(article)
        return str(saved.id)
    except Exception:
        # Library push is best-effort — research result is still returned.
        log.exception("[research] failed to push article to library")
        return None


def _research_has_sources(research: dict) -> bool:
    sources_total = int(research.get("sources_total") or 0)
    sources = research.get("sources") or []
    return sources_total > 0 or len(sources) > 0


@router.post("")
async def run_research(
    payload: dict = Body(...),
    user: UserContext | None = Depends(optional_user),
) -> dict[str, Any]:
    topic = (payload.get("topic") or payload.get("text") or "").strip()
    if not topic:
        raise HTTPException(status_code=422, detail="missing 'topic'")
    depth = int(payload.get("depth", 2) or 2)
    session_id = str(payload.get("session_id") or (str(user.id) if user else f"research-{uuid4().hex[:8]}"))

    ctx = AgentContext(session_id=session_id, raw_text=topic)
    agent = DeepResearchAgent(depth=depth)
    result = await agent.run(ctx)
    if not result.ok:
        raise HTTPException(status_code=502, detail=result.error or "research failed")

    research = ctx.extra.get("research", {})
    has_sources = _research_has_sources(research)
    article_id = await _push_article_to_library(topic, research) if has_sources else None

    return {
        "topic": topic,
        "depth": depth,
        "status": "ready" if has_sources else "empty",
        "research": research,
        "candidates": ctx.candidates,
        "article_id": article_id,
        "session_id": session_id,
    }


@router.post("/light")
async def run_research_light(
    payload: dict = Body(...),
    user: UserContext | None = Depends(optional_user),
) -> dict[str, Any]:
    payload = {**payload, "depth": 1}
    return await run_research(payload, user=user)
