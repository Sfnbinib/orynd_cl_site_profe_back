"""research_writer — converts DeepResearchAgent output → Library entities.

Wire-in point: agents/workspace.py _run_deep_research() after pipeline completes.

Flow:
  research_dict (from ctx.extra["research"])
    → find_or_create Topic (by slug)
    → upsert Article (authored_by="llm", layer=CLOSED)
    → upsert Hypotheses (from gaps[])
    → create ResearchSession record
    → publish event_bus("library.article.published")
    → return {topic_id, article_id, hypotheses_count}

All operations are wrapped in try/except — a library write failure
must NEVER break the main workspace agent loop.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any
from uuid import UUID, uuid4

from orynd_core.services.event_bus import bus
from orynd_core.services.library.schemas import (
    Article,
    Hypothesis,
    Layer,
    ResearchSession,
    Source,
    Topic,
    TopicStage,
)
from orynd_core.services.library.storage_factory import get_storage_backend

log = logging.getLogger("orynd.library.research_writer")

_ANON_UUID = UUID("00000000-0000-0000-0000-000000000001")


def _slugify(text: str) -> str:
    """'Алюминиевый профиль крепление' → 'aluminiievyi-profil-kreplenie'"""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:80]


def _build_body_md(research: dict[str, Any]) -> str:
    """Compose markdown article body from research output."""
    parts: list[str] = []

    recommendations = research.get("recommendations") or ""
    if recommendations:
        parts.append(f"## Рекомендации\n\n{recommendations}\n")

    open_source = research.get("open_source") or []
    if open_source:
        parts.append("## Открытые решения\n")
        for item in open_source[:5]:
            if isinstance(item, dict):
                title = item.get("title") or item.get("name") or str(item)
                url = item.get("url") or item.get("source_url") or ""
                key_info = item.get("key_info") or item.get("description") or ""
                parts.append(f"- **{title}**{f' — {url}' if url else ''}")
                if key_info:
                    parts.append(f"  {key_info[:200]}")
            else:
                parts.append(f"- {str(item)[:200]}")
        parts.append("")

    gaps = research.get("gaps") or []
    if gaps:
        parts.append("## Пробелы / что нужно построить\n")
        for gap in gaps[:10]:
            parts.append(f"- {str(gap)[:200]}")
        parts.append("")

    km = research.get("knowledge_map") or {}
    summary = km.get("summary") or km.get("context_summary") or ""
    if summary:
        parts.append(f"## Контекст исследования\n\n{summary}\n")

    return "\n".join(parts) or recommendations or "No content."


def _build_sources(research: dict[str, Any]) -> list[Source]:
    """Convert raw research source dicts → Library Source objects."""
    raw = research.get("relevant_sources") or research.get("sources") or []
    sources: list[Source] = []
    for item in raw[:10]:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("source_url") or ""
        if not url:
            continue
        sources.append(Source(
            url=url,
            title=str(item.get("title") or item.get("name") or "")[:200],
            reliability_score=float(item.get("relevance", 0.5) or 0.5),
        ))
    return sources


async def save_research(
    topic_text: str,
    research: dict[str, Any],
    session_id: str = "anonymous",
    user_id: str | None = None,
) -> dict[str, Any]:
    """Persist a completed DeepResearch run to the Knowledge Library.

    Returns a summary dict for logging. Never raises — errors are logged
    and an empty dict is returned so the caller's flow is never broken.
    """
    t0 = time.time()
    try:
        storage = get_storage_backend()
        user_uuid = _ANON_UUID

        # ── 1. find_or_create Topic ──────────────────────────────────────────
        slug = _slugify(topic_text)
        topic = await storage.find_topic_by_slug(slug)
        if topic is None:
            topic = Topic(
                slug=slug,
                title=topic_text[:200],
                description=f"Auto-created from deep research: {topic_text[:100]}",
                current_stage=TopicStage.STAGE_0,
                layer=Layer.CLOSED,
                created_by=user_uuid,
                keywords=topic_text.lower().split()[:20],
            )
            topic = await storage.upsert_topic(topic)
            log.info("[research_writer] created topic slug=%s id=%s", slug, topic.id)
        else:
            log.info("[research_writer] found existing topic slug=%s id=%s", slug, topic.id)

        # ── 2. create Article ────────────────────────────────────────────────
        body_md = _build_body_md(research)
        recommendations = research.get("recommendations") or ""
        abstract = (recommendations[:300] + "…") if len(recommendations) > 300 else recommendations
        sources = _build_sources(research)

        article = Article(
            topic_id=topic.id,
            title=topic_text[:200],
            abstract=abstract,
            body_md=body_md,
            authored_by="llm",
            llm_model="deep_research_agent",
            sources=sources,
            quality_score=float(research.get("confidence", 0.5) or 0.5),
            contributes_to_stage=TopicStage.STAGE_0,
            layer=Layer.CLOSED,
            is_ai_marked=True,
        )
        article = await storage.upsert_article(article)
        log.info("[research_writer] saved article id=%s topic=%s", article.id, topic.id)

        # ── 3. create Hypotheses from gaps ───────────────────────────────────
        gaps = research.get("gaps") or []
        hypotheses_saved = 0
        for gap_text in gaps[:5]:
            if not gap_text:
                continue
            h = Hypothesis(
                topic_id=topic.id,
                text=str(gap_text)[:500],
                generated_by="llm",
                status="pending",
                derived_from_articles=[article.id],
                initial_confidence=0.4,
                current_confidence=0.4,
            )
            await storage.upsert_hypothesis(h)
            hypotheses_saved += 1

        # ── 4. create ResearchSession ────────────────────────────────────────
        rs = ResearchSession(
            topic_id=topic.id,
            user_id=user_uuid,
            query=topic_text[:500],
            mode="deep" if research.get("iterations", 1) > 1 else "quick",
            articles_generated=[article.id],
            sources_consulted=[s.id for s in sources],
            contribution_pct_to_topic=30.0,
            user_specific_pct=70.0,
            tokens_consumed=0,
            duration_seconds=int(time.time() - t0),
        )
        await storage.create_session(rs)

        # ── 5. publish event ─────────────────────────────────────────────────
        payload = {
            "topic": topic_text,
            "topic_id": str(topic.id),
            "article_id": str(article.id),
            "hypotheses_count": hypotheses_saved,
            "sources_count": len(sources),
            "session_id": session_id,
            "confidence": research.get("confidence", 0.5),
        }
        await bus.publish("library.article.published", payload)
        log.info("[research_writer] published library.article.published payload=%s", payload)

        return payload

    except Exception as exc:
        log.exception("[research_writer] failed to save research: %s", exc)
        return {}
