"""/library/* — Knowledge Library HTTP API.

Phase 1 endpoints (CRUD + search). Stage transition engine, hypothesis loop,
and OPEN publishing flow attach in Phases 10/11.

Auth is intentionally NOT enforced here yet — the existing routers/auth.py
JWT dep will be applied in Phase 14 along with RLS policy verification per
SECURITY_CHECKLIST.md. For Phase 1 we expose anonymous reads so the UI shell
can render the panel during dev.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from orynd_core.errors import ArticleNotFoundError, TopicNotFoundError
from orynd_core.services.library.schemas import (
    Article,
    Hypothesis,
    Layer,
    ResearchSession,
    Skill,
    Topic,
)
from orynd_core.services.library.storage_abstract import (
    ArticleSearchResult,
    StageMetrics,
)
from orynd_core.services.library.storage_factory import get_storage_backend
from orynd_core.services.observability.metrics import (
    library_articles_total,
    library_searches_total,
)

router = APIRouter(prefix="/library", tags=["library"])


# ---- topics --------------------------------------------------------------


@router.get("/topics", response_model=list[Topic])
async def list_topics(
    layer: Optional[Layer] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    backend = get_storage_backend()
    return await backend.list_topics(layer=layer, limit=limit, offset=offset)


@router.get("/open/topics", response_model=list[Topic])
async def list_open_topics(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Open-layer topics only — meant for unauthenticated community discovery."""
    backend = get_storage_backend()
    return await backend.list_topics(layer=Layer.OPEN, limit=limit, offset=offset)


@router.get("/topics/{topic_id}", response_model=Topic)
async def get_topic(topic_id: UUID):
    backend = get_storage_backend()
    topic = await backend.get_topic(topic_id)
    if not topic:
        raise TopicNotFoundError(details={"topic_id": str(topic_id)})
    return topic


@router.get("/topics/by-slug/{slug}", response_model=Topic)
async def get_topic_by_slug(slug: str):
    backend = get_storage_backend()
    topic = await backend.find_topic_by_slug(slug)
    if not topic:
        raise TopicNotFoundError(details={"slug": slug})
    return topic


@router.post("/topics", response_model=Topic)
async def upsert_topic(topic: Topic):
    backend = get_storage_backend()
    return await backend.upsert_topic(topic)


@router.get("/topics/{topic_id}/metrics", response_model=StageMetrics)
async def topic_metrics(topic_id: UUID):
    backend = get_storage_backend()
    return await backend.topic_metrics(topic_id)


# ---- articles ------------------------------------------------------------
# Note: /articles/search MUST be declared before /articles/{article_id} so
# FastAPI matches the literal path before attempting UUID parsing.


@router.get("/articles/search", response_model=list[ArticleSearchResult])
async def search_articles(
    q: str = Query(..., min_length=1),
    k: int = Query(10, ge=1, le=100),
    topic_id: Optional[UUID] = None,
    mode: str = Query("fts", pattern="^(fts|semantic)$"),
):
    backend = get_storage_backend()
    library_searches_total.labels(type=mode).inc()
    if mode == "semantic":
        return await backend.search_articles_semantic(q, k=k, topic_id=topic_id)
    return await backend.search_articles_fts(q, k=k, topic_id=topic_id)


@router.get("/articles/{article_id}", response_model=Article)
async def get_article(article_id: UUID):
    backend = get_storage_backend()
    article = await backend.get_article(article_id)
    if not article:
        raise ArticleNotFoundError(details={"article_id": str(article_id)})
    return article


@router.get("/topics/{topic_id}/articles", response_model=list[Article])
async def list_articles_by_topic(
    topic_id: UUID,
    layer: Optional[Layer] = None,
    limit: int = Query(50, ge=1, le=200),
):
    backend = get_storage_backend()
    return await backend.list_articles_by_topic(topic_id, layer=layer, limit=limit)


@router.post("/articles", response_model=Article)
async def upsert_article(article: Article):
    backend = get_storage_backend()
    saved = await backend.upsert_article(article)
    library_articles_total.labels(layer=saved.layer.value, authored_by=saved.authored_by).inc()
    return saved


# ---- skills --------------------------------------------------------------


@router.get("/skills/{skill_id}", response_model=Skill)
async def get_skill(skill_id: UUID):
    backend = get_storage_backend()
    skill = await backend.get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail="skill not found")
    return skill


@router.get("/skills/by-slug/{slug}", response_model=Skill)
async def get_skill_by_slug(slug: str):
    backend = get_storage_backend()
    skill = await backend.find_skill_by_slug(slug)
    if not skill:
        raise HTTPException(status_code=404, detail="skill not found")
    return skill


@router.get("/topics/{topic_id}/skills", response_model=list[Skill])
async def list_skills_by_topic(topic_id: UUID):
    backend = get_storage_backend()
    return await backend.list_skills_by_topic(topic_id)


@router.post("/skills", response_model=Skill)
async def upsert_skill(skill: Skill):
    backend = get_storage_backend()
    return await backend.upsert_skill(skill)


# ---- hypotheses ---------------------------------------------------------


@router.get("/topics/{topic_id}/hypotheses", response_model=list[Hypothesis])
async def list_hypotheses(topic_id: UUID):
    backend = get_storage_backend()
    return await backend.list_hypotheses_by_topic(topic_id)


@router.post("/hypotheses", response_model=Hypothesis)
async def upsert_hypothesis(h: Hypothesis):
    backend = get_storage_backend()
    return await backend.upsert_hypothesis(h)


# ---- research sessions --------------------------------------------------


@router.post("/sessions", response_model=ResearchSession)
async def create_session(session: ResearchSession):
    backend = get_storage_backend()
    return await backend.create_session(session)


@router.get("/sessions/{session_id}", response_model=ResearchSession)
async def get_session(session_id: UUID):
    backend = get_storage_backend()
    session = await backend.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    return session


@router.patch("/sessions/{session_id}", response_model=ResearchSession)
async def update_session(session_id: UUID, session: ResearchSession):
    backend = get_storage_backend()
    if session.id != session_id:
        raise HTTPException(status_code=400, detail="path id != body id")
    return await backend.update_session(session)
