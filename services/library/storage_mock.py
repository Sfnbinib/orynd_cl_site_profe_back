"""In-memory StorageBackend.

Used:
* by tests (deterministic, no external services)
* as fallback when ``LIBRARY_BACKEND=supabase`` but Supabase keys are missing
  (the factory logs a warning and returns this instead of crashing the app)

Search semantics here are intentionally simple — substring match against
title/body. Real semantic search arrives with embeddings + Supabase RPC.
"""

from __future__ import annotations

import asyncio
from typing import Optional
from uuid import UUID

from orynd_core.services.library.schemas import (
    Article,
    Hypothesis,
    Layer,
    ResearchSession,
    Skill,
    Topic,
    TopicStage,
)
from orynd_core.services.library.storage_abstract import (
    ArticleSearchResult,
    StageMetrics,
    StorageBackend,
)


class MockStorageBackend(StorageBackend):
    def __init__(self) -> None:
        self._topics: dict[UUID, Topic] = {}
        self._topics_by_slug: dict[str, UUID] = {}
        self._articles: dict[UUID, Article] = {}
        self._skills: dict[UUID, Skill] = {}
        self._skills_by_slug: dict[str, UUID] = {}
        self._sessions: dict[UUID, ResearchSession] = {}
        self._hypotheses: dict[UUID, Hypothesis] = {}
        self._lock = asyncio.Lock()

    # ---- topics ---------------------------------------------------------

    async def get_topic(self, topic_id: UUID) -> Optional[Topic]:
        return self._topics.get(topic_id)

    async def find_topic_by_slug(self, slug: str) -> Optional[Topic]:
        tid = self._topics_by_slug.get(slug)
        return self._topics.get(tid) if tid else None

    async def search_topics(self, query: str, k: int = 10) -> list[Topic]:
        q = query.lower()
        matches = [
            t
            for t in self._topics.values()
            if q in t.title.lower() or q in t.slug or q in t.description.lower()
        ]
        return matches[:k]

    async def list_topics(
        self,
        *,
        layer: Optional[Layer] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Topic]:
        items = list(self._topics.values())
        if layer is not None:
            items = [t for t in items if t.layer == layer]
        items.sort(key=lambda t: t.updated_at, reverse=True)
        return items[offset : offset + limit]

    async def upsert_topic(self, topic: Topic) -> Topic:
        async with self._lock:
            self._topics[topic.id] = topic
            self._topics_by_slug[topic.slug] = topic.id
        return topic

    # ---- articles -------------------------------------------------------

    async def get_article(self, article_id: UUID) -> Optional[Article]:
        return self._articles.get(article_id)

    async def list_articles_by_topic(
        self,
        topic_id: UUID,
        layer: Optional[Layer] = None,
        limit: int = 50,
    ) -> list[Article]:
        articles = [a for a in self._articles.values() if a.topic_id == topic_id]
        if layer is not None:
            articles = [a for a in articles if a.layer == layer]
        articles.sort(key=lambda a: a.updated_at, reverse=True)
        return articles[:limit]

    async def search_articles_semantic(
        self,
        query: str,
        k: int = 10,
        topic_id: Optional[UUID] = None,
    ) -> list[ArticleSearchResult]:
        # Mock impl falls back to substring scoring — real semantic comes
        # online when SupabaseBackend is wired with embeddings.
        return await self.search_articles_fts(query, k=k, topic_id=topic_id)

    async def search_articles_fts(
        self,
        query: str,
        k: int = 10,
        topic_id: Optional[UUID] = None,
    ) -> list[ArticleSearchResult]:
        q = query.lower()
        results: list[ArticleSearchResult] = []
        for a in self._articles.values():
            if topic_id is not None and a.topic_id != topic_id:
                continue
            score = _substring_score(a, q)
            if score > 0:
                results.append(ArticleSearchResult(article=a, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    async def upsert_article(self, article: Article) -> Article:
        async with self._lock:
            self._articles[article.id] = article
            topic = self._topics.get(article.topic_id)
            if topic:
                topic.article_count = sum(
                    1 for a in self._articles.values() if a.topic_id == topic.id
                )
        return article

    # ---- skills ---------------------------------------------------------

    async def get_skill(self, skill_id: UUID) -> Optional[Skill]:
        return self._skills.get(skill_id)

    async def find_skill_by_slug(self, slug: str) -> Optional[Skill]:
        sid = self._skills_by_slug.get(slug)
        return self._skills.get(sid) if sid else None

    async def list_skills_by_topic(self, topic_id: UUID) -> list[Skill]:
        return [s for s in self._skills.values() if s.topic_id == topic_id]

    async def upsert_skill(self, skill: Skill) -> Skill:
        async with self._lock:
            self._skills[skill.id] = skill
            self._skills_by_slug[skill.slug] = skill.id
            topic = self._topics.get(skill.topic_id)
            if topic:
                topic.skill_count = sum(
                    1 for s in self._skills.values() if s.topic_id == topic.id
                )
        return skill

    # ---- sessions -------------------------------------------------------

    async def create_session(self, session: ResearchSession) -> ResearchSession:
        async with self._lock:
            self._sessions[session.id] = session
        return session

    async def update_session(self, session: ResearchSession) -> ResearchSession:
        return await self.create_session(session)

    async def get_session(self, session_id: UUID) -> Optional[ResearchSession]:
        return self._sessions.get(session_id)

    # ---- hypotheses -----------------------------------------------------

    async def upsert_hypothesis(self, h: Hypothesis) -> Hypothesis:
        async with self._lock:
            self._hypotheses[h.id] = h
        return h

    async def list_hypotheses_by_topic(self, topic_id: UUID) -> list[Hypothesis]:
        return [h for h in self._hypotheses.values() if h.topic_id == topic_id]

    # ---- aggregations ---------------------------------------------------

    async def topic_metrics(self, topic_id: UUID) -> StageMetrics:
        topic = self._topics.get(topic_id)
        if not topic:
            return StageMetrics(
                topic_id=topic_id,
                current_stage=TopicStage.STAGE_0,
                article_count=0,
                skill_count=0,
                contributor_count=0,
                total_iterations=0,
            )
        articles = [a for a in self._articles.values() if a.topic_id == topic_id]
        skills = [s for s in self._skills.values() if s.topic_id == topic_id]
        contributors = {
            a.user_id for a in articles if a.user_id is not None
        } | {s.created_by for s in skills}
        avg_quality = (
            sum(a.quality_score for a in articles) / len(articles)
            if articles
            else 0.0
        )
        # Phase 1 eligibility: enough articles + good quality is sufficient.
        # The full distributed-research check (≥3 distinct contributors etc.)
        # lands in Phase 11's StageTransitionEngine.
        eligible = len(articles) >= 3 and avg_quality >= 0.6
        rationale: list[str] = []
        if eligible:
            rationale.append(
                f"{len(articles)} articles, avg quality {avg_quality:.2f}, "
                f"{len(contributors)} contributors"
            )
        return StageMetrics(
            topic_id=topic_id,
            current_stage=topic.current_stage,
            article_count=len(articles),
            skill_count=len(skills),
            contributor_count=len(contributors),
            total_iterations=topic.total_iterations,
            avg_quality_score=avg_quality,
            coverage_ratio=topic.progress_ratio,
            promotion_eligible=eligible,
            rationale=rationale,
        )


def _substring_score(article: Article, q: str) -> float:
    title_hit = q in article.title.lower()
    body_hit = q in article.body_md.lower()
    abstract_hit = q in article.abstract.lower()
    if not (title_hit or body_hit or abstract_hit):
        return 0.0
    score = 0.0
    if title_hit:
        score += 0.6
    if abstract_hit:
        score += 0.3
    if body_hit:
        score += 0.1
    return min(score, 1.0)


__all__ = ["MockStorageBackend"]
