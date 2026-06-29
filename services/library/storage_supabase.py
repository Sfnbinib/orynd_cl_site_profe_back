"""SupabaseBackend — wraps the Python supabase client.

The ``supabase`` package is intentionally an **optional** runtime dep. If it
is missing or the env vars aren't set, the factory falls back to
``MockStorageBackend`` and the app keeps working (with a logged warning).

We use the sync client (``create_client``) and wrap calls in
``asyncio.to_thread`` because the official Python SDK still does sync I/O.
That trade-off is acceptable for Phase 1 — once supabase-py-async stabilises
or we migrate to PostgresBackend, this file can be swapped without touching
callers.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional
from uuid import UUID

from orynd_core.errors import ExternalAPIError
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
from orynd_core.services.logging import get_logger

log = get_logger("orynd.library.supabase")

try:  # pragma: no cover — runtime optional
    from supabase import Client, create_client  # type: ignore
except Exception:  # pragma: no cover
    Client = None  # type: ignore
    create_client = None  # type: ignore

# Columns present in each DB table. model_dump() produces more fields than the
# table schema; we filter before upsert to avoid "column not found" errors.
_TOPIC_COLS = {"id", "slug", "title", "description", "current_stage", "layer", "keywords", "created_by", "created_at", "updated_at"}
_ARTICLE_COLS = {"id", "topic_id", "title", "abstract", "body_md", "authored_by", "llm_model", "quality_score", "layer", "is_ai_marked", "contributes_to_stage", "sources", "created_at", "updated_at"}
_SKILL_COLS = {"id", "topic_id", "slug", "layer", "created_at"}  # title+body_md mapped from name+description below
_SESSION_COLS = {"id", "topic_id", "user_id", "query", "mode", "articles_generated", "sources_consulted", "contribution_pct_to_topic", "user_specific_pct", "tokens_consumed", "duration_seconds", "created_at"}
_HYPOTHESIS_COLS = {"id", "topic_id", "text", "generated_by", "status", "initial_confidence", "current_confidence", "derived_from_articles", "created_at"}


class SupabaseBackend(StorageBackend):
    def __init__(self, url: str, key: str) -> None:
        if create_client is None:
            raise ExternalAPIError(
                "supabase package not installed",
                details={"hint": "pip install supabase"},
            )
        self.url = url
        self.key = key
        # Eagerly init — create_client is synchronous so safe to call here.
        self._client: Optional[Client] = create_client(url, key)

    @classmethod
    def from_env(cls) -> "SupabaseBackend":
        # Library-specific vars win; fall back to the shared project keys so a
        # single free-tier Supabase project covers auth + library (founder A4).
        url = os.environ.get("SUPABASE_LIBRARY_URL") or os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_LIBRARY_KEY") or os.environ.get("SUPABASE_ANON_KEY")
        if not (url and key) or "your-project" in (url or ""):
            raise ExternalAPIError(
                "Supabase library env vars missing",
                details={"required": ["SUPABASE_LIBRARY_URL|SUPABASE_URL", "SUPABASE_LIBRARY_KEY|SUPABASE_ANON_KEY"]},
            )
        return cls(url=url, key=key)

    async def connect(self) -> None:
        assert create_client is not None
        self._client = await asyncio.to_thread(create_client, self.url, self.key)

    async def disconnect(self) -> None:
        self._client = None

    # ---- topics ---------------------------------------------------------

    async def get_topic(self, topic_id: UUID) -> Optional[Topic]:
        row = await self._fetch_single("topics", {"id": str(topic_id)})
        return Topic(**row) if row else None

    async def find_topic_by_slug(self, slug: str) -> Optional[Topic]:
        row = await self._fetch_single("topics", {"slug": slug})
        return Topic(**row) if row else None

    async def search_topics(self, query: str, k: int = 10) -> list[Topic]:
        client = self._require_client()
        # Simple ilike across title/slug — semantic later via RPC.
        def _run() -> list[dict]:
            r = (
                client.table("topics")
                .select("*")
                .ilike("title", f"%{query}%")
                .limit(k)
                .execute()
            )
            return r.data or []

        rows = await asyncio.to_thread(_run)
        return [Topic(**row) for row in rows]

    async def list_topics(
        self,
        *,
        layer: Optional[Layer] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Topic]:
        client = self._require_client()

        def _run() -> list[dict]:
            q = client.table("topics").select("*")
            if layer is not None:
                q = q.eq("layer", layer.value)
            r = q.order("updated_at", desc=True).range(offset, offset + limit - 1).execute()
            return r.data or []

        return [Topic(**row) for row in await asyncio.to_thread(_run)]

    async def upsert_topic(self, topic: Topic) -> Topic:
        row = {k: v for k, v in topic.model_dump(mode="json").items() if k in _TOPIC_COLS}
        await self._upsert("topics", row)
        return topic

    # ---- articles -------------------------------------------------------

    async def get_article(self, article_id: UUID) -> Optional[Article]:
        row = await self._fetch_single("articles", {"id": str(article_id)})
        return Article(**row) if row else None

    async def list_articles_by_topic(
        self,
        topic_id: UUID,
        layer: Optional[Layer] = None,
        limit: int = 50,
    ) -> list[Article]:
        client = self._require_client()

        def _run() -> list[dict]:
            q = client.table("articles").select("*").eq("topic_id", str(topic_id))
            if layer is not None:
                q = q.eq("layer", layer.value)
            r = q.order("updated_at", desc=True).limit(limit).execute()
            return r.data or []

        return [Article(**row) for row in await asyncio.to_thread(_run)]

    async def search_articles_semantic(
        self,
        query: str,
        k: int = 10,
        topic_id: Optional[UUID] = None,
    ) -> list[ArticleSearchResult]:
        # Requires Postgres RPC `match_articles` with vector index — see
        # migrations/library_cluster_v1.sql. Until embedding service is
        # wired we degrade to FTS for parity with the mock backend.
        return await self.search_articles_fts(query, k=k, topic_id=topic_id)

    async def search_articles_fts(
        self,
        query: str,
        k: int = 10,
        topic_id: Optional[UUID] = None,
    ) -> list[ArticleSearchResult]:
        client = self._require_client()

        def _run() -> list[dict]:
            q = client.table("articles").select("*").ilike("body_md", f"%{query}%")
            if topic_id is not None:
                q = q.eq("topic_id", str(topic_id))
            r = q.limit(k).execute()
            return r.data or []

        rows = await asyncio.to_thread(_run)
        return [ArticleSearchResult(article=Article(**row), score=0.5) for row in rows]

    async def upsert_article(self, article: Article) -> Article:
        row = {k: v for k, v in article.model_dump(mode="json").items() if k in _ARTICLE_COLS}
        # sources is a list of Source objects — serialize to plain dicts
        row["sources"] = [s if isinstance(s, dict) else s for s in (row.get("sources") or [])]
        await self._upsert("articles", row)
        return article

    # ---- skills ---------------------------------------------------------

    async def get_skill(self, skill_id: UUID) -> Optional[Skill]:
        row = await self._fetch_single("skills", {"id": str(skill_id)})
        return Skill(**row) if row else None

    async def find_skill_by_slug(self, slug: str) -> Optional[Skill]:
        row = await self._fetch_single("skills", {"slug": slug})
        return Skill(**row) if row else None

    async def list_skills_by_topic(self, topic_id: UUID) -> list[Skill]:
        client = self._require_client()

        def _run() -> list[dict]:
            r = (
                client.table("skills")
                .select("*")
                .eq("topic_id", str(topic_id))
                .execute()
            )
            return r.data or []

        return [Skill(**row) for row in await asyncio.to_thread(_run)]

    async def upsert_skill(self, skill: Skill) -> Skill:
        d = skill.model_dump(mode="json")
        row = {k: v for k, v in d.items() if k in _SKILL_COLS}
        row["title"] = d.get("name", "")        # pydantic: name → DB: title
        row["body_md"] = d.get("description", "")  # pydantic: description → DB: body_md
        await self._upsert("skills", row)
        return skill

    # ---- sessions / hypotheses -----------------------------------------

    async def create_session(self, session: ResearchSession) -> ResearchSession:
        d = session.model_dump(mode="json")
        row = {k: v for k, v in d.items() if k in _SESSION_COLS}
        # pydantic has started_at; DB uses created_at
        if "created_at" not in row and "started_at" in d:
            row["created_at"] = d["started_at"]
        # user_id may be None — Supabase accepts null for UUID column
        row.setdefault("user_id", None)
        await self._upsert("research_sessions", row)
        return session

    async def update_session(self, session: ResearchSession) -> ResearchSession:
        return await self.create_session(session)

    async def get_session(self, session_id: UUID) -> Optional[ResearchSession]:
        row = await self._fetch_single("research_sessions", {"id": str(session_id)})
        return ResearchSession(**row) if row else None

    async def upsert_hypothesis(self, h: Hypothesis) -> Hypothesis:
        row = {k: v for k, v in h.model_dump(mode="json").items() if k in _HYPOTHESIS_COLS}
        await self._upsert("hypotheses", row)
        return h

    async def list_hypotheses_by_topic(self, topic_id: UUID) -> list[Hypothesis]:
        client = self._require_client()

        def _run() -> list[dict]:
            r = (
                client.table("hypotheses")
                .select("*")
                .eq("topic_id", str(topic_id))
                .execute()
            )
            return r.data or []

        return [Hypothesis(**row) for row in await asyncio.to_thread(_run)]

    async def topic_metrics(self, topic_id: UUID) -> StageMetrics:
        # Aggregations done client-side for simplicity — Phase 11 can move
        # this to a PG view.
        topic = await self.get_topic(topic_id)
        if not topic:
            return StageMetrics(
                topic_id=topic_id,
                current_stage=TopicStage.STAGE_0,
                article_count=0,
                skill_count=0,
                contributor_count=0,
                total_iterations=0,
            )
        articles = await self.list_articles_by_topic(topic_id, limit=500)
        skills = await self.list_skills_by_topic(topic_id)
        contributors = {a.user_id for a in articles if a.user_id is not None} | {
            s.created_by for s in skills
        }
        avg_quality = (
            sum(a.quality_score for a in articles) / len(articles)
            if articles
            else 0.0
        )
        eligible = len(articles) >= 3 and avg_quality >= 0.6
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
        )

    # ---- helpers --------------------------------------------------------

    def _require_client(self) -> Any:
        if self._client is None:
            raise ExternalAPIError(
                "SupabaseBackend not connected — call connect() first",
                details={"backend": "supabase"},
            )
        return self._client

    async def _fetch_single(self, table: str, where: dict) -> Optional[dict]:
        client = self._require_client()

        def _run() -> Optional[dict]:
            q = client.table(table).select("*")
            for k, v in where.items():
                q = q.eq(k, v)
            r = q.limit(1).execute()
            data = r.data or []
            return data[0] if data else None

        return await asyncio.to_thread(_run)

    async def _upsert(self, table: str, row: dict) -> None:
        client = self._require_client()

        def _run() -> None:
            client.table(table).upsert(row).execute()

        try:
            await asyncio.to_thread(_run)
        except Exception as exc:
            log.error("supabase.upsert_failed", table=table, error=str(exc), exc_info=True)
            raise ExternalAPIError(
                f"Supabase upsert {table} failed",
                details={"table": table, "error": str(exc)},
            ) from exc


__all__ = ["SupabaseBackend"]
