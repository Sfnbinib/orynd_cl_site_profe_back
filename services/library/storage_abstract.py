"""StorageBackend interface — concrete impls live next to this file."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from orynd_core.services.library.schemas import (
    Article,
    Hypothesis,
    Layer,
    ResearchSession,
    Skill,
    Topic,
    TopicStage,
)


class ArticleSearchResult(BaseModel):
    article: Article
    score: float = 0.0  # 0..1, higher = more relevant


class StageMetrics(BaseModel):
    topic_id: UUID
    current_stage: TopicStage
    article_count: int
    skill_count: int
    contributor_count: int
    total_iterations: int
    avg_quality_score: float = 0.0
    coverage_ratio: float = 0.0  # 0..1
    promotion_eligible: bool = False
    rationale: list[str] = Field(default_factory=list)


class StorageBackend(ABC):
    """Backend-agnostic CRUD + search."""

    # ---- lifecycle -------------------------------------------------------

    async def connect(self) -> None:
        """Initialise the backend (pool, client). Override when needed."""

    async def disconnect(self) -> None:
        """Release the backend (pool, client). Override when needed."""

    # ---- topics ----------------------------------------------------------

    @abstractmethod
    async def get_topic(self, topic_id: UUID) -> Optional[Topic]: ...

    @abstractmethod
    async def find_topic_by_slug(self, slug: str) -> Optional[Topic]: ...

    @abstractmethod
    async def search_topics(self, query: str, k: int = 10) -> list[Topic]: ...

    @abstractmethod
    async def list_topics(
        self,
        *,
        layer: Optional[Layer] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Topic]: ...

    @abstractmethod
    async def upsert_topic(self, topic: Topic) -> Topic: ...

    # ---- articles --------------------------------------------------------

    @abstractmethod
    async def get_article(self, article_id: UUID) -> Optional[Article]: ...

    @abstractmethod
    async def list_articles_by_topic(
        self,
        topic_id: UUID,
        layer: Optional[Layer] = None,
        limit: int = 50,
    ) -> list[Article]: ...

    @abstractmethod
    async def search_articles_semantic(
        self,
        query: str,
        k: int = 10,
        topic_id: Optional[UUID] = None,
    ) -> list[ArticleSearchResult]: ...

    @abstractmethod
    async def search_articles_fts(
        self,
        query: str,
        k: int = 10,
        topic_id: Optional[UUID] = None,
    ) -> list[ArticleSearchResult]: ...

    @abstractmethod
    async def upsert_article(self, article: Article) -> Article: ...

    # ---- skills ----------------------------------------------------------

    @abstractmethod
    async def get_skill(self, skill_id: UUID) -> Optional[Skill]: ...

    @abstractmethod
    async def find_skill_by_slug(self, slug: str) -> Optional[Skill]: ...

    @abstractmethod
    async def list_skills_by_topic(self, topic_id: UUID) -> list[Skill]: ...

    @abstractmethod
    async def upsert_skill(self, skill: Skill) -> Skill: ...

    # ---- research sessions ----------------------------------------------

    @abstractmethod
    async def create_session(self, session: ResearchSession) -> ResearchSession: ...

    @abstractmethod
    async def update_session(self, session: ResearchSession) -> ResearchSession: ...

    @abstractmethod
    async def get_session(self, session_id: UUID) -> Optional[ResearchSession]: ...

    # ---- hypotheses ------------------------------------------------------

    @abstractmethod
    async def upsert_hypothesis(self, h: Hypothesis) -> Hypothesis: ...

    @abstractmethod
    async def list_hypotheses_by_topic(self, topic_id: UUID) -> list[Hypothesis]: ...

    # ---- aggregations ---------------------------------------------------

    @abstractmethod
    async def topic_metrics(self, topic_id: UUID) -> StageMetrics: ...
