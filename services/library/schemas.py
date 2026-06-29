"""Pydantic models — Knowledge Library data plane.

Spec: ``02_data_model.md``.

All models use UUID primary keys, ISO-8601 timestamps, and explicit OPEN/CLOSED
layer flags. ``embedding`` is intentionally Optional — populated only when an
embedding service is wired in (Phase 11).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Layer(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class TopicStage(int, Enum):
    STAGE_0 = 0
    STAGE_1 = 1
    STAGE_2 = 2
    STAGE_3 = 3
    STAGE_4 = 4
    STAGE_5 = 5
    STAGE_6 = 6
    STAGE_7 = 7
    STAGE_8 = 8
    STAGE_9 = 9
    STAGE_10 = 10


# ---- Topic ---------------------------------------------------------------


class StageTransition(BaseModel):
    from_stage: TopicStage
    to_stage: TopicStage
    transition_at: datetime
    triggered_by: Literal["topic_capacity_agent", "manual", "data_threshold"]
    evidence: dict


class Topic(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    id: UUID = Field(default_factory=uuid4)
    slug: str
    title: str
    description: str = ""

    current_stage: TopicStage = TopicStage.STAGE_0
    stage_history: list[StageTransition] = Field(default_factory=list)

    article_count: int = 0
    skill_count: int = 0
    contributor_count: int = 0
    total_iterations: int = 0

    parent_topic_id: Optional[UUID] = None
    subtopic_ids: list[UUID] = Field(default_factory=list)
    related_topic_ids: list[UUID] = Field(default_factory=list)

    estimated_total_iterations: int = 100
    progress_ratio: float = 0.0

    embedding: Optional[list[float]] = None
    keywords: list[str] = Field(default_factory=list)

    layer: Layer = Layer.OPEN

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    created_by: UUID


# ---- Article / Claim / Source -------------------------------------------


class Source(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    url: str
    title: str = ""
    accessed_at: datetime = Field(default_factory=_utcnow)
    reliability_score: float = 0.5
    layer: Literal["primary", "secondary", "tertiary"] = "secondary"


class Claim(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    article_id: UUID
    text: str
    citation_ids: list[UUID] = Field(default_factory=list)
    confidence: float = 1.0
    hallucination_flag: bool = False


class UserRating(BaseModel):
    user_id: UUID
    rating: int = Field(ge=1, le=5)
    rated_at: datetime
    level: Literal["user", "session"]


class Article(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: UUID = Field(default_factory=uuid4)
    topic_id: UUID

    title: str
    abstract: str = ""
    body_md: str

    authored_by: Literal["human", "llm", "hybrid"]
    user_id: Optional[UUID] = None
    llm_model: Optional[str] = None

    claims: list[Claim] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)

    quality_score: float = 0.0
    hallucination_score: float = 0.0
    user_ratings: list[UserRating] = Field(default_factory=list)

    contributes_to_stage: TopicStage = TopicStage.STAGE_0

    layer: Layer = Layer.CLOSED
    published_to_open_at: Optional[datetime] = None
    license: Optional[str] = "CC-BY-NC-4.0"

    embedding: Optional[list[float]] = None
    is_ai_marked: bool = True

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


# ---- Skill ---------------------------------------------------------------


class Skill(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    slug: str
    name: str
    description: str = ""

    topic_id: UUID
    derived_from_articles: list[UUID] = Field(default_factory=list)

    signature: dict
    prompt_template: str
    few_shot_examples: list[dict] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)

    version: str = "0.1.0"
    parent_version: Optional[str] = None

    success_rate: float = 0.0
    usage_count: int = 0

    layer: Layer = Layer.CLOSED
    published_to_open_at: Optional[datetime] = None
    install_mode: Literal["local", "pass_through"] = "pass_through"

    storage_location: str = ""

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    created_by: UUID


# ---- Research session ---------------------------------------------------


class PathwayStep(BaseModel):
    step_id: str
    input: dict
    output: dict
    duration_ms: int
    tokens: int


class ResearchSession(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    topic_id: UUID
    user_id: UUID

    query: str
    mode: Literal["quick", "deep", "hyper"]

    pathway_steps: list[PathwayStep] = Field(default_factory=list)

    articles_generated: list[UUID] = Field(default_factory=list)
    skills_updated: list[UUID] = Field(default_factory=list)
    sources_consulted: list[UUID] = Field(default_factory=list)

    contribution_pct_to_topic: float = 0.0
    user_specific_pct: float = 1.0

    tokens_consumed: int = 0
    duration_seconds: int = 0

    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: Optional[datetime] = None
    layer: Layer = Layer.CLOSED


# ---- Hypothesis ----------------------------------------------------------


class Hypothesis(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    topic_id: UUID
    text: str

    derived_from_articles: list[UUID] = Field(default_factory=list)
    generated_by: Literal["llm", "user"]

    status: Literal["pending", "validated", "rejected", "needs_more_data"] = "pending"
    validation_evidence: list[UUID] = Field(default_factory=list)

    initial_confidence: float = 0.5
    current_confidence: float = 0.5
    confidence_history: list[Tuple[datetime, float]] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=_utcnow)
    layer: Layer = Layer.CLOSED


# ---- Contribution log ---------------------------------------------------


class Contribution(BaseModel):
    user_id: UUID
    topic_id: UUID
    contribution_type: Literal["article", "skill", "hypothesis", "rating"]
    artifact_id: UUID
    contribution_value: float
    at: datetime = Field(default_factory=_utcnow)


__all__ = [
    "Layer",
    "TopicStage",
    "StageTransition",
    "Topic",
    "Source",
    "Claim",
    "UserRating",
    "Article",
    "Skill",
    "PathwayStep",
    "ResearchSession",
    "Hypothesis",
    "Contribution",
]
