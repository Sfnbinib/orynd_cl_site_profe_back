"""/sources/* — Source Access Orchestrator HTTP surface.

Thin wrapper over the existing :mod:`orynd_core.services.source_orchestrator`
(registry + orchestrator already built and unit-tested). This router adds:

* ``GET  /sources``                   list with filters
* ``GET  /sources/stats``             registry summary (totals, by category, by region)
* ``GET  /sources/categories``        enum values for the UI
* ``GET  /sources/{site_id}``         single source detail
* ``GET  /sources/{site_id}/health``  registry-cached health
* ``POST /sources/query``             execute query (parallel, deduplicated)

The orchestrator instance is cached per-process and lazily constructed so
import-time stays cheap and tests can override it.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from orynd_core.services.observability.metrics import external_api_calls_total
from orynd_core.services.source_orchestrator import (
    SourceAccessOrchestrator,
    SourceCategory,
    get_registry,
)
from orynd_core.services.source_orchestrator.adapters.base import SearchHit

router = APIRouter(prefix="/sources", tags=["sources"])

_orchestrator: Optional[SourceAccessOrchestrator] = None


def get_orchestrator() -> SourceAccessOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        # Browser harness disabled by default in HTTP path — requires extra
        # disk + Chromium. Phase 14 toggles it on for the deep-research flow.
        _orchestrator = SourceAccessOrchestrator(enable_browser=False)
    return _orchestrator


def reset_orchestrator() -> None:
    """Test hook — clears the cached singleton."""
    global _orchestrator
    _orchestrator = None


# ---- payload models -----------------------------------------------------


class SourceSummary(BaseModel):
    site_id: str
    name: str
    url: str
    category: str
    language: str = "en"
    region: str = "global"
    priority: str = "medium"
    has_dedicated_adapter: bool
    access_method: str
    reliability_score: float
    success_rate: float
    cad_models_present: bool = False


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=300)
    category: Optional[SourceCategory] = None
    region: Optional[str] = None
    max_sources: int = Field(10, ge=1, le=50)
    limit_per_source: int = Field(5, ge=1, le=25)
    priority_min: str = Field("medium", pattern="^(low|medium|high)$")


class QueryResponse(BaseModel):
    query: str
    total_hits: int
    hits: list[SearchHit]


# ---- helpers ------------------------------------------------------------


def _summarise(source) -> SourceSummary:
    return SourceSummary(
        site_id=source.site_id,
        name=source.name or source.site_id,
        url=source.url,
        category=source.category.value
        if hasattr(source.category, "value")
        else str(source.category),
        language=source.language,
        region=source.region,
        priority=source.priority,
        has_dedicated_adapter=source.has_dedicated_adapter,
        access_method=source.access_method.value,
        reliability_score=source.reliability_score,
        success_rate=source.success_rate,
        cad_models_present=source.cad_models_present,
    )


# ---- endpoints ----------------------------------------------------------


@router.get("/categories")
async def list_categories() -> list[str]:
    return [c.value for c in SourceCategory]


@router.get("/stats")
async def stats() -> dict:
    return get_registry().stats()


@router.get("", response_model=list[SourceSummary])
async def list_sources(
    category: Optional[SourceCategory] = None,
    region: Optional[str] = None,
    priority_min: str = Query("low", pattern="^(low|medium|high)$"),
    has_dedicated_only: bool = False,
    cad_models_only: bool = False,
    limit: int = Query(100, ge=1, le=500),
):
    registry = get_registry()
    matches = registry.query(
        category=category,
        region=region,
        priority_min=priority_min,
        has_dedicated_only=has_dedicated_only,
        cad_models_only=cad_models_only,
    )
    return [_summarise(s) for s in matches[:limit]]


@router.get("/{site_id}", response_model=SourceSummary)
async def get_source(site_id: str):
    source = get_registry().get(site_id)
    if not source:
        raise HTTPException(status_code=404, detail="source not found")
    return _summarise(source)


@router.get("/{site_id}/health")
async def source_health(site_id: str) -> dict:
    source = get_registry().get(site_id)
    if not source:
        raise HTTPException(status_code=404, detail="source not found")
    # Cheap registry-only health — does NOT make a live request.
    # Live probing happens on actual query (reliability is EWMA-updated then).
    return {
        "site_id": source.site_id,
        "reliability_score": source.reliability_score,
        "success_rate": source.success_rate,
        "avg_latency_ms": source.avg_latency_ms,
        "has_dedicated_adapter": source.has_dedicated_adapter,
        "access_method": source.access_method.value,
    }


@router.post("/query", response_model=QueryResponse)
async def query_sources(payload: QueryRequest):
    orch = get_orchestrator()
    try:
        hits = await orch.search(
            payload.query,
            category=payload.category,
            region=payload.region,
            max_sources=payload.max_sources,
            limit_per_source=payload.limit_per_source,
            priority_min=payload.priority_min,
        )
        external_api_calls_total.labels(service="source_orchestrator", status="ok").inc()
    except Exception:
        external_api_calls_total.labels(service="source_orchestrator", status="error").inc()
        raise
    return QueryResponse(query=payload.query, total_hits=len(hits), hits=hits)
