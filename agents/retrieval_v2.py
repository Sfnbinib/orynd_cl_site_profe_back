"""
RetrievalAgent V2 — uses SourceAccessOrchestrator with full sites registry.

Replaces the old RetrievalAgent that only knew about 4 hardcoded sources.
Now sees all 184+ sources loaded from research JSONL files.

To switch: in workspace.py / chat.py, replace `RetrievalAgent` with `RetrievalAgentV2`.
"""
from __future__ import annotations
import logging

from orynd_core.agents.base import AgentContext, AgentResult, BaseAgent
from orynd_core.services.source_orchestrator import (
    SourceAccessOrchestrator,
    SourceCategory,
)

log = logging.getLogger(__name__)


def _hit_to_candidate(hit) -> dict:
    """Convert SearchHit → Candidate dict for downstream agents."""
    return {
        "name": hit.title,
        "url": hit.url,
        "source": hit.source_name,
        "thumbnail_url": hit.thumbnail_url,
        "stl_url": hit.download_url if hit.file_format == "stl" else None,
        "download_url": hit.download_url,
        "file_format": hit.file_format,
        "description": hit.snippet,
        "score": hit.score,
        "adapter": hit.adapter_used,
        "latency_ms": hit.latency_ms,
    }


def _category_from_intent(intent: dict) -> SourceCategory | None:
    """Infer SourceCategory from intent keywords."""
    keywords = intent.get("keywords", []) if isinstance(intent.get("keywords"), list) else []
    text = " ".join(str(k) for k in keywords).lower()
    text += " " + str(intent.get("object_name", "")).lower()

    if any(t in text for t in ("model", "stl", "print", "fdm", "3d")):
        return SourceCategory.THREE_D_MODELS
    if any(t in text for t in ("cad", "cadquery", "freecad", "solidworks")):
        return SourceCategory.CAD_RESOURCE
    if any(t in text for t in ("paper", "research", "arxiv")):
        return SourceCategory.ACADEMIC
    if any(t in text for t in ("code", "github", "library")):
        return SourceCategory.CODE
    return None


class RetrievalAgentV2(BaseAgent):
    """
    New retrieval agent backed by SourceAccessOrchestrator (full registry).

    Reads ctx.intent → infers category → queries orchestrator → stores candidates.
    """

    name = "retrieval_agent_v2"

    def __init__(
        self,
        limit: int = 8,
        max_sources: int = 6,
        enable_browser: bool = False,  # off by default — opt-in via ctx.extra
    ):
        super().__init__(provider=None)
        self.limit = limit
        self.max_sources = max_sources
        self._orch = SourceAccessOrchestrator(
            enable_browser=enable_browser,
            enable_ddg_fallback=True,
            max_parallel=4,
            per_source_timeout_s=15,
        )

    async def run_logic(self, ctx: AgentContext) -> AgentResult:
        intent = ctx.intent or {}
        keywords = intent.get("keywords", [])
        if isinstance(keywords, list):
            query = " ".join(keywords)
        else:
            query = str(keywords)

        object_name = intent.get("object_name", "")
        if object_name and object_name != "3d model":
            query = (object_name + " " + query).strip()

        vision = ctx.extra.get("vision", {})
        if vision.get("search_query"):
            query = vision["search_query"]

        query = (query or ctx.raw_text or "").strip()
        if not query:
            ctx.candidates = []
            return AgentResult.success(self.name, {"sources": [], "total": 0})

        # Clean query
        query = " ".join(query.split()[:8])

        # Allow per-request override
        category = _category_from_intent(intent)
        region = ctx.extra.get("preferred_region")
        max_sources = ctx.extra.get("max_sources", self.max_sources)
        enable_browser = ctx.extra.get("enable_browser", False)

        # Switch browser mode for this request
        self._orch.enable_browser = enable_browser

        log.info(
            "[retrieval_v2] query='%s' category=%s region=%s max_sources=%d browser=%s",
            query, category, region, max_sources, enable_browser,
        )

        hits = await self._orch.search(
            query=query,
            category=category,
            region=region,
            max_sources=max_sources,
            limit_per_source=5,
            priority_min="medium",
        )

        candidates = [_hit_to_candidate(h) for h in hits[:self.limit]]
        ctx.candidates = candidates

        sources_used = sorted({h.source_name for h in hits})
        return AgentResult.success(
            self.name,
            {
                "sources": sources_used,
                "total": len(candidates),
                "registry_size": self._orch.registry.count(),
            },
        )
