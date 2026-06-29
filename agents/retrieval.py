"""
RetrievalAgent — parallel multi-source search.

Tiered architecture:
  Tier 0: Curated Index (<500ms, Supabase — stub until Phase 5)
  Tier 1: APIs in parallel (Thingiverse + MakerWorld)
  Tier 2: Scraping (Printables)
  Stops tier when ≥4 candidates found.

Input:  ctx.intent (from IntentAgent)
Output: ctx.candidates (list[dict])
"""

from __future__ import annotations
import asyncio
import logging

from orynd_core.agents.base import AgentContext, AgentResult, BaseAgent
from orynd_core.models.schemas import Candidate
from orynd_core.services.search import curated, thingiverse, makerworld, printables

log = logging.getLogger(__name__)

_STOP_AT = 4   # enough candidates to stop the next tier


def _candidate_to_dict(c: Candidate) -> dict:
    return c.model_dump()


def _deduplicate(candidates: list[dict]) -> list[dict]:
    seen_names: set[str] = set()
    seen_stls: set[str] = set()
    result = []
    for c in candidates:
        key_name = c.get("name", "").lower().strip()
        key_stl = c.get("stl_url", "")
        if key_name in seen_names or (key_stl and key_stl in seen_stls):
            continue
        seen_names.add(key_name)
        if key_stl:
            seen_stls.add(key_stl)
        result.append(c)
    return result


def _rank(candidates: list[dict]) -> list[dict]:
    return sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)


class RetrievalAgent(BaseAgent):
    """
    Fetches candidates from all configured sources in parallel tiers.
    No LLM required — pure algorithm.
    """

    name = "retrieval_agent"

    def __init__(self, limit: int = 5) -> None:
        super().__init__(provider=None)
        self.limit = limit

    async def run_logic(self, ctx: AgentContext) -> AgentResult:
        intent = ctx.intent or {}
        keywords = intent.get("keywords", [])
        if isinstance(keywords, list):
            query = " ".join(keywords)
        else:
            query = str(keywords)

        # Use object_name as primary query if available (more precise)
        object_name = intent.get("object_name", "")
        if object_name and object_name != "3d model":
            query = object_name + " " + query

        # For search intent, prefer the search_query from vision if available
        vision = ctx.extra.get("vision", {})
        if vision.get("search_query"):
            query = vision["search_query"]

        query = query.strip() or ctx.raw_text or ""

        if not query:
            ctx.candidates = []
            return AgentResult.success(self.name, {"sources": [], "total": 0})

        # Clean query: remove non-search words, limit length
        query = " ".join(query.split()[:8])  # max 8 words

        all_candidates: list[dict] = []
        sources_used: list[str] = []

        # ── Tier 0: Curated Index ────────────────────────────────────────────
        try:
            tier0 = await curated.search(query, limit=self.limit)
            if tier0:
                all_candidates.extend(_candidate_to_dict(c) for c in tier0)
                sources_used.append("curated")
                log.debug("[retrieval] tier0 curated: %d results", len(tier0))
        except Exception as e:
            log.warning("[retrieval] curated failed: %s", e)

        if len(all_candidates) >= _STOP_AT:
            ctx.candidates = _rank(_deduplicate(all_candidates))[:self.limit]
            return AgentResult.success(self.name, {"sources": sources_used, "total": len(ctx.candidates)})

        # ── Tier 1: Parallel APIs ────────────────────────────────────────────
        tv_task = asyncio.create_task(thingiverse.search(query, limit=self.limit))
        mw_task = asyncio.create_task(makerworld.search(query, limit=self.limit))

        tier1_results = await asyncio.gather(tv_task, mw_task, return_exceptions=True)

        for source_name, result in zip(["thingiverse", "makerworld"], tier1_results):
            if isinstance(result, Exception):
                log.warning("[retrieval] %s failed: %s", source_name, result)
                continue
            if result:
                all_candidates.extend(_candidate_to_dict(c) for c in result)
                sources_used.append(source_name)
                log.debug("[retrieval] %s: %d results", source_name, len(result))

        if len(all_candidates) >= _STOP_AT:
            ctx.candidates = _rank(_deduplicate(all_candidates))[:self.limit]
            return AgentResult.success(self.name, {"sources": sources_used, "total": len(ctx.candidates)})

        # ── Tier 2: Scraping (Printables) ────────────────────────────────────
        try:
            tier2 = await printables.search(query, limit=self.limit)
            if tier2:
                all_candidates.extend(_candidate_to_dict(c) for c in tier2)
                sources_used.append("printables")
                log.debug("[retrieval] printables: %d results", len(tier2))
        except Exception as e:
            log.warning("[retrieval] printables failed: %s", e)

        ctx.candidates = _rank(_deduplicate(all_candidates))[:self.limit]
        return AgentResult.success(
            self.name,
            {"sources": sources_used, "total": len(ctx.candidates)},
        )
