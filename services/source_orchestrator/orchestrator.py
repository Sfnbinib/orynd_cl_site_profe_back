"""
SourceAccessOrchestrator — coordinates queries across all 242+ sources.

Strategy:
  1. Pick source candidates from registry (by category/region).
  2. For each, route to the right adapter (dedicated → generic_html → browser_harness → ddg fallback).
  3. Run in parallel batches with timeout + circuit breaker per source.
  4. Aggregate, deduplicate, rank hits.
  5. Update reliability scores from observed outcomes.

Usage:
    orch = SourceAccessOrchestrator()
    hits = await orch.search("mounting bracket", category=SourceCategory.THREE_D_MODELS)
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Optional

from .registry import SourceRegistry, Source, SourceCategory, AccessMethod, get_registry
from .adapters import (
    AdapterBase,
    SearchHit,
    AdapterError,
    AdapterBlocked,
    GenericHTMLAdapter,
    BrowserHarnessAdapter,
    DuckDuckGoFallbackAdapter,
    DedicatedProxyAdapter,
)

log = logging.getLogger(__name__)


class SourceAccessOrchestrator:
    def __init__(
        self,
        registry: Optional[SourceRegistry] = None,
        enable_browser: bool = True,
        enable_ddg_fallback: bool = True,
        max_parallel: int = 5,
        per_source_timeout_s: int = 15,
    ):
        self.registry = registry or get_registry()
        self.enable_browser = enable_browser
        self.enable_ddg_fallback = enable_ddg_fallback
        self.max_parallel = max_parallel
        self.per_source_timeout_s = per_source_timeout_s

        # Adapter instances (cached)
        self._generic = GenericHTMLAdapter()
        self._browser = BrowserHarnessAdapter(use_browser_use=False)
        self._ddg = DuckDuckGoFallbackAdapter()
        self._dedicated_cache: dict[str, DedicatedProxyAdapter] = {}

    # ─── Adapter selection ───────────────────────────────────
    def _select_adapter(self, source: Source) -> AdapterBase:
        if source.has_dedicated_adapter and source.adapter_module:
            if source.adapter_module not in self._dedicated_cache:
                self._dedicated_cache[source.adapter_module] = DedicatedProxyAdapter(source.adapter_module)
            return self._dedicated_cache[source.adapter_module]

        if source.access_method == AccessMethod.BROWSER_HARNESS and self.enable_browser:
            return self._browser

        return self._generic

    # ─── Public search ───────────────────────────────────────
    async def search(
        self,
        query: str,
        category: Optional[SourceCategory] = None,
        region: Optional[str] = None,
        max_sources: int = 10,
        limit_per_source: int = 5,
        priority_min: str = "medium",
    ) -> list[SearchHit]:
        """
        Search across N most-relevant sources, return aggregated hits.
        """
        # Pick candidate sources
        candidates = self.registry.query(
            category=category,
            region=region,
            priority_min=priority_min,
        )
        if not candidates:
            log.warning("[orchestrator] no candidate sources matched")
            return []

        # Limit to max_sources, prefer dedicated adapters
        selected = candidates[:max_sources]
        log.info("[orchestrator] querying %d sources for '%s'", len(selected), query)

        # Parallel batches
        all_hits: list[SearchHit] = []
        sem = asyncio.Semaphore(self.max_parallel)

        async def query_one(source: Source) -> list[SearchHit]:
            async with sem:
                adapter = self._select_adapter(source)
                try:
                    hits = await asyncio.wait_for(
                        adapter.search(query, source.url, limit=limit_per_source),
                        timeout=self.per_source_timeout_s,
                    )
                    self._record_success(source, len(hits))
                    return hits
                except asyncio.TimeoutError:
                    log.debug("[orchestrator] timeout %s", source.site_id)
                    self._record_failure(source, "timeout")
                except AdapterBlocked as e:
                    log.debug("[orchestrator] blocked %s: %s", source.site_id, e)
                    self._record_failure(source, "blocked")
                except AdapterError as e:
                    log.debug("[orchestrator] adapter_error %s: %s", source.site_id, e)
                    self._record_failure(source, "error")
                except Exception as e:
                    log.warning("[orchestrator] unexpected %s: %s", source.site_id, e)
                    self._record_failure(source, "exception")
                return []

        tasks = [query_one(s) for s in selected]
        batch_results = await asyncio.gather(*tasks, return_exceptions=False)

        for hits in batch_results:
            all_hits.extend(hits)

        # Optionally try DDG fallback for sources that returned nothing
        if self.enable_ddg_fallback and not all_hits:
            log.info("[orchestrator] no results — trying DDG fallback")
            for s in selected[:3]:
                try:
                    hits = await asyncio.wait_for(
                        self._ddg.search(query, s.url, limit=limit_per_source),
                        timeout=10,
                    )
                    all_hits.extend(hits)
                except Exception as e:
                    log.debug("[orchestrator] DDG fallback failed for %s: %s", s.site_id, e)

        # Deduplicate by URL
        all_hits = self._deduplicate(all_hits)
        # Rank
        all_hits.sort(key=lambda h: h.score, reverse=True)
        return all_hits

    def _deduplicate(self, hits: list[SearchHit]) -> list[SearchHit]:
        seen: set[str] = set()
        out: list[SearchHit] = []
        for h in hits:
            key = (h.url or "").lower().rstrip("/")
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(h)
        return out

    # ─── Reliability tracking ────────────────────────────────
    def _record_success(self, source: Source, hit_count: int) -> None:
        alpha = 0.1
        source.success_rate = (1 - alpha) * source.success_rate + alpha * (1.0 if hit_count > 0 else 0.5)
        source.reliability_score = source.success_rate

    def _record_failure(self, source: Source, reason: str) -> None:
        alpha = 0.1
        source.success_rate = (1 - alpha) * source.success_rate
        source.reliability_score = source.success_rate
