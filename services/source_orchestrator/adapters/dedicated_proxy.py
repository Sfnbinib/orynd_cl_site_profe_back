"""
DedicatedProxyAdapter — wraps existing dedicated adapters (Printables, Thingiverse, etc.)
into the unified AdapterBase interface.

Allows orchestrator to treat all sources uniformly.
"""
from __future__ import annotations
import importlib
import logging
import time
from typing import Optional

from .base import AdapterBase, SearchHit, AdapterError

log = logging.getLogger(__name__)


class DedicatedProxyAdapter(AdapterBase):
    """Routes to existing dedicated module (services/search/{module_name}.py)."""

    name = "dedicated_proxy"

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.name = f"dedicated:{module_name}"
        self._module = None

    def _load(self):
        if self._module is None:
            try:
                self._module = importlib.import_module(
                    f"orynd_core.services.search.{self.module_name}"
                )
            except ImportError as e:
                raise AdapterError(f"adapter module not found: {self.module_name} ({e})")
        return self._module

    async def search(self, query: str, source_url: str, limit: int = 10) -> list[SearchHit]:
        mod = self._load()
        if not hasattr(mod, "search"):
            raise AdapterError(f"{self.module_name} has no search() function")

        start = time.monotonic()
        try:
            result = await mod.search(query, limit=limit)
        except Exception as e:
            raise AdapterError(f"dedicated:{self.module_name} failed: {e}")

        latency = int((time.monotonic() - start) * 1000)
        hits: list[SearchHit] = []

        for c in result:
            # Existing modules return Candidate objects (pydantic) or dicts
            if hasattr(c, "model_dump"):
                d = c.model_dump()
            elif isinstance(c, dict):
                d = c
            else:
                continue

            hits.append(SearchHit(
                title=d.get("name", "") or d.get("title", ""),
                url=d.get("url", "") or d.get("model_url", ""),
                snippet=d.get("description", "")[:200] if d.get("description") else "",
                source_id=self.module_name,
                source_name=self.module_name,
                thumbnail_url=d.get("thumbnail_url") or d.get("image_url"),
                download_url=d.get("stl_url") or d.get("download_url"),
                file_format="stl" if d.get("stl_url") else None,
                score=float(d.get("score", 0.7)),
                adapter_used=self.name,
                latency_ms=latency,
                raw=d,
            ))

        return hits
