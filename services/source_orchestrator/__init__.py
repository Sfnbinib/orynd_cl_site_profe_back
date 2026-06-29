"""
Source Access Orchestrator — real implementation.

Loads 242+ sources from research JSONL files into in-memory registry.
Routes queries to the right adapter:
  - Dedicated API adapters (Printables, Thingiverse, MakerWorld, GitHub) — fast
  - Generic HTML adapter — for sites with simple HTML pages
  - Browser harness adapter — for JS-heavy sites
  - DuckDuckGo fallback — last resort

Public API:
    from orynd_core.services.source_orchestrator import (
        SourceRegistry, SourceAccessOrchestrator, get_registry
    )
"""
from .registry import SourceRegistry, Source, SourceCategory, get_registry
from .orchestrator import SourceAccessOrchestrator
from .adapters.base import SearchHit, AdapterBase

__all__ = [
    "SourceRegistry",
    "Source",
    "SourceCategory",
    "get_registry",
    "SourceAccessOrchestrator",
    "SearchHit",
    "AdapterBase",
]
