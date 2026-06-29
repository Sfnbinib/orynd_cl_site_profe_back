"""
Adapter base classes for source orchestrator.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchHit:
    """One search result from any adapter."""
    title: str
    url: str
    snippet: str = ""
    source_id: str = ""
    source_name: str = ""

    # Optional richer fields
    thumbnail_url: Optional[str] = None
    download_url: Optional[str] = None
    file_format: Optional[str] = None  # stl, step, 3mf, etc.
    score: float = 0.5  # 0..1 relevance

    # Provenance
    adapter_used: str = ""
    latency_ms: int = 0
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "thumbnail_url": self.thumbnail_url,
            "download_url": self.download_url,
            "file_format": self.file_format,
            "score": self.score,
            "adapter_used": self.adapter_used,
            "latency_ms": self.latency_ms,
        }


class AdapterError(Exception):
    pass


class AdapterTimeout(AdapterError):
    pass


class AdapterBlocked(AdapterError):
    """Site blocks our access (anti-bot, paywall, etc)."""
    pass


class AdapterBase(ABC):
    """Base class for all source adapters."""

    name: str
    timeout_s: int = 15
    max_retries: int = 2

    @abstractmethod
    async def search(self, query: str, source_url: str, limit: int = 10) -> list[SearchHit]:
        """Execute query against the source. Raise AdapterError on failure."""
        raise NotImplementedError

    async def health_check(self, source_url: str) -> bool:
        """Optional override — check if source is reachable."""
        return True
