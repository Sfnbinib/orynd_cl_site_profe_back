"""
SourceRegistry — loads all known sources from research JSONL files.

Founder ask: "у нас 242+ источников в research, почему они не используются?"
Answer: this module loads them.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class SourceCategory(str, Enum):
    THREE_D_MODELS = "3d_models"
    CAD_RESOURCE = "cad_resource_platform"
    MANUFACTURER_COMMUNITY = "manufacturer_community"
    ENGINEERING_FORUM = "engineering_forum"
    ENGINEERING_DOCS = "engineering_docs"
    ACADEMIC = "academic"
    GENERAL_WEB = "general_web"
    CODE = "code"
    VIDEO = "video"
    OTHER = "other"


class AccessMethod(str, Enum):
    DEDICATED_API = "dedicated_api"        # Printables, Thingiverse, etc — fast
    GENERIC_HTML = "generic_html"          # simple httpx + BS4 scrape
    BROWSER_HARNESS = "browser_harness"    # browser-use (JS-heavy or anti-bot)
    SEARCH_FALLBACK = "search_fallback"    # DuckDuckGo for unindexed


@dataclass
class Source:
    """One known source."""
    site_id: str
    name: str
    url: str
    category: SourceCategory = SourceCategory.OTHER
    language: str = "en"
    region: str = "global"
    description: str = ""
    priority: str = "medium"  # high / medium / low

    # Capability flags from JSONL
    search_api_available: bool = False
    cad_models_present: bool = False
    content_types: list[str] = field(default_factory=list)

    # Determined by orchestrator
    access_method: AccessMethod = AccessMethod.GENERIC_HTML
    has_dedicated_adapter: bool = False
    adapter_module: Optional[str] = None  # e.g. "printables"

    # Reliability (learned over time)
    reliability_score: float = 0.5
    success_rate: float = 0.5
    avg_latency_ms: int = 2000

    # Cost
    free_tier_qpd: Optional[int] = None
    rate_limit_per_min: int = 30

    @classmethod
    def from_jsonl_row(cls, row: dict) -> "Source":
        """Parse a row from qwen_output_task*.jsonl format."""
        # Determine access method
        if row.get("search_api_available"):
            access = AccessMethod.GENERIC_HTML  # we don't know if there's a dedicated adapter yet
        else:
            access = AccessMethod.BROWSER_HARNESS

        # Map category
        cat_str = row.get("category", "other")
        try:
            category = SourceCategory(cat_str)
        except ValueError:
            category = SourceCategory.OTHER

        return cls(
            site_id=row.get("site_id", row.get("url", "unknown")),
            name=row.get("name", ""),
            url=row.get("url", ""),
            category=category,
            language=row.get("language", "en"),
            region=row.get("region", "global"),
            description=row.get("description", ""),
            priority=row.get("priority", "medium"),
            search_api_available=row.get("search_api_available", False),
            cad_models_present=row.get("cad_models_present", False),
            content_types=row.get("content_types", []),
            access_method=access,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["category"] = self.category.value
        d["access_method"] = self.access_method.value
        return d


# ─── Dedicated adapter registry ──────────────────────────────
# Map domain → existing dedicated adapter module name
DEDICATED_ADAPTERS = {
    "printables.com": "printables",
    "thingiverse.com": "thingiverse",
    "makerworld.com": "makerworld",
    "github.com": "github_search",
    "duckduckgo.com": "web",
}


class SourceRegistry:
    """In-memory registry loaded from JSONL files."""

    def __init__(self):
        self._sources: dict[str, Source] = {}
        self._by_category: dict[SourceCategory, list[Source]] = {}
        self._by_region: dict[str, list[Source]] = {}

    def load_from_jsonl_dir(self, jsonl_dir: Path) -> int:
        """Load all *.jsonl files in directory. Returns count loaded."""
        if not jsonl_dir.exists():
            log.warning("Sources dir does not exist: %s", jsonl_dir)
            return 0

        loaded = 0
        for jsonl_path in sorted(jsonl_dir.glob("*.jsonl")):
            loaded += self.load_jsonl_file(jsonl_path)
        log.info("[source_registry] Loaded %d sources from %s", loaded, jsonl_dir)
        return loaded

    def load_jsonl_file(self, path: Path) -> int:
        """Load one JSONL file."""
        count = 0
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        log.debug("Bad JSON in %s:%d: %s", path.name, line_no, e)
                        continue
                    # Skip rows without proper site_id
                    if not row.get("site_id") and not row.get("url"):
                        continue
                    source = Source.from_jsonl_row(row)
                    self._register(source)
                    count += 1
        except Exception as e:
            log.warning("Failed to load %s: %s", path, e)
        return count

    def _register(self, source: Source) -> None:
        # Map to dedicated adapter if available
        for domain, adapter_name in DEDICATED_ADAPTERS.items():
            if domain in source.url.lower():
                source.has_dedicated_adapter = True
                source.adapter_module = adapter_name
                source.access_method = AccessMethod.DEDICATED_API
                break

        # Dedupe by URL
        if source.site_id in self._sources:
            existing = self._sources[source.site_id]
            # Merge: keep more complete description / higher priority
            if len(source.description) > len(existing.description):
                existing.description = source.description
            return

        self._sources[source.site_id] = source
        self._by_category.setdefault(source.category, []).append(source)
        self._by_region.setdefault(source.region, []).append(source)

    def all(self) -> list[Source]:
        return list(self._sources.values())

    def count(self) -> int:
        return len(self._sources)

    def get(self, site_id: str) -> Optional[Source]:
        return self._sources.get(site_id)

    def by_category(self, category: SourceCategory) -> list[Source]:
        return self._by_category.get(category, [])

    def by_region(self, region: str) -> list[Source]:
        return self._by_region.get(region, [])

    def query(
        self,
        category: Optional[SourceCategory] = None,
        region: Optional[str] = None,
        priority_min: str = "low",
        has_dedicated_only: bool = False,
        cad_models_only: bool = False,
    ) -> list[Source]:
        """Filter sources by criteria."""
        priority_rank = {"low": 0, "medium": 1, "high": 2}
        min_rank = priority_rank.get(priority_min, 0)

        results = []
        for s in self._sources.values():
            if category and s.category != category:
                continue
            if region and s.region != region:
                continue
            if priority_rank.get(s.priority, 1) < min_rank:
                continue
            if has_dedicated_only and not s.has_dedicated_adapter:
                continue
            if cad_models_only and not s.cad_models_present:
                continue
            results.append(s)

        # Sort: high priority + has adapter first
        results.sort(
            key=lambda s: (
                priority_rank.get(s.priority, 1),
                int(s.has_dedicated_adapter),
                s.reliability_score,
            ),
            reverse=True,
        )
        return results

    def stats(self) -> dict:
        """Summary of loaded registry."""
        with_adapter = sum(1 for s in self._sources.values() if s.has_dedicated_adapter)
        by_method = {}
        for s in self._sources.values():
            by_method[s.access_method.value] = by_method.get(s.access_method.value, 0) + 1
        return {
            "total": len(self._sources),
            "with_dedicated_adapter": with_adapter,
            "by_access_method": by_method,
            "categories": {c.value: len(self._by_category.get(c, [])) for c in SourceCategory},
            "regions": {r: len(srcs) for r, srcs in self._by_region.items()},
        }


# ─── Singleton ───────────────────────────────────────────────
_registry_singleton: Optional[SourceRegistry] = None


def get_registry(reload: bool = False) -> SourceRegistry:
    """Lazy-load global registry."""
    global _registry_singleton
    if _registry_singleton is not None and not reload:
        return _registry_singleton

    registry = SourceRegistry()

    # Load from research JSONL
    workspace_root = Path(__file__).resolve().parents[3]
    research_dir = workspace_root / "knowledge_base" / "05_research" / "2026-06-02_qwen_deep_research_outputs"
    if research_dir.exists():
        registry.load_from_jsonl_dir(research_dir)

    # Also load any baseline registry file
    baseline = workspace_root / "knowledge_base" / "05_research" / "sites_registry.jsonl"
    if baseline.exists():
        registry.load_jsonl_file(baseline)

    _registry_singleton = registry
    return registry
