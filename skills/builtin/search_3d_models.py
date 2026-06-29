"""Search 3D model platforms via the Source Access Orchestrator."""

from __future__ import annotations

from typing import Any, Optional

from orynd_core.services.source_orchestrator import (
    SourceAccessOrchestrator,
    SourceCategory,
    get_registry,
)
from orynd_core.skills.base import Skill, SkillSignature

_orch: SourceAccessOrchestrator | None = None


def _get_orchestrator() -> SourceAccessOrchestrator:
    global _orch
    if _orch is None:
        _orch = SourceAccessOrchestrator(
            registry=get_registry(),
            enable_browser=False,
            enable_ddg_fallback=True,
        )
    return _orch


class Search3DModelsSkill(Skill):
    slug = "search_3d_models"
    name = "Search 3D Models"
    description = (
        "Search Printables / Thingiverse / MakerWorld and 170+ other 3D platforms "
        "in parallel. Deduplicates and ranks by reliability."
    )
    signature = SkillSignature(
        inputs={
            "query": "str — what to search for (e.g. 'mounting bracket M3')",
            "max_sources": "int — fan-out (default 5)",
            "limit_per_source": "int — hits per site (default 3)",
            "region": "str | None — 'global' / 'jp' / 'kr' / 'ru' / 'cn'",
        },
        outputs={
            "hits": "list[SearchHit]",
            "total": "int",
        },
        instructions="Search 3D model sources and return ranked hits.",
    )
    tools = ["source_orchestrator"]
    version = "0.1.0"

    async def invoke(
        self,
        query: str,
        max_sources: int = 5,
        limit_per_source: int = 3,
        region: Optional[str] = None,
    ) -> dict[str, Any]:
        orch = _get_orchestrator()
        hits = await orch.search(
            query,
            category=SourceCategory.THREE_D_MODELS,
            region=region,
            max_sources=max_sources,
            limit_per_source=limit_per_source,
        )
        return {
            "hits": [h.to_dict() for h in hits],
            "total": len(hits),
        }
