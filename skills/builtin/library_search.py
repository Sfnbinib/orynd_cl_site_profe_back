"""Search the Knowledge Library.

Thin wrapper around the library storage backend so any agent / chat slash
command can search articles without touching the storage layer directly.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from orynd_core.services.library.storage_factory import get_storage_backend
from orynd_core.skills.base import Skill, SkillSignature


class LibrarySearchSkill(Skill):
    slug = "library_search"
    name = "Library Search"
    description = "Search Knowledge Library articles by keyword (FTS or semantic)."
    signature = SkillSignature(
        inputs={
            "query": "str — search query",
            "k": "int — number of results (default 5)",
            "mode": "str — 'fts' or 'semantic' (default 'fts')",
            "topic_id": "str | None — restrict to one topic",
        },
        outputs={
            "results": "list[{article: Article, score: float}]",
            "total": "int",
        },
        instructions="Search the Knowledge Library and return top-k matches.",
    )
    tools = ["library_storage"]
    version = "0.1.0"

    async def invoke(
        self,
        query: str,
        k: int = 5,
        mode: str = "fts",
        topic_id: Optional[str] = None,
    ) -> dict[str, Any]:
        backend = get_storage_backend()
        tid = UUID(topic_id) if topic_id else None
        if mode == "semantic":
            results = await backend.search_articles_semantic(query, k=k, topic_id=tid)
        else:
            results = await backend.search_articles_fts(query, k=k, topic_id=tid)
        return {
            "results": [r.model_dump(mode="json") for r in results],
            "total": len(results),
        }
