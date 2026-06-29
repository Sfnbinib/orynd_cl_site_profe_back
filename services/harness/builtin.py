"""Built-in capabilities — wrap existing internal services.

Add capabilities here when:
* The capability is shipped with ORYND (not user-installed)
* It's not already exposed as a Skill (skills get registered separately
  via skill_adapter.py)

Each capability includes an estimated_cost_tokens and permission_category
so the composition planner can budget + gate execution accordingly.
"""

from __future__ import annotations

from typing import Any

from orynd_core.services.harness.capabilities import Capability, CapabilityRegistry
from orynd_core.services.logging import get_logger

log = get_logger("orynd.harness.builtin")


# ---- built-in handlers ---------------------------------------------------


async def _ping_handler(**_: Any) -> dict[str, Any]:
    return {"pong": True}


async def _library_metrics_handler(topic_id: str, **_: Any) -> dict[str, Any]:
    from uuid import UUID
    from orynd_core.services.library.storage_factory import get_storage_backend

    backend = get_storage_backend()
    metrics = await backend.topic_metrics(UUID(topic_id))
    return metrics.model_dump(mode="json")


async def _list_categories_handler(**_: Any) -> dict[str, Any]:
    from orynd_core.services.source_orchestrator import SourceCategory

    return {"categories": [c.value for c in SourceCategory]}


# ---- registration --------------------------------------------------------


def register_builtin_capabilities(registry: CapabilityRegistry) -> int:
    """Idempotent — call once at harness bootstrap."""
    added = 0
    for cap in (
        Capability(
            id="builtin.ping",
            name="Ping",
            description="No-op harness smoke. Returns {pong: true}.",
            category="tool",
            handler=_ping_handler,
            source="builtin",
            tags=["smoke", "health"],
        ),
        Capability(
            id="builtin.library.topic_metrics",
            name="Topic Metrics",
            description="Aggregate metrics + stage promotion eligibility for a topic.",
            category="tool",
            handler=_library_metrics_handler,
            input_schema={"topic_id": "str — UUID of the topic"},
            output_schema={
                "topic_id": "str",
                "current_stage": "int",
                "article_count": "int",
                "promotion_eligible": "bool",
            },
            source="builtin",
            tags=["library", "metrics"],
        ),
        Capability(
            id="builtin.sources.list_categories",
            name="Source Categories",
            description="Enum of source categories the orchestrator knows.",
            category="tool",
            handler=_list_categories_handler,
            source="builtin",
            tags=["sources", "metadata"],
        ),
    ):
        registry.register(cap)
        added += 1
    log.info("harness.builtin_registered", count=added)
    return added


__all__ = ["register_builtin_capabilities"]
