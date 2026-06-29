"""Capability registry — composable units the harness can invoke.

A ``Capability`` is the harness-facing description of *anything* an agent
can call: a built-in tool, a skill, an MCP tool, an HTTP integration. The
``handler`` field is the async callable; everything else is metadata used
by the planner and the UI.

Process-wide registry. Skills register themselves via :func:`load_skill_adapters`
which is called on first :func:`get_capability_registry` access.
"""

from __future__ import annotations

import threading
from typing import Any, Awaitable, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from orynd_core.services.logging import get_logger

log = get_logger("orynd.harness.capabilities")

CapabilityHandler = Callable[..., Awaitable[Any]]


class Capability(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    name: str
    description: str = ""
    category: Literal["tool", "mcp", "skill", "api", "local"] = "tool"

    input_schema: dict[str, str] = Field(default_factory=dict)
    output_schema: dict[str, str] = Field(default_factory=dict)

    handler: CapabilityHandler

    estimated_cost_tokens: int = 0
    requires_permission: bool = False
    permission_category: Literal["low", "medium", "high", "critical"] = "low"

    source: str = "builtin"
    tags: list[str] = Field(default_factory=list)

    def public_manifest(self) -> dict[str, Any]:
        """Serialisable view — drops the handler callable."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "estimated_cost_tokens": self.estimated_cost_tokens,
            "requires_permission": self.requires_permission,
            "permission_category": self.permission_category,
            "source": self.source,
            "tags": list(self.tags),
        }


class CapabilityRegistry:
    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}

    def register(self, cap: Capability) -> None:
        if cap.id in self._capabilities:
            log.warning("harness.capability_overwritten", id=cap.id)
        self._capabilities[cap.id] = cap

    def unregister(self, cap_id: str) -> None:
        self._capabilities.pop(cap_id, None)

    def get(self, cap_id: str) -> Optional[Capability]:
        return self._capabilities.get(cap_id)

    def list_all(self) -> list[Capability]:
        return list(self._capabilities.values())

    def list_by_category(self, category: str) -> list[Capability]:
        return [c for c in self._capabilities.values() if c.category == category]

    def search(self, query: str, k: int = 10) -> list[Capability]:
        """Heuristic tokenized substring + tag scoring.

        Real semantic search lands when embeddings are wired (Phase 11).
        """
        tokens = [t for t in query.lower().split() if t]
        if not tokens:
            return []
        scored: list[tuple[float, Capability]] = []
        for cap in self._capabilities.values():
            score = 0.0
            id_ = cap.id.lower()
            name_ = cap.name.lower()
            desc_ = cap.description.lower()
            tags_ = [t.lower() for t in cap.tags]
            for token in tokens:
                if token in id_:
                    score += 0.4
                if token in name_:
                    score += 0.3
                if token in desc_:
                    score += 0.2
                if any(token in t for t in tags_):
                    score += 0.1
            if score > 0:
                scored.append((score, cap))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [cap for _, cap in scored[:k]]

    def clear(self) -> None:
        self._capabilities.clear()


_lock = threading.Lock()
_registry: Optional[CapabilityRegistry] = None


def get_capability_registry() -> CapabilityRegistry:
    global _registry
    if _registry is not None:
        return _registry
    with _lock:
        if _registry is not None:
            return _registry
        _registry = CapabilityRegistry()
        try:
            from orynd_core.services.harness import builtin, skill_adapter

            builtin.register_builtin_capabilities(_registry)
            skill_adapter.register_skill_capabilities(_registry)
        except Exception as exc:
            log.warning("harness.capability_bootstrap_failed", error=str(exc))
        return _registry


def reset_capability_registry() -> None:
    """Test hook — drop singleton."""
    global _registry
    with _lock:
        _registry = None


__all__ = [
    "Capability",
    "CapabilityRegistry",
    "get_capability_registry",
    "reset_capability_registry",
]
