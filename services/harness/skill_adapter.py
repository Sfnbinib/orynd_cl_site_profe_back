"""Adapt every registered Skill into a Capability.

Called once at harness bootstrap. Each skill becomes a capability with id
``skill.<slug>`` and category ``skill`` so it appears in the dynamic
harness alongside built-in tools / MCP tools / HTTP integrations.
"""

from __future__ import annotations

from typing import Any

from orynd_core.services.harness.capabilities import Capability, CapabilityRegistry
from orynd_core.services.logging import get_logger
from orynd_core.skills.invoker import invoke_skill
from orynd_core.skills.registry import get_registry as get_skill_registry

log = get_logger("orynd.harness.skill_adapter")


def _make_handler(slug: str):
    async def _handler(**args: Any) -> Any:
        return await invoke_skill(slug, args)

    return _handler


def register_skill_capabilities(registry: CapabilityRegistry) -> int:
    """Walk skill registry, register each as a Capability. Returns added count."""
    added = 0
    for skill_cls in get_skill_registry().list_all():
        try:
            registry.register(
                Capability(
                    id=f"skill.{skill_cls.slug}",
                    name=skill_cls.name,
                    description=skill_cls.description,
                    category="skill",
                    input_schema=skill_cls.signature.inputs,
                    output_schema=skill_cls.signature.outputs,
                    handler=_make_handler(skill_cls.slug),
                    source=f"skill:{skill_cls.slug}",
                    tags=["skill"] + list(skill_cls.tools),
                )
            )
            added += 1
        except Exception as exc:
            log.warning(
                "harness.skill_adapter_failed",
                skill=skill_cls.slug,
                error=str(exc),
            )
    log.info("harness.skills_registered", count=added)
    return added


__all__ = ["register_skill_capabilities"]
