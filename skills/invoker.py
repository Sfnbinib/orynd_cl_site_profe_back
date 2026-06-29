"""Skill invocation surface.

Handles:
* Lookup → instantiate → call ``invoke(**args)``
* Langfuse trace (no-op when key missing)
* Prometheus skill-invocation counter
* Event-bus signal so other agents can react

Raises :class:`SkillNotFoundError` (404) or :class:`SkillExecutionError` (500)
which the global FastAPI handler converts to the standard error envelope.
"""

from __future__ import annotations

import time
from typing import Any

from orynd_core.errors import SkillExecutionError
from orynd_core.services.event_bus import bus
from orynd_core.services.logging import get_logger
from orynd_core.services.observability.langfuse_client import traced_llm_call
from orynd_core.services.observability.metrics import (
    library_skill_invocations_total,
)
from orynd_core.skills.registry import get_registry

log = get_logger("orynd.skills.invoker")


async def invoke_skill(slug: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = get_registry()
    skill_cls = registry.get(slug)  # raises SkillNotFoundError
    instance = skill_cls()
    args = args or {}

    started = time.time()
    library_skill_invocations_total.labels(slug=slug).inc()
    await bus.publish("skill.invoked", {"slug": slug, "args_keys": list(args.keys())})

    try:
        async with traced_llm_call(f"skill.{slug}", inputs=list(args.keys())):
            result = await instance.invoke(**args)
    except Exception as exc:
        log.error("skill.invocation_failed", slug=slug, exc_info=True)
        await bus.publish(
            "skill.failed",
            {"slug": slug, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise SkillExecutionError(
            f"Skill '{slug}' failed",
            details={"slug": slug, "error": str(exc)},
        ) from exc

    duration_ms = int((time.time() - started) * 1000)
    log.info("skill.invoked.complete", slug=slug, duration_ms=duration_ms)
    await bus.publish(
        "skill.complete",
        {"slug": slug, "duration_ms": duration_ms, "ok": True},
    )

    if not isinstance(result, dict):
        # Be strict: skills must return dict so the router can serialise.
        return {"result": result}
    return result


__all__ = ["invoke_skill"]
