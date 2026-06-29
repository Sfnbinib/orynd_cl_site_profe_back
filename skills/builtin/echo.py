"""Trivial echo skill — useful for smoke tests and invoker validation."""

from __future__ import annotations

from typing import Any

from orynd_core.skills.base import Skill, SkillSignature


class EchoSkill(Skill):
    slug = "echo"
    name = "Echo"
    description = "Return the inputs verbatim. Smoke-test the skill harness."
    signature = SkillSignature(
        inputs={"message": "str"},
        outputs={"echo": "str", "length": "int"},
        instructions="Return the message and its length.",
    )
    version = "0.1.0"

    async def invoke(self, message: str = "", **_: Any) -> dict[str, Any]:
        return {"echo": message, "length": len(message)}
