"""SkillBase + Signature.

A Skill is a self-contained capability with:
* ``slug`` — unique identifier
* ``signature`` — DSPy-style input/output type description (strings, intentionally
  loose — strict schema enforcement happens in the invoker via Pydantic when we
  introduce typed signatures in Phase 11)
* ``invoke(**inputs)`` — async entry point, returns a dict

Phase 1 keeps skills as Python classes. Phase 11 adds JSON-on-disk skills that
the registry constructs dynamically from a manifest.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field


class SkillSignature(BaseModel):
    """DSPy-style natural-language signature."""

    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    instructions: str = ""


class Skill:
    """Base class for in-process skills.

    Subclasses set the class-level ``slug``, ``name``, ``description``,
    ``signature`` and implement ``invoke()``.
    """

    slug: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str] = ""
    signature: ClassVar[SkillSignature] = SkillSignature()
    tools: ClassVar[list[str]] = []
    version: ClassVar[str] = "0.1.0"
    topic_slug: ClassVar[Optional[str]] = None

    async def invoke(self, **inputs: Any) -> dict[str, Any]:
        raise NotImplementedError(
            f"{type(self).__name__}.invoke() must be implemented"
        )

    @classmethod
    def manifest(cls) -> dict[str, Any]:
        """Serialise just the metadata — used by ``GET /skills``."""
        return {
            "slug": cls.slug,
            "name": cls.name,
            "description": cls.description,
            "signature": cls.signature.model_dump(),
            "tools": list(cls.tools),
            "version": cls.version,
            "topic_slug": cls.topic_slug,
        }


def bump_version(current: str, level: str) -> str:
    """Founder semver — MAJOR.MINOR.PATCH; level ∈ {major, minor, patch}."""
    major, minor, patch = (int(p) for p in current.split("."))
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    if level == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unknown bump level: {level}")


__all__ = ["Skill", "SkillSignature", "bump_version"]
