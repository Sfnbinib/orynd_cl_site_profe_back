"""/skills/* — built-in + installed skill discovery and invocation."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body

from orynd_core.skills.invoker import invoke_skill
from orynd_core.skills.registry import get_registry

router = APIRouter(prefix="/skills", tags=["skills"])


@router.get("")
async def list_skills() -> list[dict[str, Any]]:
    return [skill.manifest() for skill in get_registry().list_all()]


@router.get("/{slug}")
async def get_skill_manifest(slug: str) -> dict[str, Any]:
    skill = get_registry().get(slug)  # raises SkillNotFoundError → 404
    return skill.manifest()


@router.post("/{slug}/invoke")
async def invoke(slug: str, args: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    return await invoke_skill(slug, args)
