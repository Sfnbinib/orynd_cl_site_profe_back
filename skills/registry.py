"""Skill registry.

* Auto-discovers built-in skills by walking ``orynd_core.skills.builtin``.
* Process-wide singleton via :func:`get_registry`.
* Test-friendly :func:`reset_registry` clears state between cases.
"""

from __future__ import annotations

import importlib
import pkgutil
import threading
from typing import Optional, Type

from orynd_core.errors import SkillNotFoundError
from orynd_core.services.logging import get_logger
from orynd_core.skills.base import Skill

log = get_logger("orynd.skills.registry")


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Type[Skill]] = {}

    def load_builtin(self) -> int:
        """Walk ``orynd_core.skills.builtin`` and register every Skill subclass."""
        import orynd_core.skills.builtin as builtin_pkg

        added = 0
        for _, modname, _ in pkgutil.iter_modules(builtin_pkg.__path__):
            try:
                module = importlib.import_module(
                    f"orynd_core.skills.builtin.{modname}"
                )
            except Exception as exc:
                log.warning(
                    "skills.builtin_import_failed",
                    module=modname,
                    error=str(exc),
                )
                continue
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Skill)
                    and attr is not Skill
                    and getattr(attr, "slug", None)
                ):
                    self.register(attr)
                    added += 1
        return added

    def register(self, skill_cls: Type[Skill]) -> None:
        if not getattr(skill_cls, "slug", None):
            raise ValueError(f"{skill_cls.__name__}: missing slug")
        self._skills[skill_cls.slug] = skill_cls

    def unregister(self, slug: str) -> None:
        self._skills.pop(slug, None)

    def get(self, slug: str) -> Type[Skill]:
        skill = self._skills.get(slug)
        if not skill:
            raise SkillNotFoundError(details={"slug": slug})
        return skill

    def find(self, slug: str) -> Optional[Type[Skill]]:
        return self._skills.get(slug)

    def list_all(self) -> list[Type[Skill]]:
        return list(self._skills.values())

    def clear(self) -> None:
        self._skills.clear()


_lock = threading.Lock()
_registry: Optional[SkillRegistry] = None


def get_registry() -> SkillRegistry:
    global _registry
    if _registry is not None:
        return _registry
    with _lock:
        if _registry is not None:
            return _registry
        _registry = SkillRegistry()
        try:
            count = _registry.load_builtin()
            log.info("skills.builtin_loaded", count=count)
        except Exception as exc:
            log.warning("skills.builtin_load_failed", error=str(exc))
        return _registry


def reset_registry() -> None:
    """Test hook — drop the singleton."""
    global _registry
    with _lock:
        _registry = None


__all__ = ["SkillRegistry", "get_registry", "reset_registry"]
