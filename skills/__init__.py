"""ORYND skill system — reusable, optimizable agent capabilities.

Public exports kept slim. Built-in skill modules are loaded lazily via
``SkillRegistry.load_builtin()`` so that importing this package doesn't
force-import every skill's heavy dependencies (AI Model 4 etc.).
"""

from orynd_core.skills.base import Skill, SkillSignature
from orynd_core.skills.invoker import invoke_skill
from orynd_core.skills.registry import SkillRegistry, get_registry

__all__ = [
    "Skill",
    "SkillSignature",
    "invoke_skill",
    "SkillRegistry",
    "get_registry",
]
