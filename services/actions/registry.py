"""Shared action registry.

This is intentionally small: it gives chat, MCP, command palette, and future
agents one vocabulary for core ORYND actions without forcing a large refactor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ActionSpec:
    name: str
    intent: str
    aliases: tuple[str, ...] = ()
    input_schema: dict[str, Any] = field(default_factory=dict)
    mode: str = "append"
    emits: tuple[str, ...] = ()
    executor: Callable[..., Any] | None = None


_ACTIONS: dict[str, ActionSpec] = {}


def register(spec: ActionSpec) -> ActionSpec:
    _ACTIONS[spec.name] = spec
    return spec


def get_action(name: str) -> ActionSpec | None:
    return _ACTIONS.get(name)


def list_actions() -> list[ActionSpec]:
    return list(_ACTIONS.values())


def find_by_alias(alias: str) -> ActionSpec | None:
    needle = alias.strip().lower()
    for spec in _ACTIONS.values():
        if needle == spec.name.lower() or needle in {a.lower() for a in spec.aliases}:
            return spec
    return None


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


register(ActionSpec(
    name="create_box",
    intent="create_object",
    aliases=("box", "cube", "куб", "коробка"),
    input_schema=_schema({"sx": {"type": "number"}, "sy": {"type": "number"}, "sz": {"type": "number"}}),
    mode="replace_or_append",
    emits=("workspace.object.created", "model.ready"),
))
register(ActionSpec(
    name="create_cylinder",
    intent="create_object",
    aliases=("cylinder", "цилиндр", "вал"),
    input_schema=_schema({"radius": {"type": "number"}, "height": {"type": "number"}}),
    mode="replace_or_append",
    emits=("workspace.object.created", "model.ready"),
))
register(ActionSpec(
    name="create_spur_gear",
    intent="create_object",
    aliases=("gear", "spur gear", "шестерня", "шестеренка", "шестерёнка", "зубчатое колесо"),
    input_schema=_schema({
        "teeth": {"type": "integer"},
        "module": {"type": "number"},
        "thickness": {"type": "number"},
        "bore": {"type": "number"},
    }),
    mode="replace_or_append",
    emits=("workspace.object.created", "model.ready"),
))
register(ActionSpec(
    name="create_brake_disc",
    intent="create_object",
    aliases=("brake disc", "brake rotor", "тормозной диск"),
    input_schema=_schema({
        "diameter": {"type": "number"},
        "thickness": {"type": "number"},
        "bolts": {"type": "integer"},
    }),
    mode="replace_or_append",
    emits=("workspace.object.created", "model.ready"),
))
register(ActionSpec(
    name="gear_mesh",
    intent="connect_objects",
    aliases=("mesh gears", "gear mesh", "соедини по зубьям", "сцепи шестерни"),
    input_schema=_schema({"object_a": {"type": "string"}, "object_b": {"type": "string"}}),
    mode="modify",
    emits=("workspace.constraint.created", "workspace.document.updated", "model.ready"),
))
register(ActionSpec(
    name="align_axis",
    intent="connect_objects",
    aliases=("coaxial", "соосно", "по оси"),
    input_schema=_schema({"object_a": {"type": "string"}, "object_b": {"type": "string"}}),
    mode="modify",
    emits=("workspace.constraint.created", "workspace.document.updated", "model.ready"),
))
register(ActionSpec(
    name="search_models",
    intent="search_external_model",
    aliases=("find", "search", "найди", "поиск"),
    mode="query",
    emits=("candidates",),
))
register(ActionSpec(
    name="deep_research",
    intent="research_topic",
    aliases=("deep research", "исследуй", "ресерч"),
    mode="query",
    emits=("research.ready", "research.empty"),
))
