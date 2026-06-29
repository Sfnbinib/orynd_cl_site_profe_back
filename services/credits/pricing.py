"""Credit pricing for ORYND actions.

Single source of truth for "how many credits does X cost". UI calls
`/api/credits/quote` to show cost before action; backend calls
`/api/credits/commit` to atomically charge after action succeeds.

Prices are deterministic given the input params (no LLM in the loop).
That lets the UI confirmation be honest: what we quote is what we charge.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class PricingError(Exception):
    pass


@dataclass(frozen=True)
class ActionPrice:
    """Base + per-unit pricing for a single action type."""

    action: str
    base: int                          # flat cost for invoking
    unit_label: str | None = None      # what 'units' means (e.g. 'MB', 'phase', 'operation')
    per_unit: int = 0                  # cost per unit on top of base
    description: str = ""


@dataclass(frozen=True)
class PriceQuote:
    action: str
    cost: int
    breakdown: dict[str, Any]


# ---------------------------------------------------------------------------
# Pricing table — tweak via ENV without redeploy if needed.
# ---------------------------------------------------------------------------
def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _table() -> dict[str, ActionPrice]:
    """Built fresh each call so ENV overrides apply without reload."""
    return {
        "mesh_analyze": ActionPrice(
            action="mesh_analyze",
            base=_env_int("PRICE_MESH_ANALYZE_BASE", 10),
            unit_label="MB",
            per_unit=_env_int("PRICE_MESH_ANALYZE_PER_MB", 1),
            description="Analyze mesh file (STL/OBJ/PLY/3MF) — features, regions, geometry summary.",
        ),
        "vision_analyze": ActionPrice(
            action="vision_analyze",
            base=_env_int("PRICE_VISION_ANALYZE", 8),
            description="Analyze image of a part — extract geometry and dimensions.",
        ),
        "cad_execute": ActionPrice(
            action="cad_execute",
            base=_env_int("PRICE_CAD_EXECUTE_BASE", 2),
            unit_label="operation",
            per_unit=_env_int("PRICE_CAD_EXECUTE_PER_OP", 1),
            description="Execute a CoreOps program (extrude / cut / fillet / etc.) via CADAgent.",
        ),
        "deep_search": ActionPrice(
            action="deep_search",
            base=_env_int("PRICE_DEEP_SEARCH", 5),
            description="Quick search across Printables / Thingiverse / GitHub / web.",
        ),
        "deep_research": ActionPrice(
            action="deep_research",
            base=_env_int("PRICE_DEEP_RESEARCH_BASE", 30),
            unit_label="phase",
            per_unit=_env_int("PRICE_DEEP_RESEARCH_PER_PHASE", 5),
            description="Multi-phase DeepResearchAgent — synthesizes findings across sources.",
        ),
        "fabricate": ActionPrice(
            action="fabricate",
            base=_env_int("PRICE_FABRICATE", 3),
            description="Recommend fabrication method + material + parameters.",
        ),
        "slice": ActionPrice(
            action="slice",
            base=_env_int("PRICE_SLICE", 2),
            description="Generate G-code from STL via PrusaSlicer CLI.",
        ),
        "review_design": ActionPrice(
            action="review_design",
            base=_env_int("PRICE_REVIEW_DESIGN", 15),
            description="Workflow OS review — printability / geometry / cost / confidence.",
        ),
        "chat": ActionPrice(
            action="chat",
            base=_env_int("PRICE_CHAT_MIN", 1),
            unit_label="1k tokens",
            per_unit=_env_int("PRICE_CHAT_PER_KTOK", 1),
            description="WorkspaceAgent chat — token-based pricing.",
        ),
    }


def list_actions() -> list[ActionPrice]:
    return list(_table().values())


def quote_action(action: str, params: dict[str, Any] | None = None) -> PriceQuote:
    """Compute a deterministic quote for an action + its parameters.

    Recognized params:
      - file_size_mb (mesh_analyze) — round up
      - operation_count (cad_execute)
      - phase_count (deep_research) — defaults to 5 (matches current pipeline)
      - estimated_input_tokens, estimated_output_tokens (chat) — combined / 1000
    Unrecognized params are ignored but echoed in breakdown for transparency.
    """
    table = _table()
    price = table.get(action)
    if price is None:
        raise PricingError(f"unknown action: {action}")

    params = dict(params or {})
    units = 0

    if action == "mesh_analyze":
        size_mb = float(params.get("file_size_mb", 0) or 0)
        units = max(0, int(size_mb + 0.999))  # round up
    elif action == "cad_execute":
        units = max(0, int(params.get("operation_count", 1) or 1) - 1)  # 1 op is base
    elif action == "deep_research":
        units = max(0, int(params.get("phase_count", 5) or 5) - 1)  # 1 phase is base
    elif action == "chat":
        toks = int(params.get("estimated_input_tokens", 0) or 0) + int(
            params.get("estimated_output_tokens", 0) or 0
        )
        units = max(0, (toks + 999) // 1000)

    extra = units * price.per_unit
    cost = price.base + extra

    breakdown: dict[str, Any] = {
        "base": price.base,
        "units": units,
        "unit_label": price.unit_label,
        "per_unit": price.per_unit,
        "extra": extra,
        "echoed_params": params,
    }
    return PriceQuote(action=action, cost=cost, breakdown=breakdown)
