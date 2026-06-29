"""License tiers + feature gating.

Tier ladder:
    DEMO  — anonymous / first-launch state. Limited but everything works.
    FREE  — registered, cloud-only, capped weekly research.
    PRO   — Claude Code-style 5h sessions, includes offline mode + local models.
    MAX   — Pro + Hyper search, unlimited.

The FEATURE_GATES dict is the **single source of truth** for what unlocks
when. Add features here, then guard endpoints with the dependency in
``decorators.py``.
"""

from __future__ import annotations

from enum import Enum


class Tier(str, Enum):
    DEMO = "demo"
    FREE = "free"
    PRO = "pro"
    MAX = "max"


# Numeric rank used for tier_includes() comparisons.
_RANK: dict[Tier, int] = {Tier.DEMO: 0, Tier.FREE: 1, Tier.PRO: 2, Tier.MAX: 3}


# ``feature_id`` → minimum tier required. Add features here; guard endpoints
# via ``requires_tier(feature_id)`` in ``decorators.py``.
FEATURE_GATES: dict[str, Tier] = {
    # Core (always available)
    "library.read_open": Tier.DEMO,
    "library.write_personal": Tier.DEMO,
    "mesh.decompose": Tier.DEMO,
    "skills.invoke_builtin": Tier.DEMO,
    "harness.plan": Tier.DEMO,
    "harness.execute_local": Tier.DEMO,
    # Cloud-paid features
    "research.light": Tier.FREE,  # 5 per week limit enforced separately
    "research.deep": Tier.PRO,
    "research.hyper": Tier.MAX,
    "library.publish_open": Tier.FREE,
    "skills.dspy_optimize": Tier.PRO,
    "modes.bypass": Tier.PRO,
    "modes.offline": Tier.PRO,
    "skills.install_community": Tier.FREE,
    "skills.publish_community": Tier.PRO,
    "anthropic.proxy": Tier.FREE,  # Even free uses proxy; quota separate
    "anthropic.direct_byok": Tier.PRO,  # Bring-your-own-key only for Pro
    "ai_model_4.cad_build": Tier.DEMO,  # Demo period — all open
    "quality_rater.use": Tier.FREE,
    "sources.browser_harness": Tier.PRO,  # Chromium-heavy
}


def tier_includes(have: Tier, need: Tier) -> bool:
    """``have`` is at least as high as ``need`` on the ladder."""
    return _RANK.get(have, 0) >= _RANK.get(need, 0)


def tier_for_feature(feature_id: str) -> Tier:
    """Minimum tier needed to use ``feature_id``. Unknown → MAX (deny)."""
    return FEATURE_GATES.get(feature_id, Tier.MAX)


__all__ = ["Tier", "FEATURE_GATES", "tier_includes", "tier_for_feature"]
