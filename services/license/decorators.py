"""FastAPI dependencies that gate endpoints by license tier.

Usage::

    @router.post("/library/articles", dependencies=[Depends(requires_tier("library.publish_open"))])
    async def publish_article(article: Article):
        ...

Or for ad-hoc checks inside an endpoint::

    state = check_feature("research.deep")  # raises ForbiddenError if denied
"""

from __future__ import annotations

from typing import Callable

from fastapi import Depends

from orynd_core.errors import ForbiddenError
from orynd_core.services.license.state import LicenseState, get_license_state
from orynd_core.services.license.tiers import Tier, tier_for_feature, tier_includes


def check_feature(feature_id: str) -> LicenseState:
    """Synchronous check used inside endpoint bodies. Raises on denial."""
    state = get_license_state()
    if state.is_locked:
        raise ForbiddenError(
            "license expired beyond grace period",
            details={"feature_id": feature_id, "grace_remaining_s": 0},
        )
    needed = tier_for_feature(feature_id)
    if not tier_includes(state.tier, needed):
        raise ForbiddenError(
            f"feature {feature_id!r} requires tier {needed.value}",
            details={
                "feature_id": feature_id,
                "required_tier": needed.value,
                "current_tier": state.tier.value,
            },
        )
    return state


def requires_feature(feature_id: str) -> Callable[[], LicenseState]:
    """FastAPI dep factory — pass to ``Depends(requires_feature("..."))``."""

    def _dep() -> LicenseState:
        return check_feature(feature_id)

    return _dep


def requires_tier(min_tier: Tier) -> Callable[[], LicenseState]:
    """FastAPI dep factory — pass to ``Depends(requires_tier(Tier.PRO))``."""

    def _dep() -> LicenseState:
        state = get_license_state()
        if state.is_locked:
            raise ForbiddenError(
                "license expired beyond grace period",
                details={"required_tier": min_tier.value, "grace_remaining_s": 0},
            )
        if not tier_includes(state.tier, min_tier):
            raise ForbiddenError(
                f"this action requires tier {min_tier.value}",
                details={
                    "required_tier": min_tier.value,
                    "current_tier": state.tier.value,
                },
            )
        return state

    return _dep


__all__ = ["check_feature", "requires_feature", "requires_tier", "Depends"]
