"""Axis matcher — given a user-selected axis, rank catalog candidates.

This is the **hint engine** behind the hybrid manual UX (HYBRID_WORKFLOW
§ axes attachment). Юзер кликнул ось → matcher возвращает top-N кандидатов
+ оценки → UI показывает их как ⌘1..⌘9 hints в right-click menu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from orynd_core.services.attachment.catalog import CatalogPart, all_parts


@dataclass
class Axis:
    """A selected axis in world space.

    origin — point on the axis (e.g. centre of a hole / shaft end).
    direction — unit-ish vector (will be normalised).
    diameter — measured / estimated axis diameter (mm). Optional — if None,
               matcher returns all parts (diameter unknown).
    length — axis length (mm). Optional.
    """

    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    diameter: Optional[float] = None
    length: Optional[float] = None

    def normalised_direction(self) -> tuple[float, float, float]:
        dx, dy, dz = self.direction
        mag = math.sqrt(dx * dx + dy * dy + dz * dz)
        if mag == 0:
            return (0.0, 0.0, 1.0)
        return (dx / mag, dy / mag, dz / mag)


@dataclass
class MatchCandidate:
    part: CatalogPart
    score: float  # 0..1
    fit_reason: str

    def to_dict(self) -> dict:
        return {
            "part_id": self.part.part_id,
            "name": self.part.name,
            "category": self.part.category,
            "score": round(self.score, 3),
            "fit_reason": self.fit_reason,
            "primitive_type": self.part.primitive_type,
            "default_parameters": dict(self.part.default_parameters),
            "tags": list(self.part.tags),
        }


def match_axis(
    axis: Axis,
    *,
    intent: Optional[str] = None,
    category: Optional[str] = None,
    k: int = 9,
) -> list[MatchCandidate]:
    """Return top-k catalog parts for the axis.

    Scoring:
      - diameter fit (центр диапазона = лучший score)  → 0..0.5
      - intent keyword match в name/tags               → +0.3
      - category match                                  → +0.2
    Parts that fail the hard diameter constraint are excluded.
    k defaults to 9 so they map to ⌘1..⌘9 hint slots.
    """
    intent_l = (intent or "").lower().strip()
    candidates: list[MatchCandidate] = []

    for part in all_parts():
        if category and part.category != category:
            continue

        # Hard constraint: diameter must fit (if known)
        if axis.diameter is not None:
            length = axis.length if axis.length is not None else (
                (part.axis_length_min + part.axis_length_max) / 2
            )
            if not part.fits_axis(axis.diameter, length):
                continue

        score = 0.0
        reasons: list[str] = []

        # Diameter centre-of-range bonus
        if axis.diameter is not None:
            mid = (part.axis_diameter_min + part.axis_diameter_max) / 2
            span = max(part.axis_diameter_max - part.axis_diameter_min, 0.01)
            closeness = 1.0 - min(abs(axis.diameter - mid) / span, 1.0)
            score += 0.5 * closeness
            reasons.append(f"Ø{axis.diameter:.1f}mm fit")
        else:
            score += 0.2  # neutral when diameter unknown

        # Intent keyword
        if intent_l:
            hay = f"{part.name} {part.description} {' '.join(part.tags)}".lower()
            if any(tok in hay for tok in intent_l.split()):
                score += 0.3
                reasons.append(f"matches '{intent}'")

        # Category exact
        if category and part.category == category:
            score += 0.2
            reasons.append(f"category {category}")

        candidates.append(
            MatchCandidate(part=part, score=min(score, 1.0), fit_reason=", ".join(reasons) or "generic fit")
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:k]


__all__ = ["Axis", "MatchCandidate", "match_axis"]
