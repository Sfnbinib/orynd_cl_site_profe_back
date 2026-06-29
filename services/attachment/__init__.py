"""Axes attachment — select an axis, get catalog component suggestions.

Founder priority feature (blueprint 99). Hybrid manual UX:
  user clicks axis → matcher hints (⌘1..⌘9) → pick → adaptive fit → CoreOps.
"""

from orynd_core.services.attachment.catalog import (
    CatalogPart,
    all_parts,
    get_part,
    parts_by_category,
)
from orynd_core.services.attachment.matcher import (
    Axis,
    MatchCandidate,
    match_axis,
)

__all__ = [
    "CatalogPart",
    "all_parts",
    "get_part",
    "parts_by_category",
    "Axis",
    "MatchCandidate",
    "match_axis",
]
