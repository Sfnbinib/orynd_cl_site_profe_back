"""Manual mesh decomposition — hybrid workflow (HYBRID_WORKFLOW_PHILOSOPHY).

Founder: *"Manual workflow с micro-agent прокладками"*. Юзер сам помечает
regions на mesh, AI подсказывает primitive type. Это **manual + hints** mode.

Flow:
    1. Юзер выделил bounding box / face / point cluster в 3D viewport
    2. Frontend posts the selection → /mesh/manual/suggest_primitive
    3. Backend проverbs convexity / linearity / curvature → returns
       suggested primitive type + confidence
    4. Юзер confirms или меняет → /mesh/manual/assemble — собирает CoreOps
       из manually-labeled regions
    5. CAD bridge строит STEP (как обычно)

Suggestions = "hints" уровня. Юзер всегда последнее слово.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

PrimitiveType = Literal["box", "cylinder", "sphere", "plane", "cone", "torus", "unknown"]


@dataclass
class RegionSelection:
    """One region selected by the user in the 3D viewport.

    bbox_min / bbox_max — world-space AABB. Optional face_ids / vertex_ids
    for finer selection (face-click vs bbox-drag).
    """

    region_id: str
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    face_ids: list[int] = field(default_factory=list)
    vertex_ids: list[int] = field(default_factory=list)
    user_hint: Optional[PrimitiveType] = None  # user pre-selected type


@dataclass
class PrimitiveSuggestion:
    """AI hint for what primitive this region likely is."""

    primitive_type: PrimitiveType
    confidence: float  # 0..1
    parameters: dict  # type-specific (radius for cylinder, etc.)
    rationale: str  # human-readable why


def suggest_primitive_for_region(
    region: RegionSelection,
    mesh_path: Optional[str] = None,
) -> PrimitiveSuggestion:
    """Hybrid hint: takes user selection + (optionally) the mesh, returns
    most likely primitive type.

    Algorithm (Phase 0 — heuristic, replace with AI Model 4 Pass 2 fitter later):
      1. If user provided ``user_hint`` — return that with confidence 1.0
         ("manual override always wins").
      2. Otherwise infer from bbox aspect ratio:
         - cube-ish (all sides ~equal) → box
         - one long axis → cylinder (most likely)
         - flat → plane
         - sphere-ish → sphere
      3. Confidence is low (0.5-0.7) — это hint, не decision. Юзер
         корректирует если ошиблись.
    """
    if region.user_hint and region.user_hint != "unknown":
        return PrimitiveSuggestion(
            primitive_type=region.user_hint,
            confidence=1.0,
            parameters={},
            rationale="user-specified",
        )

    dx = region.bbox_max[0] - region.bbox_min[0]
    dy = region.bbox_max[1] - region.bbox_min[1]
    dz = region.bbox_max[2] - region.bbox_min[2]
    sizes = sorted([dx, dy, dz])
    smallest, middle, largest = sizes

    if largest <= 0:
        return PrimitiveSuggestion(
            primitive_type="unknown",
            confidence=0.0,
            parameters={},
            rationale="degenerate bbox (zero size)",
        )

    # All sides ~equal → cube-ish
    if smallest / largest > 0.75:
        return PrimitiveSuggestion(
            primitive_type="box",
            confidence=0.65,
            parameters={"sx": dx, "sy": dy, "sz": dz},
            rationale=f"bbox ~cube ({dx:.1f}×{dy:.1f}×{dz:.1f}mm)",
        )

    # Flat (one tiny dim) → plane
    if smallest / largest < 0.1:
        return PrimitiveSuggestion(
            primitive_type="plane",
            confidence=0.7,
            parameters={"width": middle, "height": largest},
            rationale=f"flat region ({smallest:.2f}mm thin)",
        )

    # One long axis + ~equal other two → cylinder
    if middle / largest < 0.5 and smallest / middle > 0.85:
        radius = (smallest + middle) / 4
        return PrimitiveSuggestion(
            primitive_type="cylinder",
            confidence=0.6,
            parameters={"radius": radius, "height": largest},
            rationale=f"long axis ({largest:.1f}mm) + circular cross-section",
        )

    # Default — box with low confidence
    return PrimitiveSuggestion(
        primitive_type="box",
        confidence=0.4,
        parameters={"sx": dx, "sy": dy, "sz": dz},
        rationale=f"non-obvious shape — fallback box ({dx:.1f}×{dy:.1f}×{dz:.1f}mm)",
    )


def assemble_coreops_from_manual_regions(
    regions_with_primitives: list[tuple[RegionSelection, PrimitiveSuggestion]],
) -> dict:
    """Build a CoreOps document from manually-labelled regions.

    Output mirrors the structure of AI Model 4's Pass 2 output so the
    existing CADAgent bridge can consume it without changes.
    """
    operations = []
    for idx, (region, suggestion) in enumerate(regions_with_primitives):
        if suggestion.primitive_type == "unknown":
            continue
        operations.append(
            {
                "op_id": f"manual_{idx}",
                "type": suggestion.primitive_type,
                "parameters": dict(suggestion.parameters),
                "origin": [
                    (region.bbox_min[0] + region.bbox_max[0]) / 2,
                    (region.bbox_min[1] + region.bbox_max[1]) / 2,
                    (region.bbox_min[2] + region.bbox_max[2]) / 2,
                ],
                "source": "manual",
                "user_confidence": suggestion.confidence,
                "rationale": suggestion.rationale,
            }
        )

    summary: dict[str, int] = {}
    for op in operations:
        ptype = op["type"]
        summary[ptype] = summary.get(ptype, 0) + 1

    return {
        "operations": operations,
        "summary": summary,
        "total": len(operations),
        "source": "manual_assembly",
        "meta": {
            "regions_count": len(regions_with_primitives),
            "skipped_unknown": sum(
                1 for _, s in regions_with_primitives if s.primitive_type == "unknown"
            ),
        },
    }


__all__ = [
    "PrimitiveType",
    "RegionSelection",
    "PrimitiveSuggestion",
    "suggest_primitive_for_region",
    "assemble_coreops_from_manual_regions",
]
