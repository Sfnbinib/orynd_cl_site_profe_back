"""
Engineering Filter — between Pass 1 (rough decomposition) and Pass 2 (primitive rebuild).

Founder voice:
  "Чтобы не было каких-то суперсглаживаний, которые невозможно реализовать.
   Убираем лишние отростки. Делаем ровный инженерный элемент."

Tags each Pass 1 region as:
  - buildable: clean enough for primitive fitting
  - noise: too fragmented / non-manifold / fractal-like
  - complex: real form but not primitive-fittable (keep as mesh)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from orynd_core.services.mesh.decomposer import MeshRegion, SurfaceType

log = logging.getLogger(__name__)


class BuildabilityTag(str, Enum):
    BUILDABLE = "buildable"      # ready for primitive fit
    COMPLEX = "complex"           # real form, keep as mesh
    NOISE = "noise"               # likely artifact, drop


@dataclass
class FilteredPart:
    """A region after engineering filter classification."""
    region: MeshRegion
    tag: BuildabilityTag
    confidence: float = 0.0  # 0..1

    # Diagnostic
    buildability_score: float = 0.0
    size_sanity_ok: bool = True
    simplicity_score: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "region_id": self.region.region_id,
            "tag": self.tag.value,
            "confidence": round(self.confidence, 3),
            "buildability_score": round(self.buildability_score, 3),
            "size_sanity_ok": self.size_sanity_ok,
            "simplicity_score": round(self.simplicity_score, 3),
            "surface_type": self.region.surface_type.value,
            "area_mm2": round(self.region.area_mm2, 2),
            "notes": self.notes,
        }


class EngineeringFilter:
    """
    Classifies Pass 1 regions for Pass 2 feasibility.

    Heuristic-based (no ML). Pure functions, fast.
    Tuneable thresholds for different domains.
    """

    # Thresholds
    MIN_AREA_MM2 = 1.0                 # below this = likely noise
    MAX_ASPECT_RATIO = 100.0           # super-elongated = noise/artifact
    MIN_FACE_COUNT = 3                 # too few faces = unreliable
    MAX_NORMAL_VARIANCE_BUILDABLE = 0.15  # higher = freeform, keep as complex

    def filter(self, regions: list[MeshRegion]) -> list[FilteredPart]:
        """Tag every region."""
        results = []
        for region in regions:
            result = self._classify(region)
            results.append(result)

        # Log summary
        counts = {tag.value: 0 for tag in BuildabilityTag}
        for r in results:
            counts[r.tag.value] += 1
        log.info(f"[engineering_filter] {len(results)} regions → {counts}")

        return results

    def _classify(self, region: MeshRegion) -> FilteredPart:
        notes = []

        # ── Buildability score (manifold-ish, finite size, non-fractal) ──
        buildability = self._buildability_score(region)
        if buildability < 0.3:
            notes.append(f"low_buildability={buildability:.2f}")

        # ── Size sanity ──
        size_ok = self._size_sanity(region)
        if not size_ok:
            notes.append("size_sanity_fail")

        # ── Simplicity (variance of normals = how "smooth" surface is) ──
        simplicity = 1.0 - min(1.0, region.normal_variance / 0.5)  # 0..1

        # ── Decision ──
        if not size_ok or region.area_mm2 < self.MIN_AREA_MM2:
            tag = BuildabilityTag.NOISE
            confidence = 1.0 - buildability
        elif region.surface_type == SurfaceType.FREEFORM and simplicity < 0.4:
            # Freeform + non-smooth = real complex form, keep as mesh
            tag = BuildabilityTag.COMPLEX
            confidence = 0.7
            notes.append("freeform_complex")
        elif buildability >= 0.5 and len(region.face_indices) >= self.MIN_FACE_COUNT:
            # Good for primitive fitting
            tag = BuildabilityTag.BUILDABLE
            confidence = buildability
        else:
            # Default to noise — safer to drop than misfit
            tag = BuildabilityTag.NOISE
            confidence = 0.6

        return FilteredPart(
            region=region,
            tag=tag,
            confidence=confidence,
            buildability_score=buildability,
            size_sanity_ok=size_ok,
            simplicity_score=simplicity,
            notes=notes,
        )

    def _buildability_score(self, region: MeshRegion) -> float:
        """0..1 — how likely this region represents a buildable engineering form."""
        score = 1.0

        # Penalty: too few faces
        if len(region.face_indices) < self.MIN_FACE_COUNT:
            score *= 0.3
        elif len(region.face_indices) < 10:
            score *= 0.7

        # Penalty: high normal variance (= fractal/noisy surface)
        if region.normal_variance > 0.3:
            score *= 0.5

        # Penalty: extreme aspect ratio
        size = region.size_mm() if hasattr(region, 'size_mm') else (region.bbox_max - region.bbox_min)
        nonzero = size[size > 1e-3]
        if len(nonzero) >= 2:
            aspect = nonzero.max() / nonzero.min()
            if aspect > self.MAX_ASPECT_RATIO:
                score *= 0.3
            elif aspect > 20:
                score *= 0.7

        # Bonus: clean planar or cylindrical
        if region.surface_type in (SurfaceType.PLANAR, SurfaceType.CYLINDRICAL, SurfaceType.SPHERICAL):
            if region.normal_variance < 0.1:
                score = min(1.0, score * 1.2)

        return float(max(0.0, min(1.0, score)))

    def _size_sanity(self, region: MeshRegion) -> bool:
        """Region size in reasonable engineering range."""
        size = region.bbox_max - region.bbox_min
        # Reject sub-pixel artifacts
        if np.max(size) < 0.5:  # mm
            return False
        # Reject impossibly large (likely bug)
        if np.max(size) > 10_000:  # 10m
            return False
        return True
