"""
Pass 2 — Engineering-clean primitive rebuild.

For each region tagged BUILDABLE by EngineeringFilter:
  1. Try every primitive fitter.
  2. Pick best fit (lowest RMS / acceptable threshold).
  3. If no fit good enough → keep as mesh, tag "complex".

Output: list of PrimitiveFit objects ready for CoreOps generation.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from orynd_core.services.mesh.loader import MeshData
from orynd_core.services.mesh.decomposer import MeshRegion

from .engineering_filter import FilteredPart, BuildabilityTag
from .fitters import ALL_FITTERS, FitResult, PrimitiveType

log = logging.getLogger(__name__)


@dataclass
class PrimitiveFit:
    """One region, best primitive fit."""
    region_id: int
    chosen_primitive: PrimitiveType
    fit_result: FitResult
    rejected_fits: list[FitResult] = field(default_factory=list)
    acceptable: bool = True  # whether fit error within threshold

    def to_dict(self) -> dict:
        return {
            "region_id": self.region_id,
            "chosen_primitive": self.chosen_primitive.value,
            "fit": self.fit_result.to_dict(),
            "rejected_count": len(self.rejected_fits),
            "acceptable": self.acceptable,
        }


class Pass2Rebuilder:
    """
    Tries each primitive fitter, picks best.

    Decision rule:
      - If best fit RMS < error_threshold_rel × region_size → accept.
      - Else fallback to MESH (keep original triangles).
    """

    def __init__(
        self,
        error_threshold_rel: float = 0.05,  # 5% of region diagonal
        error_threshold_abs_mm: float = 1.0,  # but always at least 1mm tolerance
    ):
        self.error_threshold_rel = error_threshold_rel
        self.error_threshold_abs_mm = error_threshold_abs_mm

    def rebuild(
        self,
        mesh: MeshData,
        filtered: list[FilteredPart],
    ) -> list[PrimitiveFit]:
        """For every BUILDABLE region, find best primitive."""
        results = []
        for part in filtered:
            if part.tag != BuildabilityTag.BUILDABLE:
                continue

            fit = self._fit_region(mesh, part.region)
            results.append(fit)

        log.info(
            f"[pass2] Rebuilt {len(results)} regions, "
            f"{sum(1 for r in results if r.acceptable)} acceptable fits"
        )
        return results

    def _fit_region(self, mesh: MeshData, region: MeshRegion) -> PrimitiveFit:
        # Extract points + normals for this region
        face_indices = np.array(region.face_indices)
        # Get vertices of these faces
        face_vert_indices = mesh.faces[face_indices]  # (N, 3)
        unique_verts = np.unique(face_vert_indices.flatten())
        region_points = mesh.vertices[unique_verts]  # (M, 3)

        # Approximate normals: use face normals
        region_normals = mesh.face_normals[face_indices] if hasattr(mesh, 'face_normals') else None

        # For per-vertex normals, we'd need to aggregate — skip for simplicity
        # Pass per-face normals to fitters that accept them

        # Try every fitter
        fit_results = []
        for fitter in ALL_FITTERS:
            try:
                # For fitters that want normals, pass face normals tiled to vertex count
                normals_arg = None
                if region_normals is not None:
                    # Provide normals via face-centroid points
                    face_centroids = mesh.vertices[mesh.faces[face_indices]].mean(axis=1)
                    if fitter.primitive_type == PrimitiveType.CYLINDER:
                        # Cylinder fitter can use these
                        result = fitter.fit(face_centroids, region_normals)
                    else:
                        result = fitter.fit(region_points)
                else:
                    result = fitter.fit(region_points)
                fit_results.append(result)
            except Exception as e:
                log.debug(f"[pass2] Fitter {fitter.primitive_type} failed: {e}")
                continue

        # Filter successful fits
        successful = [r for r in fit_results if r.success]
        if not successful:
            # No fit succeeded → MESH fallback
            return PrimitiveFit(
                region_id=region.region_id,
                chosen_primitive=PrimitiveType.MESH,
                fit_result=FitResult(
                    type=PrimitiveType.MESH,
                    success=True,
                    notes="no_primitive_fit",
                    point_count=region_points.shape[0],
                ),
                rejected_fits=fit_results,
                acceptable=False,
            )

        # Pick lowest RMS
        best = min(successful, key=lambda r: r.rms_error)

        # Check if acceptable
        region_size = float(np.linalg.norm(region.bbox_max - region.bbox_min))
        threshold = max(self.error_threshold_abs_mm, self.error_threshold_rel * region_size)
        acceptable = best.rms_error < threshold

        rejected = [r for r in successful if r is not best]

        return PrimitiveFit(
            region_id=region.region_id,
            chosen_primitive=best.type,
            fit_result=best,
            rejected_fits=rejected,
            acceptable=acceptable,
        )
