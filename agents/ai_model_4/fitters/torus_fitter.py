"""
Torus fitter — major radius R + minor radius r + axis.

Heuristic fit (no full nonlinear solver):
  1. Estimate axis via PCA.
  2. Project to plane perpendicular to axis → annulus.
  3. R = avg of (r_max + r_min) / 2, r = (r_max - r_min) / 2 in the annulus.
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from .base import FitterBase, FitResult, PrimitiveType, FitFailed, compute_centroid, compute_rms_error


class TorusFitter(FitterBase):
    primitive_type = PrimitiveType.TORUS
    min_points = 40

    def fit(self, points: np.ndarray, normals: Optional[np.ndarray] = None) -> FitResult:
        try:
            self._check_min_points(points)

            centroid = compute_centroid(points)
            centered = points - centroid
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = np.argsort(eigenvalues)  # smallest first
            # For a torus lying flat, axis = smallest spread direction (perpendicular to the plane of the ring)
            axis = eigenvectors[:, order[0]]
            axis /= np.linalg.norm(axis)

            # Project points to plane perpendicular to axis
            axial = centered @ axis
            radial_vec = centered - np.outer(axial, axis)
            radial = np.linalg.norm(radial_vec, axis=1)

            # In annulus, R ≈ mean radial, r ≈ stdev radial (rough)
            R = float(np.mean(radial))
            r_estimate = float(np.std(radial))

            # Refine r via axial spread (for a torus, axial coordinates form a small distribution)
            r_axial = float((axial.max() - axial.min()) / 2)
            r = max(r_estimate, r_axial * 0.7)

            if r <= 1e-3 or R <= r:
                raise FitFailed(f"degenerate torus R={R} r={r}")

            # Residuals: for each point, distance to nearest point on torus surface
            # torus surface: (sqrt(x²+y²) - R)² + z² = r²  (in axis-aligned frame)
            # We have radial and axial coords already
            residuals = np.abs(np.sqrt((radial - R) ** 2 + axial ** 2) - r)
            rms = compute_rms_error(residuals)
            max_err = float(residuals.max())

            return FitResult(
                type=PrimitiveType.TORUS,
                parameters={
                    "major_radius_mm": R,
                    "minor_radius_mm": r,
                    "axis": axis.tolist(),
                },
                placement={
                    "center": centroid.tolist(),
                },
                rms_error=rms,
                max_error=max_err,
                point_count=points.shape[0],
                success=True,
            )

        except FitFailed as e:
            return FitResult(type=PrimitiveType.TORUS, success=False, notes=str(e))
        except Exception as e:
            return FitResult(type=PrimitiveType.TORUS, success=False, notes=f"error: {e}")
