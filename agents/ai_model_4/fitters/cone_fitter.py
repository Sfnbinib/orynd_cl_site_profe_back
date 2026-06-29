"""
Cone fitter — apex + axis + half-angle.

Strategy:
  1. PCA-derived axis (longest dimension).
  2. Iterative: find apex along axis that minimizes residuals to cone surface.
  3. Half-angle from average opening.
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from .base import FitterBase, FitResult, PrimitiveType, FitFailed, compute_centroid, compute_rms_error


class ConeFitter(FitterBase):
    primitive_type = PrimitiveType.CONE
    min_points = 30

    def fit(self, points: np.ndarray, normals: Optional[np.ndarray] = None) -> FitResult:
        try:
            self._check_min_points(points)

            centroid = compute_centroid(points)
            centered = points - centroid
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = np.argsort(eigenvalues)[::-1]
            axis = eigenvectors[:, order[0]]
            axis /= np.linalg.norm(axis)

            # Axial coordinate
            axial = centered @ axis
            # Radial coordinate
            radial_vec = centered - np.outer(axial, axis)
            radial = np.linalg.norm(radial_vec, axis=1)

            # For a cone: radius ∝ |axial - apex_axial|
            # Linear regression: radial = slope * (axial - apex)
            # → radial = slope * axial - slope * apex
            A_mat = np.column_stack([axial, np.ones_like(axial)])
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, radial, rcond=None)
            slope, intercept = coeffs

            if abs(slope) < 1e-6:
                raise FitFailed("cone slope too small (looks like cylinder)")

            apex_axial = -intercept / slope
            half_angle_rad = float(np.arctan(abs(slope)))

            apex = centroid + apex_axial * axis

            # Height of cone (from apex to furthest point on axis)
            heights_from_apex = axial - apex_axial
            max_h = float(heights_from_apex.max())
            min_h = float(heights_from_apex.min())

            # Residuals
            expected_radial = abs(slope) * (axial - apex_axial)
            residuals = np.abs(radial - expected_radial)
            rms = compute_rms_error(residuals)
            max_err = float(residuals.max())

            return FitResult(
                type=PrimitiveType.CONE,
                parameters={
                    "half_angle_deg": float(np.degrees(half_angle_rad)),
                    "height_mm": max_h - min_h,
                    "base_radius_mm": float(abs(slope) * max(abs(max_h), abs(min_h))),
                    "axis": axis.tolist(),
                },
                placement={
                    "apex": apex.tolist(),
                    "axis_end": (apex + axis * max_h).tolist(),
                },
                rms_error=rms,
                max_error=max_err,
                point_count=points.shape[0],
                success=True,
            )

        except FitFailed as e:
            return FitResult(type=PrimitiveType.CONE, success=False, notes=str(e))
        except Exception as e:
            return FitResult(type=PrimitiveType.CONE, success=False, notes=f"error: {e}")
