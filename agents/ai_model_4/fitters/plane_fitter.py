"""
Plane fitter — PCA-based, simplest fitter.
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from .base import FitterBase, FitResult, PrimitiveType, FitFailed, fit_plane_pca, compute_rms_error


class PlaneFitter(FitterBase):
    primitive_type = PrimitiveType.PLANE
    min_points = 3

    def fit(self, points: np.ndarray, normals: Optional[np.ndarray] = None) -> FitResult:
        try:
            self._check_min_points(points)
            normal, centroid = fit_plane_pca(points)

            # Signed distances from points to plane
            distances = np.abs((points - centroid) @ normal)
            rms = compute_rms_error(distances)
            max_err = float(distances.max())

            # Bounding box on plane
            # Project points to 2D in-plane coords
            u = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = u - normal * np.dot(u, normal)
            u /= np.linalg.norm(u)
            v = np.cross(normal, u)

            projected = (points - centroid) @ np.column_stack([u, v])
            uv_min = projected.min(axis=0)
            uv_max = projected.max(axis=0)
            extent = (uv_max - uv_min).tolist()

            return FitResult(
                type=PrimitiveType.PLANE,
                parameters={
                    "normal": normal.tolist(),
                    "extent_uv_mm": extent,
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
            return FitResult(type=PrimitiveType.PLANE, success=False, notes=str(e))
        except Exception as e:
            return FitResult(type=PrimitiveType.PLANE, success=False, notes=f"error: {e}")
