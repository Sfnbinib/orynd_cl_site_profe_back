"""
Sphere fitter — algebraic least squares.

Sphere: (x-cx)² + (y-cy)² + (z-cz)² = r²
Expanded: x² + y² + z² + Ax + By + Cz + D = 0
Solve linear system for (A, B, C, D), derive (cx, cy, cz, r).
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from .base import FitterBase, FitResult, PrimitiveType, FitFailed, compute_rms_error


class SphereFitter(FitterBase):
    primitive_type = PrimitiveType.SPHERE
    min_points = 10

    def fit(self, points: np.ndarray, normals: Optional[np.ndarray] = None) -> FitResult:
        try:
            self._check_min_points(points)

            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]

            # System: x² + y² + z² + Ax + By + Cz + D = 0
            A_mat = np.column_stack([x, y, z, np.ones_like(x)])
            b_vec = -(x ** 2 + y ** 2 + z ** 2)

            coeffs, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            A, B, C, D = coeffs

            cx = -A / 2
            cy = -B / 2
            cz = -C / 2
            r2 = cx ** 2 + cy ** 2 + cz ** 2 - D
            if r2 <= 0:
                raise FitFailed(f"invalid radius² = {r2}")
            r = float(np.sqrt(r2))

            center = np.array([cx, cy, cz])
            # Residuals
            distances = np.linalg.norm(points - center, axis=1)
            residuals = np.abs(distances - r)
            rms = compute_rms_error(residuals)
            max_err = float(residuals.max())

            return FitResult(
                type=PrimitiveType.SPHERE,
                parameters={
                    "radius_mm": r,
                },
                placement={
                    "center": center.tolist(),
                },
                rms_error=rms,
                max_error=max_err,
                point_count=points.shape[0],
                success=True,
            )

        except FitFailed as e:
            return FitResult(type=PrimitiveType.SPHERE, success=False, notes=str(e))
        except Exception as e:
            return FitResult(type=PrimitiveType.SPHERE, success=False, notes=f"error: {e}")
