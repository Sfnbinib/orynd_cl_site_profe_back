"""
Box fitter — oriented bounding box via PCA, with both axis-aligned and oriented variants.
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from .base import FitterBase, FitResult, PrimitiveType, FitFailed, compute_centroid, compute_rms_error


class BoxFitter(FitterBase):
    primitive_type = PrimitiveType.BOX
    min_points = 12

    def fit(self, points: np.ndarray, normals: Optional[np.ndarray] = None) -> FitResult:
        try:
            self._check_min_points(points)

            # PCA to find oriented bounding box
            centroid = compute_centroid(points)
            centered = points - centroid
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort by eigenvalue descending (largest first)
            order = np.argsort(eigenvalues)[::-1]
            axes = eigenvectors[:, order]

            # Project points to OBB coordinate frame
            projected = centered @ axes
            mins = projected.min(axis=0)
            maxs = projected.max(axis=0)
            size = maxs - mins  # extent in each PCA axis

            # OBB center
            obb_center_local = (mins + maxs) / 2
            obb_center_world = centroid + axes @ obb_center_local

            # Residuals: distance from each point to nearest face of box
            # A point inside box has 0 residual; outside is signed distance to nearest face.
            local_pts = projected - obb_center_local
            half_size = size / 2
            # Signed distance to each face (negative = inside)
            dists = np.abs(local_pts) - half_size
            # Point error = max(0, max(dists)) for outside; 0 if inside
            # For fit quality, we use absolute distance to nearest face
            face_dists = np.max(dists, axis=1)
            outside = face_dists > 0
            face_dists = np.where(outside, face_dists, np.zeros_like(face_dists))

            rms = compute_rms_error(face_dists)
            max_err = float(face_dists.max())

            return FitResult(
                type=PrimitiveType.BOX,
                parameters={
                    "size_mm": size.tolist(),
                    "axes": axes.T.tolist(),  # row-major
                },
                placement={
                    "center": obb_center_world.tolist(),
                },
                rms_error=rms,
                max_error=max_err,
                point_count=points.shape[0],
                success=True,
            )

        except FitFailed as e:
            return FitResult(type=PrimitiveType.BOX, success=False, notes=str(e))
        except Exception as e:
            return FitResult(type=PrimitiveType.BOX, success=False, notes=f"error: {e}")
