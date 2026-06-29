"""
Cylinder fitter — RANSAC-based.

A cylinder is defined by axis (line) + radius.
Strategy:
  1. Estimate axis from PCA of point normals (or fallback to point distribution).
  2. Project points to plane perpendicular to axis → 2D circle fit.
  3. Compute residuals.
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from .base import FitterBase, FitResult, PrimitiveType, FitFailed, compute_centroid, compute_rms_error


class CylinderFitter(FitterBase):
    primitive_type = PrimitiveType.CYLINDER
    min_points = 20

    def fit(self, points: np.ndarray, normals: Optional[np.ndarray] = None) -> FitResult:
        try:
            self._check_min_points(points)

            # ── Step 1: Estimate axis ──
            if normals is not None and normals.shape[0] == points.shape[0]:
                # For a cylinder, normals are perpendicular to the axis.
                # So the smallest principal component of normals = the axis direction.
                axis = self._estimate_axis_from_normals(normals)
            else:
                # Fallback: largest principal component of points = axis (elongation)
                axis = self._estimate_axis_from_points(points)

            # ── Step 2: Project to plane perpendicular to axis ──
            centroid = compute_centroid(points)
            # Construct 2D basis perpendicular to axis
            u = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = u - axis * np.dot(u, axis)
            u /= np.linalg.norm(u)
            v = np.cross(axis, u)

            centered = points - centroid
            uv = np.column_stack([centered @ u, centered @ v])

            # ── Step 3: Fit 2D circle (algebraic least squares) ──
            cx, cy, r = self._fit_circle_2d(uv)

            # 3D center on axis
            center_3d = centroid + cx * u + cy * v

            # ── Step 4: Height (extent along axis) ──
            axial = centered @ axis
            height = float(axial.max() - axial.min())

            # ── Step 5: Residuals ──
            radial_distances = np.sqrt((uv[:, 0] - cx) ** 2 + (uv[:, 1] - cy) ** 2)
            residuals = np.abs(radial_distances - r)
            rms = compute_rms_error(residuals)
            max_err = float(residuals.max())

            return FitResult(
                type=PrimitiveType.CYLINDER,
                parameters={
                    "radius_mm": float(r),
                    "height_mm": height,
                    "axis": axis.tolist(),
                },
                placement={
                    "center": center_3d.tolist(),
                    "axis_start": (center_3d + axis * axial.min()).tolist(),
                    "axis_end": (center_3d + axis * axial.max()).tolist(),
                },
                rms_error=rms,
                max_error=max_err,
                point_count=points.shape[0],
                success=True,
            )

        except FitFailed as e:
            return FitResult(type=PrimitiveType.CYLINDER, success=False, notes=str(e))
        except Exception as e:
            return FitResult(type=PrimitiveType.CYLINDER, success=False, notes=f"error: {e}")

    def _estimate_axis_from_normals(self, normals: np.ndarray) -> np.ndarray:
        """For a cylinder, all normals lie in plane perpendicular to axis."""
        normals_unit = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
        cov = normals_unit.T @ normals_unit / normals.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Smallest eigenvalue → least variance → axis direction
        axis = eigenvectors[:, 0]
        return axis / np.linalg.norm(axis)

    def _estimate_axis_from_points(self, points: np.ndarray) -> np.ndarray:
        """Fallback: largest principal axis of points."""
        centered = points - compute_centroid(points)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Largest eigenvalue → axis direction (longest spread)
        axis = eigenvectors[:, -1]
        return axis / np.linalg.norm(axis)

    def _fit_circle_2d(self, points2d: np.ndarray) -> tuple[float, float, float]:
        """
        Algebraic circle fit.
        Solve: x² + y² + Ax + By + C = 0
        Returns: (cx, cy, r)
        """
        x = points2d[:, 0]
        y = points2d[:, 1]

        # Linear system: [[x, y, 1], ...] @ [A, B, C] = -(x² + y²)
        A_matrix = np.column_stack([x, y, np.ones_like(x)])
        b_vec = -(x ** 2 + y ** 2)

        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_matrix, b_vec, rcond=None)
        except np.linalg.LinAlgError:
            raise FitFailed("circle lstsq failed")

        a, b, c = coeffs
        cx = -a / 2
        cy = -b / 2
        r2 = cx ** 2 + cy ** 2 - c
        if r2 <= 0:
            raise FitFailed(f"invalid radius² = {r2}")
        r = float(np.sqrt(r2))

        return float(cx), float(cy), r
