"""
Base classes for primitive fitters.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class PrimitiveType(str, Enum):
    PLANE = "plane"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    BOX = "box"
    CONE = "cone"
    TORUS = "torus"
    MESH = "mesh"  # fallback when no primitive fits


@dataclass
class FitResult:
    """Result of attempting to fit a primitive to points."""
    type: PrimitiveType
    parameters: dict = field(default_factory=dict)
    placement: dict = field(default_factory=dict)  # position, orientation
    rms_error: float = float('inf')
    max_error: float = float('inf')
    point_count: int = 0
    success: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "parameters": self.parameters,
            "placement": self.placement,
            "rms_error": round(self.rms_error, 4),
            "max_error": round(self.max_error, 4),
            "point_count": self.point_count,
            "success": self.success,
            "notes": self.notes,
        }


class FitFailed(Exception):
    """Raised when a primitive cannot be fit to points."""
    pass


class FitterBase(ABC):
    """Base class for all primitive fitters."""

    primitive_type: PrimitiveType
    min_points: int = 10  # minimum points to attempt fit

    @abstractmethod
    def fit(self, points: np.ndarray, normals: Optional[np.ndarray] = None) -> FitResult:
        """
        Fit primitive to points.

        Args:
            points: (N, 3) array of 3D points
            normals: (N, 3) optional surface normals at each point

        Returns:
            FitResult with rms_error indicating quality.
            If fit completely fails, return result with success=False.
        """
        raise NotImplementedError

    def _check_min_points(self, points: np.ndarray) -> None:
        if points.shape[0] < self.min_points:
            raise FitFailed(f"Too few points: {points.shape[0]} < {self.min_points}")


# ─── Common utility functions ───

def compute_centroid(points: np.ndarray) -> np.ndarray:
    """Centroid of point cloud."""
    return points.mean(axis=0)


def fit_plane_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit plane to points via PCA.
    Returns (normal, centroid).
    """
    centroid = compute_centroid(points)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Smallest eigenvalue corresponds to the normal direction
    normal = eigenvectors[:, 0]
    return normal, centroid


def compute_rms_error(distances: np.ndarray) -> float:
    """RMS of signed/unsigned distances."""
    return float(np.sqrt(np.mean(distances ** 2)))


def rotation_matrix_from_vectors(vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
    """Rotation matrix that aligns vec_from to vec_to."""
    a = vec_from / (np.linalg.norm(vec_from) + 1e-12)
    b = vec_to / (np.linalg.norm(vec_to) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if abs(c + 1.0) < 1e-6:
        # 180° rotation, pick arbitrary perpendicular axis
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, perp)
        v /= np.linalg.norm(v)
        return -np.eye(3) + 2 * np.outer(v, v)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3)
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
