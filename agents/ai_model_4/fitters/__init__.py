"""
Primitive fitters for Pass 2.

Each fitter attempts to fit a geometric primitive to a set of mesh points.
Returns FitResult with RMS error — lower is better.

Available primitives:
  - plane (degenerate, but useful for flat regions)
  - cylinder
  - sphere
  - box (axis-aligned + oriented)
  - torus
  - cone
"""
from .base import FitterBase, FitResult, PrimitiveType
from .plane_fitter import PlaneFitter
from .cylinder_fitter import CylinderFitter
from .sphere_fitter import SphereFitter
from .box_fitter import BoxFitter
from .cone_fitter import ConeFitter
from .torus_fitter import TorusFitter

ALL_FITTERS = [
    PlaneFitter(),
    CylinderFitter(),
    SphereFitter(),
    BoxFitter(),
    ConeFitter(),
    TorusFitter(),
]

__all__ = [
    "FitterBase",
    "FitResult",
    "PrimitiveType",
    "PlaneFitter",
    "CylinderFitter",
    "SphereFitter",
    "BoxFitter",
    "ConeFitter",
    "TorusFitter",
    "ALL_FITTERS",
]
