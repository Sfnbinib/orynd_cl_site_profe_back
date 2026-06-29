"""
Mesh Feature Extractor — convert decomposed mesh regions into CoreOps-compatible features.

This bridges the gap between raw mesh geometry and the parametric CoreOps JSON schema.
Each region becomes one or more manufacturing features (holes, pockets, bosses, etc).

Pipeline B flow:
  MeshData → decompose → regions → extract_features → CoreOps JSON
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .loader import MeshData
from .decomposer import DecompositionResult, MeshRegion, SurfaceType

log = logging.getLogger(__name__)


class FeatureType(str, Enum):
    """Manufacturing feature types recognized by CoreOps."""
    FLAT_FACE = "flat_face"
    HOLE = "hole"
    POCKET = "pocket"
    BOSS = "boss"
    FILLET = "fillet"
    CHAMFER = "chamfer"
    SLOT = "slot"
    STEP = "step"
    RIB = "rib"
    SHELL = "shell"
    CYLINDER = "cylinder"
    CONE = "cone"
    SPHERE = "sphere"
    FREEFORM_SURFACE = "freeform_surface"
    BASE_PLATE = "base_plate"


@dataclass
class MeshFeature:
    """A manufacturing feature extracted from mesh regions."""
    feature_id: str
    feature_type: FeatureType
    region_ids: list[int]                 # which mesh regions form this feature
    surface_type: SurfaceType

    # Geometry
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # center
    direction: np.ndarray = field(default_factory=lambda: np.zeros(3))  # normal or axis
    dimensions_mm: dict = field(default_factory=dict)  # width, height, depth, radius, etc.
    area_mm2: float = 0.0
    volume_mm3: float = 0.0  # estimated

    # Manufacturing hints
    is_through: bool = False          # for holes: goes all the way through
    is_blind: bool = False            # for holes/pockets: has a bottom
    depth_mm: float = 0.0
    draft_angle_deg: float = 0.0      # for moldability

    # Confidence
    confidence: float = 0.0           # 0..1 how sure we are about this classification

    # Relations
    parent_feature: Optional[str] = None
    child_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for CoreOps JSON schema."""
        d = {
            "feature_id": self.feature_id,
            "feature_type": self.feature_type.value,
            "surface_type": self.surface_type.value,
            "region_ids": self.region_ids,
            "position": self.position.tolist(),
            "direction": self.direction.tolist(),
            "dimensions_mm": self.dimensions_mm,
            "area_mm2": round(self.area_mm2, 2),
            "confidence": round(self.confidence, 3),
        }
        if self.is_through:
            d["is_through"] = True
        if self.is_blind:
            d["is_blind"] = True
        if self.depth_mm > 0:
            d["depth_mm"] = round(self.depth_mm, 3)
        if self.volume_mm3 > 0:
            d["volume_mm3"] = round(self.volume_mm3, 2)
        if self.parent_feature:
            d["parent_feature"] = self.parent_feature
        if self.child_features:
            d["child_features"] = self.child_features
        return d


@dataclass
class FeatureExtractionResult:
    """Complete feature extraction output."""
    features: list[MeshFeature]
    base_plate: Optional[MeshFeature] = None
    overall_dimensions_mm: dict = field(default_factory=dict)
    total_features: int = 0
    feature_summary: dict = field(default_factory=dict)

    def to_coreops_json(self) -> dict:
        """
        Convert to CoreOps-compatible JSON schema.
        This is the bridge between mesh analysis and the CAD generation pipeline.
        """
        return {
            "source": "mesh_pipeline",
            "model_version": "4.0",
            "overall_dimensions_mm": self.overall_dimensions_mm,
            "base_plate": self.base_plate.to_dict() if self.base_plate else None,
            "features": [f.to_dict() for f in self.features],
            "feature_summary": self.feature_summary,
            "total_features": self.total_features,
        }


# ── Main API ────────────────────────────────────────────────────────────────

def extract_features(
    mesh: MeshData,
    decomposition: DecompositionResult,
) -> FeatureExtractionResult:
    """
    Extract manufacturing features from decomposed mesh.

    This is the main entry point for Pipeline B feature extraction.
    """
    log.info(f"[feature_extractor] Extracting features from {len(decomposition.regions)} regions")

    features: list[MeshFeature] = []
    base_plate: Optional[MeshFeature] = None
    feat_counter = 0

    # Pass 1: Identify base plate (largest flat face facing down, or largest overall)
    base_plate = _find_base_plate(decomposition.regions)
    if base_plate:
        base_plate.feature_id = f"feat_{feat_counter:03d}"
        feat_counter += 1

    # Pass 2: Classify individual regions
    for region in decomposition.regions:
        if base_plate and region.region_id in base_plate.region_ids:
            continue  # skip base plate region

        feature = _classify_region_to_feature(
            region, mesh, decomposition, feat_counter
        )
        if feature:
            features.append(feature)
            feat_counter += 1

    # Pass 3: Detect composite features (holes = concave cylinder + optional flat bottom)
    composite = _detect_composite_features(
        features, decomposition, mesh, feat_counter
    )
    features.extend(composite)
    feat_counter += len(composite)

    # Pass 4: Detect patterns (arrays of identical features)
    _detect_patterns(features)

    # Build summary
    summary = {}
    for f in features:
        t = f.feature_type.value
        summary[t] = summary.get(t, 0) + 1

    size = mesh.size_mm()
    result = FeatureExtractionResult(
        features=features,
        base_plate=base_plate,
        overall_dimensions_mm={
            "width": round(float(size[0]), 2),
            "height": round(float(size[1]), 2),
            "depth": round(float(size[2]), 2),
        },
        total_features=len(features),
        feature_summary=summary,
    )

    log.info(f"[feature_extractor] Extracted {len(features)} features: {summary}")
    return result


# ── Base Plate Detection ────────────────────────────────────────────────────

def _find_base_plate(regions: list[MeshRegion]) -> Optional[MeshFeature]:
    """Find the base plate — largest planar region facing -Z (or largest flat)."""
    planar = [r for r in regions if r.surface_type == SurfaceType.PLANAR]
    if not planar:
        return None

    # Prefer downward-facing (normal ≈ -Z)
    down_facing = [r for r in planar if r.normal_mean[2] < -0.7]

    candidates = down_facing if down_facing else planar
    best = max(candidates, key=lambda r: r.area_mm2)

    return MeshFeature(
        feature_id="",  # assigned later
        feature_type=FeatureType.BASE_PLATE,
        region_ids=[best.region_id],
        surface_type=SurfaceType.PLANAR,
        position=best.centroid.copy(),
        direction=best.normal_mean.copy(),
        dimensions_mm={
            "width": round(float(best.size_mm()[0]), 2),
            "depth": round(float(best.size_mm()[1]), 2),
        },
        area_mm2=best.area_mm2,
        confidence=0.9 if down_facing else 0.6,
    )


# ── Single Region → Feature ────────────────────────────────────────────────

def _classify_region_to_feature(
    region: MeshRegion,
    mesh: MeshData,
    decomp: DecompositionResult,
    feat_id: int,
) -> Optional[MeshFeature]:
    """Convert a single mesh region into a manufacturing feature."""

    if region.surface_type == SurfaceType.PLANAR:
        return MeshFeature(
            feature_id=f"feat_{feat_id:03d}",
            feature_type=FeatureType.FLAT_FACE,
            region_ids=[region.region_id],
            surface_type=SurfaceType.PLANAR,
            position=region.centroid.copy(),
            direction=region.normal_mean.copy(),
            dimensions_mm={
                "width": round(float(region.size_mm()[0]), 2),
                "height": round(float(region.size_mm()[1]), 2),
            },
            area_mm2=region.area_mm2,
            confidence=0.85,
        )

    elif region.surface_type == SurfaceType.CYLINDRICAL:
        # Could be hole wall, boss wall, or standalone cylinder
        is_concave = _is_concave_cylinder(region, mesh)

        return MeshFeature(
            feature_id=f"feat_{feat_id:03d}",
            feature_type=FeatureType.HOLE if is_concave else FeatureType.CYLINDER,
            region_ids=[region.region_id],
            surface_type=SurfaceType.CYLINDRICAL,
            position=region.centroid.copy(),
            direction=region.axis.copy() if region.axis is not None else region.normal_mean.copy(),
            dimensions_mm={
                "radius": round(float(region.radius_mm or 0), 3),
                "diameter": round(float((region.radius_mm or 0) * 2), 3),
                "height": round(float(region.size_mm()[2]), 2),
            },
            area_mm2=region.area_mm2,
            is_through=False,  # determined in composite pass
            confidence=0.75,
        )

    elif region.surface_type == SurfaceType.SPHERICAL:
        return MeshFeature(
            feature_id=f"feat_{feat_id:03d}",
            feature_type=FeatureType.SPHERE,
            region_ids=[region.region_id],
            surface_type=SurfaceType.SPHERICAL,
            position=region.centroid.copy(),
            direction=region.normal_mean.copy(),
            dimensions_mm={
                "radius": round(float(region.radius_mm or 0), 3),
            },
            area_mm2=region.area_mm2,
            confidence=0.6,
        )

    else:  # FREEFORM
        # Small freeform near edges → likely fillet/chamfer
        if region.area_mm2 < 50 and region.normal_variance > 0.1:
            return MeshFeature(
                feature_id=f"feat_{feat_id:03d}",
                feature_type=FeatureType.FILLET,
                region_ids=[region.region_id],
                surface_type=SurfaceType.FREEFORM,
                position=region.centroid.copy(),
                direction=region.normal_mean.copy(),
                dimensions_mm={
                    "estimated_radius": round(float(min(region.size_mm()) / 2), 2),
                },
                area_mm2=region.area_mm2,
                confidence=0.5,
            )

        return MeshFeature(
            feature_id=f"feat_{feat_id:03d}",
            feature_type=FeatureType.FREEFORM_SURFACE,
            region_ids=[region.region_id],
            surface_type=SurfaceType.FREEFORM,
            position=region.centroid.copy(),
            direction=region.normal_mean.copy(),
            area_mm2=region.area_mm2,
            confidence=0.4,
        )


def _is_concave_cylinder(region: MeshRegion, mesh: MeshData) -> bool:
    """
    Check if cylindrical region is concave (hole) or convex (boss/shaft).

    Approach: fit cylinder axis line through region centroid along region.axis,
    then for every face in the region compare its face-normal direction with the
    radial direction from the cylinder axis to the face center.

      radial = face_center - projection_onto_axis(face_center)
      normal · radial > 0  →  normal points OUTWARD (convex boss)
      normal · radial < 0  →  normal points INWARD  (concave hole)

    This is the discrete equivalent of mean curvature sign for cylindrical
    surfaces — robust regardless of where the region's centroid sits relative
    to the rest of the body. Uses ALL faces (no biased sampling).
    """
    if region.axis is None or not region.face_indices:
        return False

    axis = region.axis / (np.linalg.norm(region.axis) + 1e-9)
    fi_arr = np.array(region.face_indices, dtype=np.int64)

    # face centers (N, 3) and normals (N, 3) for region faces only
    face_centers = mesh.vertices[mesh.faces[fi_arr]].mean(axis=1)
    normals = mesh.face_normals[fi_arr]

    # Project centers onto axis line through region.centroid
    rel = face_centers - region.centroid
    along = (rel @ axis)[:, None] * axis
    radial = rel - along  # vector pointing outward from axis

    radial_lens = np.linalg.norm(radial, axis=1)
    valid = radial_lens > 1e-6
    if not np.any(valid):
        return False

    radial_dir = np.zeros_like(radial)
    radial_dir[valid] = radial[valid] / radial_lens[valid, None]

    # Cosine between normal and outward radial; weight by face area for fairness.
    # Without scipy we don't have per-face area handy, but normals are unit-length
    # and faces inside a region tend to have similar area; uniform weight is fine.
    cos = np.einsum("ij,ij->i", normals[valid], radial_dir[valid])

    # Strongly negative mean = inward = concave (hole)
    return float(cos.mean()) < -0.1


# ── Composite Features ──────────────────────────────────────────────────────

def _detect_composite_features(
    features: list[MeshFeature],
    decomp: DecompositionResult,
    mesh: MeshData,
    start_id: int,
) -> list[MeshFeature]:
    """
    Detect composite features formed by multiple adjacent regions.
    E.g., a hole = cylindrical wall + optional flat bottom.
    A pocket = flat bottom + surrounding walls.
    """
    composites = []
    adj_set = set(decomp.adjacency_pairs)

    holes = [f for f in features if f.feature_type == FeatureType.HOLE]
    flat_faces = [f for f in features if f.feature_type == FeatureType.FLAT_FACE]

    for hole in holes:
        hole_rid = hole.region_ids[0]

        # Check if any flat face is adjacent → blind hole
        for flat in flat_faces:
            flat_rid = flat.region_ids[0]
            pair = (min(hole_rid, flat_rid), max(hole_rid, flat_rid))
            if pair in adj_set:
                # Check if flat face is small and perpendicular to hole axis
                if flat.area_mm2 < hole.area_mm2 * 0.5:
                    dot = abs(np.dot(flat.direction, hole.direction))
                    if dot > 0.7:  # approximately perpendicular to wall = parallel to axis
                        hole.is_blind = True
                        hole.depth_mm = float(np.linalg.norm(
                            flat.position - hole.position
                        ))
                        hole.child_features.append(flat.feature_id)
                        flat.parent_feature = hole.feature_id
                        break

        # If no bottom found, might be through-hole
        if not hole.is_blind:
            hole.is_through = True
            hole.confidence = min(hole.confidence + 0.1, 1.0)

    # ── Counterbore / countersink: two coaxial holes with different radii ──
    # Mark the wider (shallower) hole as the counterbore, link as parent.
    _detect_counterbores(holes)

    # Detect pockets: flat bottom surrounded by walls
    for flat in flat_faces:
        if flat.parent_feature:
            continue
        flat_rid = flat.region_ids[0]

        surrounding_walls = []
        for f in features:
            if f.feature_type in (FeatureType.FLAT_FACE, FeatureType.CYLINDER):
                frid = f.region_ids[0]
                pair = (min(flat_rid, frid), max(flat_rid, frid))
                if pair in adj_set and f.feature_id != flat.feature_id:
                    surrounding_walls.append(f)

        if len(surrounding_walls) >= 3:
            # Likely a pocket
            pocket = MeshFeature(
                feature_id=f"feat_{start_id + len(composites):03d}",
                feature_type=FeatureType.POCKET,
                region_ids=[flat_rid] + [w.region_ids[0] for w in surrounding_walls],
                surface_type=SurfaceType.PLANAR,
                position=flat.position.copy(),
                direction=flat.direction.copy(),
                dimensions_mm=flat.dimensions_mm.copy(),
                area_mm2=flat.area_mm2,
                is_blind=True,
                confidence=0.6,
                child_features=[flat.feature_id] + [w.feature_id for w in surrounding_walls],
            )
            composites.append(pocket)

    return composites


def _detect_counterbores(holes: list[MeshFeature]) -> None:
    """
    Pair up coaxial holes with different radii — typical counterbore/countersink.

    Marks the larger of the two as a 'counterbore' modifier on its dimensions_mm
    and links parent/child. Skips holes already linked.
    """
    if len(holes) < 2:
        return

    # Sort by radius desc so we process big first
    sorted_holes = sorted(holes, key=lambda h: -float(h.dimensions_mm.get("radius", 0)))
    used: set[str] = set()

    for i, big in enumerate(sorted_holes):
        if big.feature_id in used:
            continue
        big_r = float(big.dimensions_mm.get("radius", 0))
        if big_r <= 0:
            continue

        for small in sorted_holes[i + 1:]:
            if small.feature_id in used:
                continue
            small_r = float(small.dimensions_mm.get("radius", 0))
            if small_r <= 0 or small_r >= big_r * 0.9:
                continue  # need a real radius drop

            if not _are_coaxial(big, small, axis_tol=0.95, dist_tol_mm=max(big_r, 2.0)):
                continue

            # Mark the pair
            big.dimensions_mm["counterbore_for"] = small.feature_id
            big.dimensions_mm["counterbore_outer_radius_mm"] = round(big_r, 3)
            big.dimensions_mm["counterbore_inner_radius_mm"] = round(small_r, 3)
            big.child_features.append(small.feature_id)
            small.parent_feature = big.feature_id
            small.dimensions_mm["has_counterbore"] = True
            used.add(big.feature_id)
            used.add(small.feature_id)
            break


def _are_coaxial(
    a: MeshFeature, b: MeshFeature, axis_tol: float = 0.95, dist_tol_mm: float = 2.0
) -> bool:
    """Two holes are coaxial if directions are nearly (anti)parallel and
    the line through one center passes near the other center."""
    da = a.direction / (np.linalg.norm(a.direction) + 1e-9)
    db = b.direction / (np.linalg.norm(b.direction) + 1e-9)
    if abs(float(np.dot(da, db))) < axis_tol:
        return False

    # Distance from b.position to the line (a.position, da)
    diff = b.position - a.position
    parallel = float(np.dot(diff, da))
    perp = diff - parallel * da
    return float(np.linalg.norm(perp)) <= dist_tol_mm


# ── Pattern Detection ───────────────────────────────────────────────────────

# Tolerances for pattern matching (mm and radians).
_PATTERN_RADIUS_TOL_MM = 0.5      # holes grouped if radius differs by less
_PATTERN_COPLANAR_TOL = 0.10      # plane-fit residual / span ratio
_PATTERN_RADIAL_CV = 0.05         # bolt circle: std/mean of radial distances
_PATTERN_ANGULAR_CV = 0.10        # bolt circle: std/mean of angular gaps
_PATTERN_GRID_TOL_MM = 0.5        # grid: cluster tol on each axis
_PATTERN_GRID_FILL = 0.6          # grid: occupied / total cells required
_PATTERN_COLLINEAR_RATIO = 0.05   # collinear: PC2_extent / PC1_extent


def _detect_patterns(features: list[MeshFeature]) -> None:
    """
    Detect arrays/patterns of identical features.

    Recognizes:
      - bolt_circle: holes equidistant from a common center with equal angular step
      - rectangular_grid: NxM array on two orthogonal axes with uniform spacing
      - polar_array: features around an axis (generalization of bolt_circle)
      - linear: collinear features with uniform spacing
      - pair: exactly two matching features
      - cluster: matching features with no clear geometric structure

    Each matched feature gets `f.dimensions_mm["pattern"] = {...}`.
    Patterns are mutually exclusive — a feature belongs to at most one pattern.
    """
    holes = [
        f for f in features
        if f.feature_type == FeatureType.HOLE and f.dimensions_mm.get("radius")
    ]
    if len(holes) < 2:
        return

    for group in _group_by_radius(holes, _PATTERN_RADIUS_TOL_MM):
        if len(group) < 2:
            continue

        positions = np.array([f.position for f in group], dtype=np.float64)
        radius = float(np.mean([f.dimensions_mm["radius"] for f in group]))

        pattern = _classify_position_pattern(positions)
        pattern["hole_radius_mm"] = round(radius, 3)
        pattern["count"] = len(group)

        for f in group:
            f.dimensions_mm["pattern"] = pattern


# ── Grouping ──

def _group_by_radius(
    holes: list[MeshFeature], tol_mm: float
) -> list[list[MeshFeature]]:
    """Greedy single-link grouping of holes by radius."""
    sorted_holes = sorted(holes, key=lambda h: h.dimensions_mm["radius"])
    groups: list[list[MeshFeature]] = []
    current: list[MeshFeature] = []
    current_r: Optional[float] = None

    for h in sorted_holes:
        r = h.dimensions_mm["radius"]
        if current_r is None or abs(r - current_r) <= tol_mm:
            current.append(h)
            current_r = r if current_r is None else (current_r + r) / 2
        else:
            if current:
                groups.append(current)
            current = [h]
            current_r = r
    if current:
        groups.append(current)
    return groups


# ── Pattern classifiers ──

def _classify_position_pattern(positions: np.ndarray) -> dict:
    """
    Classify a set of 3D positions into a pattern descriptor.

    Order of attempts (most specific first):
      1. pair        — exactly 2 points
      2. bolt_circle — coplanar, equidistant from centroid, equal angular gaps
      3. polar_array — coplanar, equidistant, irregular angular gaps
      4. rect_grid   — coplanar, lies on NxM grid (N,M >= 2)
      5. linear      — collinear with uniform spacing
      6. cluster     — fallback
    """
    n = len(positions)
    if n == 2:
        return {
            "type": "pair",
            "spacing_mm": round(float(np.linalg.norm(positions[1] - positions[0])), 2),
        }

    centroid = positions.mean(axis=0)
    centered = positions - centroid

    # PCA via SVD
    try:
        _, sv, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"type": "cluster"}

    # Project into principal-axis frame (3 cols regardless of n)
    proj = centered @ vh.T
    # Pad to 3 columns if SVD returned fewer (n < 3)
    if proj.shape[1] < 3:
        pad = np.zeros((proj.shape[0], 3 - proj.shape[1]))
        proj = np.hstack([proj, pad])

    span = np.ptp(proj, axis=0)  # extent on each principal axis
    span_max = max(float(span[0]), 1e-9)

    # 1D: collinear
    if span[1] / span_max < _PATTERN_COLLINEAR_RATIO:
        return _classify_linear(proj[:, 0])

    # 2D: coplanar (span on PC3 negligible relative to PC1)
    coplanar = span[2] / span_max < _PATTERN_COPLANAR_TOL

    if coplanar and n >= 3:
        plane_pts = proj[:, :2]

        # Grid first — more specific than circle (avoids 2x2 grid → polar_array).
        grid = _classify_grid(plane_pts)
        if grid:
            grid["plane_normal"] = [round(float(v), 4) for v in (vh[2] if vh.shape[0] >= 3 else [0, 0, 1])]
            grid["center_mm"] = [round(float(c), 3) for c in centroid.tolist()]
            return grid

        # Then fit a circle in the principal plane (Kasa method).
        circle = _fit_circle_kasa(plane_pts)
        if circle is not None:
            cx, cy, r_fit = circle
            shifted = plane_pts - np.array([cx, cy])
            radial = np.linalg.norm(shifted, axis=1)
            r_mean = float(radial.mean())
            radial_cv = float(radial.std() / r_mean) if r_mean > 1e-6 else 1.0

            if r_mean > 1e-6 and radial_cv < _PATTERN_RADIAL_CV:
                angles = np.arctan2(shifted[:, 1], shifted[:, 0])
                angles_sorted = np.sort(angles)
                gaps = np.diff(np.concatenate([angles_sorted, [angles_sorted[0] + 2 * np.pi]]))
                gap_mean = float(gaps.mean())
                gap_cv = float(gaps.std() / gap_mean) if gap_mean > 1e-9 else 1.0

                axis = vh[2].tolist() if vh.shape[0] >= 3 else [0.0, 0.0, 1.0]
                # Center back to world coords: centroid + cx*PC1 + cy*PC2
                pc1 = vh[0] if vh.shape[0] >= 1 else np.array([1.0, 0.0, 0.0])
                pc2 = vh[1] if vh.shape[0] >= 2 else np.array([0.0, 1.0, 0.0])
                center_world = (centroid + cx * pc1 + cy * pc2).tolist()

                if gap_cv < _PATTERN_ANGULAR_CV:
                    return {
                        "type": "bolt_circle",
                        "circle_radius_mm": round(r_mean, 3),
                        "angular_step_deg": round(float(np.degrees(gap_mean)), 2),
                        "center_mm": [round(float(c), 3) for c in center_world],
                        "axis": [round(float(a), 4) for a in axis],
                    }
                return {
                    "type": "polar_array",
                    "circle_radius_mm": round(r_mean, 3),
                    "center_mm": [round(float(c), 3) for c in center_world],
                    "axis": [round(float(a), 4) for a in axis],
                }

    # Fall back to linear check on PC1 even if not strictly 1D
    linear = _classify_linear(proj[:, 0]) if n >= 3 else None
    if linear and linear["type"] == "linear":
        return linear

    return {"type": "cluster"}


def _classify_linear(coords_1d: np.ndarray) -> dict:
    """Detect uniform-spacing 1D pattern along a single axis."""
    if len(coords_1d) < 2:
        return {"type": "cluster"}
    sorted_c = np.sort(coords_1d)
    gaps = np.diff(sorted_c)
    if len(gaps) == 0:
        return {"type": "cluster"}
    gap_mean = float(gaps.mean())
    if gap_mean < 1e-6:
        return {"type": "cluster"}
    gap_cv = float(gaps.std() / gap_mean)
    if gap_cv < 0.1:
        return {
            "type": "linear",
            "spacing_mm": round(gap_mean, 2),
        }
    return {"type": "cluster"}


def _classify_grid(plane_pts: np.ndarray) -> Optional[dict]:
    """
    Detect an NxM rectangular grid in 2D points.
    Returns descriptor if rows>=2, cols>=2, and >= _PATTERN_GRID_FILL cells occupied.
    """
    if len(plane_pts) < 4:
        return None

    xs = _cluster_1d(plane_pts[:, 0], _PATTERN_GRID_TOL_MM)
    ys = _cluster_1d(plane_pts[:, 1], _PATTERN_GRID_TOL_MM)
    if len(xs) < 2 or len(ys) < 2:
        return None

    # Each point should snap to one (x_idx, y_idx) cell
    cells = set()
    for px, py in plane_pts:
        xi = _nearest_idx(xs, float(px))
        yi = _nearest_idx(ys, float(py))
        if abs(xs[xi] - px) > _PATTERN_GRID_TOL_MM:
            return None
        if abs(ys[yi] - py) > _PATTERN_GRID_TOL_MM:
            return None
        cells.add((xi, yi))

    total = len(xs) * len(ys)
    if len(cells) / total < _PATTERN_GRID_FILL:
        return None

    x_gaps = np.diff(xs)
    y_gaps = np.diff(ys)
    return {
        "type": "rect_grid",
        "cols": len(xs),
        "rows": len(ys),
        "spacing_x_mm": round(float(x_gaps.mean()), 2) if len(x_gaps) else 0.0,
        "spacing_y_mm": round(float(y_gaps.mean()), 2) if len(y_gaps) else 0.0,
        "occupied_cells": len(cells),
        "total_cells": total,
    }


def _cluster_1d(values: np.ndarray, tol: float) -> list[float]:
    """Cluster sorted 1D values within `tol` and return cluster means."""
    sorted_v = np.sort(values)
    clusters: list[list[float]] = [[float(sorted_v[0])]]
    for v in sorted_v[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(float(v))
        else:
            clusters.append([float(v)])
    return [float(np.mean(c)) for c in clusters]


def _fit_circle_kasa(pts: np.ndarray) -> Optional[tuple[float, float, float]]:
    """
    Algebraic least-squares circle fit (Kasa). Returns (cx, cy, r).

    Solves: x^2 + y^2 + D*x + E*y + F = 0
            → center = (-D/2, -E/2), r = sqrt(D^2/4 + E^2/4 - F)
    """
    if len(pts) < 3:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    D, E, F = sol
    cx = -D / 2
    cy = -E / 2
    val = cx ** 2 + cy ** 2 - F
    if val <= 0:
        return None
    return float(cx), float(cy), float(np.sqrt(val))


def _nearest_idx(sorted_values: list[float], target: float) -> int:
    """Return index of closest value in a sorted list."""
    best_i, best_d = 0, abs(sorted_values[0] - target)
    for i, v in enumerate(sorted_values[1:], start=1):
        d = abs(v - target)
        if d < best_d:
            best_d, best_i = d, i
    return best_i
