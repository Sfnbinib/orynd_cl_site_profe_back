"""
Mesh Decomposer — segment a triangle mesh into geometric regions.

Strategy: region-growing by face normal similarity.
  1. Build face adjacency graph
  2. Seed from unvisited face
  3. Grow region while neighbor normal angle < threshold
  4. Classify each region: planar / cylindrical / spherical / freeform

This is the core of Pipeline B (Mesh → features → CoreOps).

Dependencies: numpy (required), scipy (optional, accelerates adjacency)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .loader import MeshData

log = logging.getLogger(__name__)

try:
    from scipy.spatial import KDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class SurfaceType(str, Enum):
    PLANAR = "planar"
    CYLINDRICAL = "cylindrical"
    SPHERICAL = "spherical"
    CONICAL = "conical"
    FREEFORM = "freeform"


@dataclass
class MeshRegion:
    """A connected group of triangles forming a geometric surface."""
    region_id: int
    face_indices: list[int]           # indices into MeshData.faces
    surface_type: SurfaceType = SurfaceType.FREEFORM
    normal_mean: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal_variance: float = 0.0       # how spread out normals are (0=perfectly flat)
    area_mm2: float = 0.0
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # For cylindrical/conical surfaces
    axis: Optional[np.ndarray] = None
    radius_mm: Optional[float] = None

    # For planar surfaces
    plane_normal: Optional[np.ndarray] = None
    plane_offset: Optional[float] = None

    def size_mm(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min

    def to_dict(self) -> dict:
        """Serialize for JSON / CoreOps."""
        d = {
            "region_id": self.region_id,
            "surface_type": self.surface_type.value,
            "face_count": len(self.face_indices),
            "area_mm2": round(self.area_mm2, 2),
            "normal_mean": self.normal_mean.tolist(),
            "normal_variance": round(self.normal_variance, 6),
            "centroid": self.centroid.tolist(),
            "bbox_min": self.bbox_min.tolist(),
            "bbox_max": self.bbox_max.tolist(),
        }
        if self.axis is not None:
            d["axis"] = self.axis.tolist()
        if self.radius_mm is not None:
            d["radius_mm"] = round(self.radius_mm, 3)
        if self.plane_normal is not None:
            d["plane_normal"] = self.plane_normal.tolist()
            d["plane_offset"] = round(self.plane_offset or 0.0, 3)
        return d


@dataclass
class DecompositionResult:
    """Output of mesh decomposition."""
    regions: list[MeshRegion]
    face_labels: np.ndarray        # (M,) int — region_id for each face
    adjacency_pairs: list[tuple[int, int]]  # which regions touch each other
    stats: dict = field(default_factory=dict)

    def summary(self) -> dict:
        by_type = {}
        for r in self.regions:
            by_type[r.surface_type.value] = by_type.get(r.surface_type.value, 0) + 1
        return {
            "total_regions": len(self.regions),
            "by_type": by_type,
            "total_faces": int(self.face_labels.max() + 1) if len(self.face_labels) else 0,
        }


# ── Main API ────────────────────────────────────────────────────────────────

def decompose_mesh(
    mesh: MeshData,
    angle_threshold_deg: float = 15.0,
    min_region_faces: int = 5,
    merge_small: bool = True,
) -> DecompositionResult:
    """
    Decompose mesh into surface regions by normal similarity.

    Args:
        mesh: loaded MeshData
        angle_threshold_deg: max angle between adjacent face normals to group (degrees)
        min_region_faces: regions smaller than this get merged into neighbors
        merge_small: whether to merge tiny regions

    Returns:
        DecompositionResult with classified regions
    """
    log.info(
        f"[decomposer] Starting: {mesh.triangle_count} faces, "
        f"threshold={angle_threshold_deg}°, min_faces={min_region_faces}"
    )

    # Step 1: Build face adjacency
    adjacency = _build_face_adjacency(mesh)
    log.info(f"[decomposer] Adjacency built: {sum(len(v) for v in adjacency)} edges")

    # Step 2: Region growing
    cos_threshold = np.cos(np.radians(angle_threshold_deg))
    face_labels = np.full(mesh.triangle_count, -1, dtype=np.int32)
    regions_faces: list[list[int]] = []
    region_id = 0

    for seed in range(mesh.triangle_count):
        if face_labels[seed] != -1:
            continue

        # BFS from seed
        region = []
        queue = [seed]
        face_labels[seed] = region_id

        while queue:
            fi = queue.pop()
            region.append(fi)
            seed_normal = mesh.face_normals[fi]

            for neighbor in adjacency[fi]:
                if face_labels[neighbor] != -1:
                    continue
                # Check normal similarity
                cos_angle = np.dot(seed_normal, mesh.face_normals[neighbor])
                if cos_angle >= cos_threshold:
                    face_labels[neighbor] = region_id
                    queue.append(neighbor)

        regions_faces.append(region)
        region_id += 1

    log.info(f"[decomposer] Initial regions: {region_id}")

    # Step 3: Merge small regions
    if merge_small:
        regions_faces, face_labels = _merge_small_regions(
            regions_faces, face_labels, adjacency, mesh, min_region_faces
        )
        log.info(f"[decomposer] After merge: {len(regions_faces)} regions")

    # Step 4: Build MeshRegion objects with classification
    regions = []
    for rid, face_list in enumerate(regions_faces):
        region = _build_region(rid, face_list, mesh)
        regions.append(region)

    # Step 5: Find region adjacency
    region_adj = _find_region_adjacency(face_labels, adjacency)

    return DecompositionResult(
        regions=regions,
        face_labels=face_labels,
        adjacency_pairs=region_adj,
        stats={
            "angle_threshold_deg": angle_threshold_deg,
            "min_region_faces": min_region_faces,
            "initial_regions": region_id,
            "final_regions": len(regions),
        },
    )


# ── Face Adjacency ──────────────────────────────────────────────────────────

def _build_face_adjacency(mesh: MeshData) -> list[list[int]]:
    """
    Build face adjacency graph.
    Two faces are adjacent if they share an edge (2 common vertices).
    """
    n_faces = len(mesh.faces)
    adjacency: list[list[int]] = [[] for _ in range(n_faces)]

    # Edge → face index mapping
    edge_to_face: dict[tuple[int, int], list[int]] = {}

    for fi, face in enumerate(mesh.faces):
        for i in range(3):
            v0 = int(face[i])
            v1 = int(face[(i + 1) % 3])
            edge = (min(v0, v1), max(v0, v1))
            if edge not in edge_to_face:
                edge_to_face[edge] = []
            edge_to_face[edge].append(fi)

    for edge, faces in edge_to_face.items():
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                fi, fj = faces[i], faces[j]
                if fj not in adjacency[fi]:
                    adjacency[fi].append(fj)
                if fi not in adjacency[fj]:
                    adjacency[fj].append(fi)

    return adjacency


# ── Region Merging ──────────────────────────────────────────────────────────

def _merge_small_regions(
    regions_faces: list[list[int]],
    face_labels: np.ndarray,
    adjacency: list[list[int]],
    mesh: MeshData,
    min_faces: int,
) -> tuple[list[list[int]], np.ndarray]:
    """Merge regions smaller than min_faces into their largest neighbor.

    Robust against cycles: when small regions only neighbor other small regions
    (e.g. a cube where every region is 2 faces and min_faces=5), the best-
    neighbor map forms a cycle. We detect cycles, route everyone in the cycle
    to its single largest member, and bail out of the outer loop once no
    structural change has happened — preventing infinite looping.
    """
    max_iterations = max(8, len(regions_faces))
    for _iter in range(max_iterations):
        merge_map: dict[int, int] = {}

        # Find a target for every small region.
        for rid, faces in enumerate(regions_faces):
            if 0 < len(faces) < min_faces:
                neighbor_counts: dict[int, int] = {}
                for fi in faces:
                    for nfi in adjacency[fi]:
                        nlabel = int(face_labels[nfi])
                        if nlabel != rid:
                            neighbor_counts[nlabel] = neighbor_counts.get(nlabel, 0) + 1
                if neighbor_counts:
                    merge_map[rid] = max(neighbor_counts, key=neighbor_counts.get)

        if not merge_map:
            break

        # Resolve every entry to a final target with cycle detection.
        final_target: dict[int, int] = {}
        for start in merge_map:
            visited = [start]
            cur = merge_map[start]
            while cur in merge_map and cur not in visited:
                visited.append(cur)
                cur = merge_map[cur]
            if cur in visited:
                # cycle — pick largest region in the cycle
                cur = max(visited, key=lambda r: len(regions_faces[r]))
            final_target[start] = cur

        # Apply merges; track whether anything actually changed.
        merged_count = 0
        for old_id, new_id in final_target.items():
            if new_id == old_id or not regions_faces[old_id]:
                continue
            regions_faces[new_id].extend(regions_faces[old_id])
            for fi in regions_faces[old_id]:
                face_labels[fi] = new_id
            regions_faces[old_id] = []
            merged_count += 1

        if merged_count == 0:
            break  # No structural progress — stop to avoid infinite loop.

        # Compact — remove empty regions, re-index
        compact = []
        remap: dict[int, int] = {}
        for rid, faces in enumerate(regions_faces):
            if faces:
                remap[rid] = len(compact)
                compact.append(faces)
        for fi in range(len(face_labels)):
            old = face_labels[fi]
            if old in remap:
                face_labels[fi] = remap[old]
        regions_faces = compact

    return regions_faces, face_labels


# ── Region Building & Classification ────────────────────────────────────────

def _build_region(rid: int, face_indices: list[int], mesh: MeshData) -> MeshRegion:
    """Compute properties and classify a region."""
    normals = mesh.face_normals[face_indices]
    normal_mean = normals.mean(axis=0)
    norm = np.linalg.norm(normal_mean)
    if norm > 0:
        normal_mean /= norm

    # Normal variance: average angular deviation from mean
    dots = np.clip(np.dot(normals, normal_mean), -1, 1)
    angles = np.arccos(dots)
    normal_variance = float(np.mean(angles ** 2))

    # Area — sum of triangle areas
    area = 0.0
    all_verts = []
    for fi in face_indices:
        v0, v1, v2 = mesh.vertices[mesh.faces[fi]]
        area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        all_verts.extend([v0, v1, v2])

    all_verts = np.array(all_verts)
    centroid = all_verts.mean(axis=0)
    bbox_min = all_verts.min(axis=0)
    bbox_max = all_verts.max(axis=0)

    region = MeshRegion(
        region_id=rid,
        face_indices=face_indices,
        normal_mean=normal_mean,
        normal_variance=normal_variance,
        area_mm2=area,
        centroid=centroid,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )

    # Classify
    _classify_region(region, normals, all_verts)

    return region


def _classify_region(
    region: MeshRegion,
    normals: np.ndarray,
    vertices: np.ndarray,
) -> None:
    """
    Classify region surface type based on normal distribution.

    Heuristics (order matters — most specific first):
      - Planar:       normals are nearly all aligned with a single direction.
                      Detected via the largest SVD singular value of the raw
                      normals being ≈ N (all aligned) — variance is a coarser
                      proxy that still works.
      - Cylindrical:  normals span a 1D ring in a 2D subspace.  Detected when
                      the smallest SVD singular value of the *centered* normals
                      is ≈ 0 (normals coplanar) — the cylinder axis is then
                      that weak direction.  No variance gate: a full cylinder
                      has normals summing to zero (variance huge), which is
                      EXACTLY when this detector is needed.
      - Spherical:    normals fill a full 2D shell around a centre — three
                      principal singular values comparable, plus large variance.
      - Freeform:     everything else.
    """
    var = region.normal_variance

    # ── Planar ──
    if var < 0.01:
        region.surface_type = SurfaceType.PLANAR
        region.plane_normal = region.normal_mean.copy()
        region.plane_offset = float(np.dot(region.normal_mean, region.centroid))
        return

    if len(normals) < 6:
        region.surface_type = SurfaceType.FREEFORM
        return

    # SVD on centered face normals — geometry of the normal cloud.
    try:
        normals_centered = normals - normals.mean(axis=0)
        _, s, vh = np.linalg.svd(normals_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        region.surface_type = SurfaceType.FREEFORM
        return

    s0 = float(s[0]) if len(s) > 0 else 0.0
    s_mid_ratio = float(s[1] / (s0 + 1e-10)) if len(s) > 1 else 0.0
    s_min_ratio = float(s[-1] / (s0 + 1e-10))

    # ── Cylindrical: weakest singular value collapses (normals in a plane).
    # We require s_mid_ratio to be non-trivial so a planar region (which
    # already returned above) cannot fall through here.
    if s_min_ratio < 0.15 and s_mid_ratio > 0.3:
        region.surface_type = SurfaceType.CYLINDRICAL
        # Cylinder axis = direction in which the normals collapse to a point.
        axis = vh[-1]
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        region.axis = axis.copy()

        # Estimate radius by least-squares circle fit in the plane perpendicular
        # to the axis. Use vertex positions (not face centers) — they hug the
        # cylinder surface more tightly.
        c = vertices.mean(axis=0)
        rel = vertices - c
        # Strip the component along the axis.
        proj = rel - np.outer(rel @ axis, axis)
        # Distance from each point to the axis line through c.
        dists = np.linalg.norm(proj, axis=1)
        # Median is robust against vertices on the rim of capped cylinders.
        region.radius_mm = float(np.median(dists))
        return

    # ── Spherical: all three principal directions comparable, large spread.
    if s_min_ratio > 0.5 and var > 0.3:
        region.surface_type = SurfaceType.SPHERICAL
        center = vertices.mean(axis=0)
        dists = np.linalg.norm(vertices - center, axis=1)
        region.radius_mm = float(np.median(dists))
        return

    region.surface_type = SurfaceType.FREEFORM


# ── Region Adjacency ────────────────────────────────────────────────────────

def _find_region_adjacency(
    face_labels: np.ndarray,
    adjacency: list[list[int]],
) -> list[tuple[int, int]]:
    """Find which regions are adjacent to each other."""
    pairs: set[tuple[int, int]] = set()
    for fi in range(len(face_labels)):
        for nfi in adjacency[fi]:
            a, b = int(face_labels[fi]), int(face_labels[nfi])
            if a != b:
                pairs.add((min(a, b), max(a, b)))
    return sorted(pairs)
