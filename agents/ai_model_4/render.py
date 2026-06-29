"""
Mesh visualization helpers for AI Model 4 verification.

Renders MeshData and dual-pass artifacts to PNG via matplotlib (no OpenGL
required — works headless on any Mac). Each function produces a triple-view
figure (iso + front + top) so the user can sanity-check geometry without
opening a CAD tool.

Used by `scripts/visualize_ai_model_4.py`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")  # headless backend — required for batch rendering

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from orynd_core.services.mesh.loader import MeshData
from orynd_core.services.mesh.decomposer import MeshRegion
from .engineering_filter import BuildabilityTag, FilteredPart
from .pass2_rebuild import PrimitiveFit
from .fitters import PrimitiveType

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Color schemes
# ─────────────────────────────────────────────

TAG_COLORS = {
    BuildabilityTag.BUILDABLE: "#2ecc71",  # green
    BuildabilityTag.COMPLEX:   "#f39c12",  # orange
    BuildabilityTag.NOISE:     "#e74c3c",  # red
}

PRIMITIVE_COLORS = {
    PrimitiveType.BOX:      "#3498db",
    PrimitiveType.CYLINDER: "#9b59b6",
    PrimitiveType.SPHERE:   "#e91e63",
    PrimitiveType.PLANE:    "#1abc9c",
    PrimitiveType.CONE:     "#f39c12",
    PrimitiveType.TORUS:    "#e67e22",
    PrimitiveType.MESH:     "#95a5a6",
}

DEFAULT_VIEWS = [
    ("iso",   30, -60),
    ("front",  0,   0),
    ("top",   89,   0),  # ~top-down; 90 has a degenerate view
]


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _triangles_for_faces(mesh: MeshData, face_indices: Iterable[int] | None = None) -> np.ndarray:
    """Return an (N, 3, 3) array of triangles (verts) for the chosen face subset."""
    if face_indices is None:
        face_idx = np.arange(len(mesh.faces))
    else:
        face_idx = np.asarray(list(face_indices), dtype=np.int64)
    if face_idx.size == 0:
        return np.empty((0, 3, 3), dtype=np.float64)
    return mesh.vertices[mesh.faces[face_idx]]


def _equal_aspect(ax, mesh: MeshData) -> None:
    """Force equal aspect so primitives don't look stretched."""
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    spans = bbox_max - bbox_min
    span = float(spans.max()) or 1.0
    centers = (bbox_min + bbox_max) / 2.0
    half = span / 2.0 * 1.05
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)
    # Newer matplotlib supports this; older ignores it. Either way looks OK.
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:  # noqa: BLE001
        pass


def _strip_axes(ax) -> None:
    ax.set_axis_off()
    try:
        ax.set_facecolor("#f8f9fa")
    except Exception:  # noqa: BLE001
        pass


def _save_three_view(
    mesh: MeshData,
    title: str,
    output_path: Path,
    draw_fn,
    figsize=(15, 5),
) -> None:
    """
    Render `mesh` from 3 angles. `draw_fn(ax)` paints whatever overlay the
    caller wants — it gets a clean axes per view.
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    for i, (label, elev, azim) in enumerate(DEFAULT_VIEWS, start=1):
        ax = fig.add_subplot(1, len(DEFAULT_VIEWS), i, projection="3d")
        draw_fn(ax)
        _equal_aspect(ax, mesh)
        ax.view_init(elev=elev, azim=azim)
        _strip_axes(ax)
        ax.set_title(label, fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────
# Public renderers
# ─────────────────────────────────────────────

def render_mesh(
    mesh: MeshData,
    output_path: Path,
    title: str = "Input mesh",
    color: str = "#bdc3c7",
) -> None:
    """Solid single-color render of the entire mesh."""
    tris = _triangles_for_faces(mesh)

    def draw(ax):
        col = Poly3DCollection(tris, facecolors=color, edgecolors="#7f8c8d",
                               linewidths=0.15, alpha=0.95)
        ax.add_collection3d(col)

    _save_three_view(mesh, title, output_path, draw)


def render_decomposition(
    mesh: MeshData,
    regions: list[MeshRegion],
    output_path: Path,
    title: str = "Pass 1: decomposition (region colors)",
) -> None:
    """Each region gets a distinct color from a cyclic palette."""
    palette = cm.get_cmap("tab20", max(len(regions), 1))

    def draw(ax):
        for i, region in enumerate(regions):
            tris = _triangles_for_faces(mesh, region.face_indices)
            if tris.size == 0:
                continue
            color = palette(i % palette.N)
            col = Poly3DCollection(tris, facecolors=[color], edgecolors="#34495e",
                                   linewidths=0.1, alpha=0.92)
            ax.add_collection3d(col)

    _save_three_view(mesh, f"{title} — {len(regions)} regions", output_path, draw)


def render_filter(
    mesh: MeshData,
    regions: list[MeshRegion],
    parts: list[FilteredPart],
    output_path: Path,
    title: str = "Engineering filter: buildable vs noise",
) -> None:
    """Color regions by buildability tag."""
    tag_by_rid = {p.region.region_id: p.tag for p in parts}

    def draw(ax):
        for region in regions:
            tag = tag_by_rid.get(region.region_id, BuildabilityTag.NOISE)
            tris = _triangles_for_faces(mesh, region.face_indices)
            if tris.size == 0:
                continue
            color = TAG_COLORS.get(tag, "#95a5a6")
            col = Poly3DCollection(tris, facecolors=color, edgecolors="#2c3e50",
                                   linewidths=0.15, alpha=0.9)
            ax.add_collection3d(col)

    counts = {t.value: sum(1 for p in parts if p.tag == t) for t in BuildabilityTag}
    legend = " | ".join(f"{k}: {v}" for k, v in counts.items())
    _save_three_view(mesh, f"{title}\n{legend}", output_path, draw)


def render_fits(
    mesh: MeshData,
    regions: list[MeshRegion],
    fits: list[PrimitiveFit],
    output_path: Path,
    title: str = "Pass 2: fitted primitives",
) -> None:
    """Color regions by the primitive type Pass 2 chose for them."""
    chosen_by_rid = {f.region_id: f.chosen_primitive for f in fits}

    def draw(ax):
        # First, plot un-fitted regions as faint grey so the user can see what
        # got dropped.
        for region in regions:
            if region.region_id in chosen_by_rid:
                continue
            tris = _triangles_for_faces(mesh, region.face_indices)
            if tris.size == 0:
                continue
            col = Poly3DCollection(tris, facecolors="#ecf0f1",
                                   edgecolors="#bdc3c7", linewidths=0.1, alpha=0.6)
            ax.add_collection3d(col)
        # Then, fits on top.
        for region in regions:
            primitive = chosen_by_rid.get(region.region_id)
            if primitive is None:
                continue
            tris = _triangles_for_faces(mesh, region.face_indices)
            if tris.size == 0:
                continue
            color = PRIMITIVE_COLORS.get(primitive, "#34495e")
            col = Poly3DCollection(tris, facecolors=color,
                                   edgecolors="#2c3e50", linewidths=0.15, alpha=0.93)
            ax.add_collection3d(col)

    counts: dict[str, int] = {}
    for f in fits:
        k = f.chosen_primitive.value
        counts[k] = counts.get(k, 0) + 1
    legend = " | ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no fits"
    _save_three_view(mesh, f"{title}\n{legend}", output_path, draw)


def render_cad_output(
    cad_stl_path: Path,
    output_path: Path,
    title: str = "CAD output (rebuilt from primitives)",
    color: str = "#3498db",
) -> bool:
    """Load the CADAgent's STL export and render it. Returns False if empty."""
    import trimesh

    if not cad_stl_path.exists():
        return False
    mesh_obj = trimesh.load(str(cad_stl_path), force="mesh")
    if mesh_obj is None or len(mesh_obj.faces) == 0:
        return False

    # Wrap as MeshData-shaped object so we can reuse _equal_aspect.
    class _Shim:
        vertices = np.asarray(mesh_obj.vertices)
        faces = np.asarray(mesh_obj.faces)
    shim = _Shim()

    tris = shim.vertices[shim.faces]

    def draw(ax):
        col = Poly3DCollection(tris, facecolors=color, edgecolors="#2c3e50",
                               linewidths=0.15, alpha=0.95)
        ax.add_collection3d(col)

    _save_three_view(shim, title, output_path, draw)
    return True
