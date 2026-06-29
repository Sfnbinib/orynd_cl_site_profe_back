"""
Mesh Loader — load STL/OBJ/PLY into unified MeshData structure.

Uses trimesh for robust loading of any mesh format.
Falls back to manual STL parsing if trimesh unavailable.

Dependencies: trimesh (pip install trimesh)
Optional: scipy (for convex hull, face adjacency acceleration)
"""
from __future__ import annotations
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    log.warning("[mesh_loader] trimesh not installed — limited to binary/ASCII STL only")

try:
    import cadquery as _cq
    HAS_CADQUERY = True
except ImportError:
    HAS_CADQUERY = False


@dataclass
class MeshData:
    """Unified mesh representation for the decomposition pipeline."""
    vertices: np.ndarray          # (N, 3) float64 — vertex positions
    faces: np.ndarray             # (M, 3) int — triangle indices
    face_normals: np.ndarray      # (M, 3) float64 — per-face normals
    vertex_normals: np.ndarray    # (N, 3) float64 — per-vertex normals (averaged)

    # Computed on load
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Metadata
    source_path: str = ""
    source_format: str = ""
    triangle_count: int = 0
    vertex_count: int = 0
    is_watertight: bool = False
    volume_mm3: float = 0.0
    surface_area_mm2: float = 0.0

    # Raw trimesh object (optional, for advanced operations)
    _trimesh: Any = field(default=None, repr=False)

    # Scale auto-detect hint (set when load_mesh(auto_scale=True))
    scale_hint: Optional[dict] = None

    def size_mm(self) -> np.ndarray:
        """Bounding box dimensions [width, height, depth] in mm."""
        return self.bbox_max - self.bbox_min

    def diagonal_mm(self) -> float:
        """Bounding box diagonal length."""
        return float(np.linalg.norm(self.size_mm()))


def load_mesh(
    path: str | Path,
    scale: float = 1.0,
    repair: bool = False,
    auto_scale: bool = False,
) -> MeshData:
    """
    Load a mesh file into MeshData.

    Args:
        path: path to STL, OBJ, PLY, STEP, or other mesh file
        scale: multiply all coordinates by this factor (e.g. 25.4 for inch→mm).
               Ignored if auto_scale=True and a confident guess is found.
        repair: run trimesh.repair to fill holes / fix normals / merge dup verts.
        auto_scale: try to detect units from bbox diagonal. See `detect_scale`.

    Returns:
        MeshData with vertices, faces, normals, and computed properties.
        If `auto_scale` was used and result is ambiguous, MeshData.scale_hint
        carries options for the UI to disambiguate.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    ext = path.suffix.lower()

    # STEP / STP — needs cadquery (OpenCascade) tessellation pass.
    if ext in (".step", ".stp"):
        if not HAS_CADQUERY:
            raise ImportError(
                f"cadquery required to load {ext} files. Install: pip install cadquery"
            )
        return _load_step_via_cadquery(path, scale, repair=repair, auto_scale=auto_scale)

    if HAS_TRIMESH:
        return _load_with_trimesh(path, scale, repair=repair, auto_scale=auto_scale)
    elif ext == ".stl":
        return _load_stl_manual(path, scale)
    else:
        raise ImportError(
            f"trimesh required to load {ext} files. Install: pip install trimesh"
        )


def load_mesh_from_bytes(
    data: bytes,
    file_type: str = "stl",
    scale: float = 1.0,
    repair: bool = False,
    auto_scale: bool = False,
) -> MeshData:
    """Load mesh from raw bytes (e.g. from HTTP upload or API response)."""
    if HAS_TRIMESH:
        mesh = trimesh.load(
            trimesh.util.wrap_as_stream(data),
            file_type=file_type,
        )
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if repair:
            _repair_trimesh(mesh)
        effective_scale = _resolve_scale(mesh, scale, auto_scale)
        result = _trimesh_to_meshdata(mesh, f"<bytes>.{file_type}", effective_scale.scale)
        result.scale_hint = effective_scale.hint
        return result
    else:
        raise ImportError("trimesh required for loading from bytes")


# ── Trimesh loader ──────────────────────────────────────────────────────────

def _load_with_trimesh(
    path: Path, scale: float, repair: bool = False, auto_scale: bool = False
) -> MeshData:
    mesh = trimesh.load(str(path), force="mesh")

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if repair:
        _repair_trimesh(mesh)

    effective_scale = _resolve_scale(mesh, scale, auto_scale)
    result = _trimesh_to_meshdata(mesh, str(path), effective_scale.scale)
    result.scale_hint = effective_scale.hint
    return result


# ── STEP via cadquery (OpenCascade) ─────────────────────────────────────────

def _load_step_via_cadquery(
    path: Path, scale: float, repair: bool = False, auto_scale: bool = False,
) -> MeshData:
    """Import a STEP/STP file via cadquery + OpenCascade.

    Strategy: importStep → tessellate via cadquery's STL export to a temp file,
    then re-load the STL through trimesh so we get a unified MeshData with
    the same downstream guarantees (face_normals, vertex_normals, bbox, etc.).

    The triangulation tolerance is set tighter than cadquery's default so
    curved features like cylinder walls don't collapse into chunky polygons.
    """
    if not HAS_TRIMESH:
        raise ImportError("trimesh required to tessellate STEP via cadquery → STL")

    log.info("[mesh_loader] STEP import via cadquery: %s", path)
    wp = _cq.importers.importStep(str(path))
    # cadquery's exporters.export uses tolerance=0.1mm by default. For
    # downstream classification we want curved surfaces denser — pass tighter.
    with _NamedTempSTL() as tmp_stl:
        _cq.exporters.export(wp, tmp_stl, tolerance=0.05, angularTolerance=0.1)
        return _load_with_trimesh(Path(tmp_stl), scale, repair=repair, auto_scale=auto_scale)


class _NamedTempSTL:
    """Context manager: create a temp .stl path, delete on exit."""
    def __enter__(self) -> str:
        import tempfile
        t = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        t.close()
        self.path = t.name
        return t.name

    def __exit__(self, *_) -> None:
        try:
            Path(self.path).unlink(missing_ok=True)
        except Exception:
            pass


# ── Repair ──────────────────────────────────────────────────────────────────

def _repair_trimesh(mesh: Any) -> None:
    """Run a sequence of in-place repair operations on a trimesh.Trimesh.

    Each step is wrapped — a single failure should not abort the rest.
    Trimesh API has churned across versions (4.x removed some helpers).
    """
    if not HAS_TRIMESH:
        return

    def _try(label: str, fn) -> None:
        try:
            fn()
        except Exception as e:
            log.debug("[mesh_loader] repair step %s failed: %s", label, e)

    _try("merge_vertices",       lambda: mesh.merge_vertices())
    _try("unique_faces",         lambda: mesh.update_faces(mesh.unique_faces()))
    _try("nondegenerate_faces",  lambda: mesh.update_faces(mesh.nondegenerate_faces()))
    _try("fix_normals",          lambda: trimesh.repair.fix_normals(mesh))
    _try("fix_inversion",        lambda: trimesh.repair.fix_inversion(mesh))
    _try("fill_holes",           lambda: trimesh.repair.fill_holes(mesh))
    _try("remove_unreferenced",  lambda: mesh.remove_unreferenced_vertices())

    log.info("[mesh_loader] repair done: %d verts / %d tris, watertight=%s",
             len(mesh.vertices), len(mesh.faces), bool(mesh.is_watertight))


# ── Scale auto-detect ───────────────────────────────────────────────────────

@dataclass
class _ScaleResolution:
    scale: float
    hint: Optional[dict] = None


def detect_scale(diagonal: float) -> dict:
    """
    Heuristic unit guess from bbox diagonal length (as loaded, unitless).

    Returns dict like:
      {"guess": "mm", "scale": 1.0, "confidence": 0.9, "ambiguous": False, "options": [...]}

    Rules of thumb:
      diag < 1        — likely meters or sub-mm: scale x1000 to mm, low confidence
      1 <= diag < 10  — likely meters → scale x1000 (engineering parts rarely <10mm overall)
      10 <= diag < 800 — already mm (most common 3D-print range): scale=1.0
      800 <= diag < 5000 — likely inches → scale x25.4 OR very large mm part: ambiguous
      diag >= 5000    — likely mm (large industrial), scale=1.0
    """
    if diagonal <= 0:
        return {"guess": "unknown", "scale": 1.0, "confidence": 0.0, "ambiguous": True, "options": []}
    if diagonal < 1.0:
        return {"guess": "m", "scale": 1000.0, "confidence": 0.6, "ambiguous": True,
                "options": [{"unit": "m", "scale": 1000.0}, {"unit": "mm", "scale": 1.0}]}
    if diagonal < 10.0:
        return {"guess": "m", "scale": 1000.0, "confidence": 0.7, "ambiguous": True,
                "options": [{"unit": "m", "scale": 1000.0}, {"unit": "cm", "scale": 10.0}, {"unit": "mm", "scale": 1.0}]}
    if diagonal < 800.0:
        return {"guess": "mm", "scale": 1.0, "confidence": 0.95, "ambiguous": False, "options": []}
    if diagonal < 5000.0:
        return {"guess": "inch", "scale": 25.4, "confidence": 0.55, "ambiguous": True,
                "options": [{"unit": "inch", "scale": 25.4}, {"unit": "mm", "scale": 1.0}]}
    return {"guess": "mm", "scale": 1.0, "confidence": 0.8, "ambiguous": False, "options": []}


def _resolve_scale(mesh: Any, manual_scale: float, auto_scale: bool) -> _ScaleResolution:
    """If auto_scale is on, derive scale from bbox; else use manual."""
    if not auto_scale:
        return _ScaleResolution(scale=manual_scale, hint=None)
    try:
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        if len(verts) == 0:
            return _ScaleResolution(scale=manual_scale, hint=None)
        bb = verts.max(axis=0) - verts.min(axis=0)
        diag = float(np.linalg.norm(bb))
    except Exception:
        return _ScaleResolution(scale=manual_scale, hint=None)

    hint = detect_scale(diag)
    chosen = hint["scale"] if not hint["ambiguous"] or hint["confidence"] >= 0.7 else manual_scale
    hint["raw_diagonal"] = round(diag, 3)
    hint["applied_scale"] = chosen
    return _ScaleResolution(scale=chosen, hint=hint)


def _trimesh_to_meshdata(mesh: Any, source: str, scale: float) -> MeshData:
    vertices = np.array(mesh.vertices, dtype=np.float64) * scale
    faces = np.array(mesh.faces, dtype=np.int32)
    face_normals = np.array(mesh.face_normals, dtype=np.float64)

    # Vertex normals (averaged from adjacent faces)
    vertex_normals = np.zeros_like(vertices)
    if hasattr(mesh, "vertex_normals"):
        vertex_normals = np.array(mesh.vertex_normals, dtype=np.float64)

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)

    return MeshData(
        vertices=vertices,
        faces=faces,
        face_normals=face_normals,
        vertex_normals=vertex_normals,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        center=(bbox_min + bbox_max) / 2,
        source_path=source,
        source_format=Path(source).suffix.lstrip("."),
        triangle_count=len(faces),
        vertex_count=len(vertices),
        is_watertight=bool(mesh.is_watertight),
        volume_mm3=float(mesh.volume) * (scale ** 3) if mesh.is_watertight else 0.0,
        surface_area_mm2=float(mesh.area) * (scale ** 2),
        _trimesh=mesh,
    )


# ── Manual STL loader (fallback) ────────────────────────────────────────────

def _load_stl_manual(path: Path, scale: float) -> MeshData:
    """Parse binary or ASCII STL without trimesh."""
    data = path.read_bytes()
    is_ascii = data[:5] == b"solid" and b"\n" in data[:80]

    if is_ascii:
        return _parse_ascii_stl(data, str(path), scale)
    else:
        return _parse_binary_stl(data, str(path), scale)


def _parse_binary_stl(data: bytes, source: str, scale: float) -> MeshData:
    num_triangles = struct.unpack_from("<I", data, 80)[0]
    vertices_list = []
    normals_list = []
    vert_map: dict[tuple, int] = {}
    faces_list = []

    offset = 84
    for _ in range(num_triangles):
        nx, ny, nz = struct.unpack_from("<fff", data, offset)
        offset += 12
        normals_list.append([nx, ny, nz])

        tri = []
        for _ in range(3):
            vx, vy, vz = struct.unpack_from("<fff", data, offset)
            offset += 12
            key = (round(vx, 6), round(vy, 6), round(vz, 6))
            if key not in vert_map:
                vert_map[key] = len(vertices_list)
                vertices_list.append([vx * scale, vy * scale, vz * scale])
            tri.append(vert_map[key])
        faces_list.append(tri)
        offset += 2  # attribute byte count

    vertices = np.array(vertices_list, dtype=np.float64)
    faces = np.array(faces_list, dtype=np.int32)
    face_normals = np.array(normals_list, dtype=np.float64)

    # Simple vertex normals by averaging adjacent face normals
    vertex_normals = np.zeros_like(vertices)
    for fi, face in enumerate(faces):
        for vi in face:
            vertex_normals[vi] += face_normals[fi]
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vertex_normals /= norms

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)

    return MeshData(
        vertices=vertices,
        faces=faces,
        face_normals=face_normals,
        vertex_normals=vertex_normals,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        center=(bbox_min + bbox_max) / 2,
        source_path=source,
        source_format="stl",
        triangle_count=len(faces),
        vertex_count=len(vertices),
        is_watertight=False,  # can't determine without trimesh
        volume_mm3=0.0,
        surface_area_mm2=0.0,
    )


def _parse_ascii_stl(data: bytes, source: str, scale: float) -> MeshData:
    import re
    text = data.decode("utf-8", errors="ignore")

    vertices_list = []
    normals_list = []
    vert_map: dict[tuple, int] = {}
    faces_list = []

    normal_pattern = re.compile(r"facet\s+normal\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)")
    vertex_pattern = re.compile(r"vertex\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)")

    facet_normals = normal_pattern.findall(text)
    all_verts = vertex_pattern.findall(text)

    for fi, (nx, ny, nz) in enumerate(facet_normals):
        normals_list.append([float(nx), float(ny), float(nz)])
        tri = []
        for j in range(3):
            idx = fi * 3 + j
            if idx < len(all_verts):
                vx, vy, vz = [float(c) for c in all_verts[idx]]
                key = (round(vx, 6), round(vy, 6), round(vz, 6))
                if key not in vert_map:
                    vert_map[key] = len(vertices_list)
                    vertices_list.append([vx * scale, vy * scale, vz * scale])
                tri.append(vert_map[key])
        if len(tri) == 3:
            faces_list.append(tri)

    vertices = np.array(vertices_list, dtype=np.float64) if vertices_list else np.zeros((0, 3))
    faces = np.array(faces_list, dtype=np.int32) if faces_list else np.zeros((0, 3), dtype=np.int32)
    face_normals = np.array(normals_list, dtype=np.float64) if normals_list else np.zeros((0, 3))

    vertex_normals = np.zeros_like(vertices)
    for fi, face in enumerate(faces):
        for vi in face:
            vertex_normals[vi] += face_normals[fi]
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vertex_normals /= norms

    bbox_min = vertices.min(axis=0) if len(vertices) > 0 else np.zeros(3)
    bbox_max = vertices.max(axis=0) if len(vertices) > 0 else np.zeros(3)

    return MeshData(
        vertices=vertices,
        faces=faces,
        face_normals=face_normals,
        vertex_normals=vertex_normals,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        center=(bbox_min + bbox_max) / 2,
        source_path=source,
        source_format="stl",
        triangle_count=len(faces),
        vertex_count=len(vertices),
    )
