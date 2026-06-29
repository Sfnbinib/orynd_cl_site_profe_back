"""
CadEngine — executes CoreOps operations via CadQuery (OpenCascade).

CadQuery is an optional dependency. When unavailable, the engine returns
a structured description of what WOULD be built (useful for LLM context).
Install: pip install cadquery
"""
from __future__ import annotations
import concurrent.futures
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CAD_TIMEOUT_SECONDS = 30   # max time per full build
MAX_OPERATIONS      = 300  # max CoreOps ops per request (mesh-decomposed parts: 60-200 primitives)
MIN_SOLID_VOLUME    = 1e-6 # mm³ — below this a body is treated as degenerate (no real geometry)

from .schemas import (
    CoreOp, CoreOpsDocument, CreateSketch, Extrude, Cut,
    CutHole, CutSlot, Fillet, Chamfer, Revolve, Loft, Boolean, Mirror,
)

log = logging.getLogger(__name__)

try:
    import cadquery as cq
    HAS_CADQUERY = True
except ImportError:
    HAS_CADQUERY = False
    log.warning("[cad_engine] CadQuery not installed — running in dry-run mode")


CAD_OUTPUT_DIR = Path(tempfile.gettempdir()) / "orynd_cad"


@dataclass
class CadResult:
    ok: bool
    body: Any = None
    properties: dict = field(default_factory=dict)
    stl_path: str | None = None
    step_path: str | None = None
    obj_path: str | None = None
    error: str | None = None
    dry_run: bool = False
    operations_executed: int = 0
    skipped_ops: list = field(default_factory=list)


class CadEngine:
    """Executes CoreOps → CadQuery → STEP/STL/OBJ."""

    def __init__(self) -> None:
        self._bodies: dict[str, Any] = {}
        self._sketches: dict[str, Any] = {}

    def execute(self, doc: CoreOpsDocument, session_id: str = "default") -> CadResult:
        ops = doc.parse_operations()
        if not ops:
            return CadResult(ok=False, error="No operations provided")

        # Guard: too many operations = likely malformed/malicious input
        if len(ops) > MAX_OPERATIONS:
            return CadResult(
                ok=False,
                error=f"Too many operations: {len(ops)} exceeds limit of {MAX_OPERATIONS}",
            )

        if not HAS_CADQUERY:
            return self._dry_run(ops, session_id)

        # Run CadQuery in a thread with timeout so complex geometry can't hang the server
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._execute_internal, ops, session_id)
            try:
                return future.result(timeout=CAD_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                log.error("[cad_engine] build timed out after %ss (ops=%d)", CAD_TIMEOUT_SECONDS, len(ops))
                return CadResult(
                    ok=False,
                    error=f"CAD build timed out after {CAD_TIMEOUT_SECONDS}s — geometry too complex",
                )

    def _execute_internal(self, ops: list, session_id: str) -> CadResult:
        """Actual CadQuery execution — called inside a timeout thread."""
        self._bodies.clear()
        self._sketches.clear()

        try:
            workplane = cq.Workplane("XY")
            current = workplane

            # Per-op geometry guard: one bad primitive (e.g. AI Model 4 emitting a
            # degenerate box with size [x, y, 5e-16]) makes OCCT throw
            # BRepSweep_Translation::Constructor. Catch it per-op, skip the broken
            # one, and keep building from the valid geometry — a partial part beats
            # a total failure. Mirrors the per-op *validation* skip in agents/cad.py,
            # applied here at the OCCT-geometry level.
            skipped_ops: list = []
            executed = 0
            for op in ops:
                try:
                    current = self._execute_op(op, current)
                    executed += 1
                except Exception as e:
                    op_name = getattr(op, "op", type(op).__name__)
                    op_id = getattr(op, "id", None)
                    msg = str(e).splitlines()[0][:140] if str(e) else type(e).__name__
                    log.warning("[cad_engine] op %s(id=%s) skipped: %s", op_name, op_id, msg)
                    skipped_ops.append({"op": op_name, "id": op_id, "reason": msg})
                    continue

            # Assemble the final solid from whatever survived (current may point at a
            # sketch or a skipped Boolean — fall back to unioning the valid bodies).
            final = self._final_solid(current)
            if final is None:
                return CadResult(
                    ok=False,
                    error=f"No valid geometry produced ({len(skipped_ops)} of {len(ops)} op(s) failed)",
                    skipped_ops=skipped_ops,
                    operations_executed=executed,
                )

            props = self._get_properties(final)

            out_dir = CAD_OUTPUT_DIR / session_id
            out_dir.mkdir(parents=True, exist_ok=True)

            stl_path = self._export_stl(final, out_dir / "part.stl")
            step_path = self._export_step(final, out_dir / "part.step")
            obj_path = self._export_obj(final, out_dir / "part.obj")

            return CadResult(
                ok=True,
                body=final,
                properties=props,
                stl_path=stl_path,
                step_path=step_path,
                obj_path=obj_path,
                operations_executed=executed,
                skipped_ops=skipped_ops,
            )
        except Exception as e:
            log.exception("[cad_engine] execution failed")
            return CadResult(ok=False, error=str(e))

    def _is_solid(self, wp: Any) -> bool:
        """True when `wp` carries a real (non-degenerate) solid."""
        try:
            return wp is not None and wp.val() is not None and wp.val().Volume() > MIN_SOLID_VOLUME
        except Exception:
            return False

    def _final_solid(self, current: Any) -> Any:
        """Pick the geometry to export.

        Prefer `current` when it already holds a valid solid (happy path: the
        final Boolean union). Otherwise union every valid body that survived the
        op loop — covers the case where the final union op was itself skipped
        because too few bodies remained.
        """
        if self._is_solid(current):
            return current
        solids = [b for b in self._bodies.values() if self._is_solid(b)]
        if not solids:
            return None
        result = solids[0]
        for b in solids[1:]:
            try:
                result = result.union(b)
            except Exception as e:
                log.warning("[cad_engine] final union skipped a body: %s", str(e).splitlines()[0][:140])
        return result

    def _execute_op(self, op: CoreOp, wp: Any) -> Any:
        if isinstance(op, CreateSketch):
            return self._op_create_sketch(op, wp)
        elif isinstance(op, Extrude):
            return self._op_extrude(op, wp)
        elif isinstance(op, Cut):
            return self._op_cut(op, wp)
        elif isinstance(op, CutHole):
            return self._op_cut_hole(op, wp)
        elif isinstance(op, CutSlot):
            return self._op_cut_slot(op, wp)
        elif isinstance(op, Fillet):
            return self._op_fillet(op, wp)
        elif isinstance(op, Chamfer):
            return self._op_chamfer(op, wp)
        elif isinstance(op, Revolve):
            return self._op_revolve(op, wp)
        elif isinstance(op, Loft):
            return self._op_loft(op, wp)
        elif isinstance(op, Boolean):
            return self._op_boolean(op, wp)
        elif isinstance(op, Mirror):
            return self._op_mirror(op, wp)
        else:
            raise ValueError(f"Unsupported operation: {type(op).__name__}")

    def _op_create_sketch(self, op: CreateSketch, wp: Any) -> Any:
        plane_map = {"XY": "XY", "XZ": "XZ", "YZ": "YZ"}
        plane = plane_map.get(op.plane, "XY")
        sketch_wp = cq.Workplane(plane).workplane(offset=op.offset)

        for shape in op.shapes:
            if shape.type == "rect":
                sketch_wp = sketch_wp.center(shape.center.x, shape.center.y).rect(
                    shape.width, shape.height
                )
            elif shape.type == "circle":
                sketch_wp = sketch_wp.center(shape.center.x, shape.center.y).circle(
                    shape.radius
                )
            elif shape.type == "polygon":
                pts = [(p.x, p.y) for p in shape.points]
                sketch_wp = sketch_wp.polyline(pts).close()

        self._sketches[op.id] = sketch_wp
        return sketch_wp

    def _op_extrude(self, op: Extrude, wp: Any) -> Any:
        sketch = self._sketches.get(op.sketch_ref, wp)
        if op.symmetric:
            result = sketch.extrude(op.height / 2, both=True)
        elif op.taper_angle != 0:
            result = sketch.extrude(op.height, taper=op.taper_angle)
        else:
            result = sketch.extrude(op.height)
        self._bodies[op.id] = result
        return result

    def _op_cut(self, op: Cut, wp: Any) -> Any:
        sketch = self._sketches.get(op.sketch_ref, wp)
        if op.through:
            result = sketch.cutThruAll()
        else:
            result = sketch.cutBlind(-op.depth)
        self._bodies[op.id] = result
        return result

    def _op_cut_hole(self, op: CutHole, wp: Any) -> Any:
        face_sel = ">Z" if op.on_face == "top" else "<Z"
        work = wp.faces(face_sel).workplane().center(op.center.x, op.center.y)
        if op.through:
            result = work.hole(op.radius * 2)
        else:
            result = work.hole(op.radius * 2, op.depth)
        self._bodies[op.id] = result
        return result

    def _op_cut_slot(self, op: CutSlot, wp: Any) -> Any:
        result = (
            wp.faces(">Z").workplane()
            .moveTo(op.start.x, op.start.y)
            .lineTo(op.end.x, op.end.y)
            .offset2D(op.width / 2)
            .cutBlind(-op.depth)
        )
        self._bodies[op.id] = result
        return result

    def _op_fillet(self, op: Fillet, wp: Any) -> Any:
        try:
            if "all" in op.edges:
                result = wp.edges().fillet(op.radius)
            else:
                result = wp.edges().fillet(op.radius)
            self._bodies[op.id] = result
            return result
        except Exception as e:
            # Fillet can fail on complex geometry (holes, thin edges).
            # Try with smaller radius, then skip gracefully.
            log.warning("[cad_engine] Fillet(r=%s) failed: %s — trying r=%s", op.radius, e, op.radius * 0.5)
            try:
                result = wp.edges().fillet(op.radius * 0.5)
                self._bodies[op.id] = result
                return result
            except Exception:
                log.warning("[cad_engine] Fillet skipped — geometry too complex for radius %s", op.radius)
                self._bodies[op.id] = wp
                return wp

    def _op_chamfer(self, op: Chamfer, wp: Any) -> Any:
        try:
            if "all" in op.edges:
                result = wp.edges().chamfer(op.distance)
            else:
                result = wp.edges().chamfer(op.distance)
            self._bodies[op.id] = result
            return result
        except Exception as e:
            log.warning("[cad_engine] Chamfer(d=%s) failed: %s — skipping", op.distance, e)
            self._bodies[op.id] = wp
            return wp

    def _op_revolve(self, op: Revolve, wp: Any) -> Any:
        sketch = self._sketches.get(op.sketch_ref, wp)
        axis_vec = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}
        vec = axis_vec.get(op.axis, (0, 1, 0))
        result = sketch.revolve(op.angle, (0, 0, 0), vec)
        self._bodies[op.id] = result
        return result

    def _op_loft(self, op: Loft, wp: Any) -> Any:
        wires = [self._sketches[ref] for ref in op.sketch_refs if ref in self._sketches]
        if len(wires) < 2:
            raise ValueError("Loft requires at least 2 sketches")
        result = cq.Workplane("XY").loft(ruled=op.ruled)
        self._bodies[op.id] = result
        return result

    def _op_boolean(self, op: Boolean, wp: Any) -> Any:
        bodies = [self._bodies[ref] for ref in op.body_refs if ref in self._bodies]
        if len(bodies) < 2:
            raise ValueError("Boolean requires at least 2 bodies")
        result = bodies[0]
        for b in bodies[1:]:
            if op.operation == "union":
                result = result.union(b)
            elif op.operation == "subtract":
                result = result.cut(b)
            elif op.operation == "intersect":
                result = result.intersect(b)
        self._bodies[op.id] = result
        return result

    def _op_mirror(self, op: Mirror, wp: Any) -> Any:
        body = self._bodies.get(op.body_ref, wp)
        plane_str = {"XY": "XY", "XZ": "XZ", "YZ": "YZ"}.get(op.plane, "YZ")
        result = body.mirror(plane_str)
        if op.keep_original:
            result = body.union(result)
        self._bodies[op.id] = result
        return result

    def _get_properties(self, wp: Any) -> dict:
        try:
            solid = wp.val()
            bb = solid.BoundingBox()
            return {
                "volume_mm3": round(solid.Volume(), 2),
                "surface_mm2": round(solid.Area(), 2),
                "bbox": {
                    "x_min": round(bb.xmin, 2), "x_max": round(bb.xmax, 2),
                    "y_min": round(bb.ymin, 2), "y_max": round(bb.ymax, 2),
                    "z_min": round(bb.zmin, 2), "z_max": round(bb.zmax, 2),
                },
                "center_of_mass": {
                    "x": round(solid.Center().x, 2),
                    "y": round(solid.Center().y, 2),
                    "z": round(solid.Center().z, 2),
                },
            }
        except Exception:
            return {}

    def _export_stl(self, wp: Any, path: Path) -> str:
        try:
            cq.exporters.export(wp, str(path), exportType="STL")
            return str(path)
        except Exception as e:
            log.warning("[cad_engine] STL export failed: %s", e)
            return ""

    def _export_step(self, wp: Any, path: Path) -> str:
        try:
            cq.exporters.export(wp, str(path), exportType="STEP")
            return str(path)
        except Exception as e:
            log.warning("[cad_engine] STEP export failed: %s", e)
            return ""

    def _export_obj(self, wp: Any, path: Path) -> str:
        try:
            stl_path = path.with_suffix(".stl")
            if not stl_path.exists():
                cq.exporters.export(wp, str(stl_path), exportType="STL")
            self._stl_to_obj(stl_path, path)
            return str(path)
        except Exception as e:
            log.warning("[cad_engine] OBJ export failed: %s", e)
            return ""

    @staticmethod
    def _stl_to_obj(stl_path: Path, obj_path: Path) -> None:
        """Convert binary/ASCII STL to OBJ (simple vertex/face conversion)."""
        import struct

        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        vert_map: dict[tuple[float, float, float], int] = {}

        data = stl_path.read_bytes()
        is_ascii = data[:5] == b"solid" and b"\n" in data[:80]

        if is_ascii:
            text = data.decode("utf-8", errors="ignore")
            import re
            for match in re.finditer(
                r"vertex\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)", text
            ):
                v = (float(match.group(1)), float(match.group(2)), float(match.group(3)))
                if v not in vert_map:
                    vert_map[v] = len(vertices) + 1
                    vertices.append(v)
            tri_verts = re.findall(
                r"vertex\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)", text
            )
            for i in range(0, len(tri_verts), 3):
                v1 = (float(tri_verts[i][0]), float(tri_verts[i][1]), float(tri_verts[i][2]))
                v2 = (float(tri_verts[i+1][0]), float(tri_verts[i+1][1]), float(tri_verts[i+1][2]))
                v3 = (float(tri_verts[i+2][0]), float(tri_verts[i+2][1]), float(tri_verts[i+2][2]))
                faces.append((vert_map[v1], vert_map[v2], vert_map[v3]))
        else:
            num_triangles = struct.unpack_from("<I", data, 80)[0]
            offset = 84
            for _ in range(num_triangles):
                offset += 12  # skip normal
                tri_indices = []
                for _ in range(3):
                    vx, vy, vz = struct.unpack_from("<fff", data, offset)
                    offset += 12
                    v = (round(vx, 6), round(vy, 6), round(vz, 6))
                    if v not in vert_map:
                        vert_map[v] = len(vertices) + 1
                        vertices.append(v)
                    tri_indices.append(vert_map[v])
                faces.append(tuple(tri_indices))
                offset += 2  # attribute byte count

        with open(obj_path, "w") as f:
            f.write("# Generated by ORYND CadEngine\n")
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    def _dry_run(self, ops: list[CoreOp], session_id: str) -> CadResult:
        """When CadQuery unavailable — describe what would be built."""
        description = []
        for op in ops:
            description.append(f"{op.op}(id={op.id})")

        return CadResult(
            ok=True,
            dry_run=True,
            properties={
                "description": f"Would execute {len(ops)} operations: {', '.join(description)}",
                "operations": [op.op for op in ops],
            },
            operations_executed=len(ops),
            error="CadQuery not installed — dry run only. Install: pip install cadquery",
        )
