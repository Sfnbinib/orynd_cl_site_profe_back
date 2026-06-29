"""
AI Model 4 → CADAgent translator.

AI Model 4 emits primitive-level operations (box / cylinder / sphere / plane /
cone / torus / mesh). CADAgent's CoreOpsDocument expects manufacturing-level
operations (CreateSketch / Extrude / Cut / Boolean / ...).

This module bridges the two formats for Phase-1 wire-up.

Scope (Phase 1):
    ✅ box       → CreateSketch(rect)   + Extrude          (axis-aligned)
    ✅ cylinder  → CreateSketch(circle) + Extrude          (Z-axis only)
    ✅ plane     → CreateSketch(rect)   + Extrude (slab)   (Z-normal only)
    ⏳ sphere    → skip (needs sketch arcs not in schema)
    ⏳ cone      → skip (Phase 2: Revolve + sketch arc)
    ⏳ torus     → skip (Phase 2: Loft / sweep)
    ⏳ mesh      → skip (free-form, no primitive equivalent)

Non-axis-aligned primitives are skipped with a recorded reason. The current
schema does not support arbitrary rotation, so we drop them rather than
silently emit wrong geometry. Phase 2 will add a `Transform` op or sketch
rotation to lift this limit.

If more than one body survives, a final Boolean(union) merges them so the
exported STEP/STL is a single solid.
"""
from __future__ import annotations

import math
from typing import Any

# How close to a world axis a normal must be to qualify as "axis-aligned".
AXIS_ALIGN_DOT_THRESHOLD = 0.95

# Minimum thickness we extrude a fitted "plane" as. Real planes have zero
# height, but exporters need a solid — 0.5mm is thin enough to look like a
# face while still being a valid body.
PLANE_SLAB_HEIGHT_MM = 0.5

# Smallest real dimension (mm). AI Model 4 sometimes fits degenerate primitives
# with a near-zero axis (e.g. size [x, y, 5e-16]); extruding those makes OCCT
# throw BRepSweep_Translation::Constructor. Drop them before assembly so the
# build proceeds on valid geometry only.
MIN_DIM_MM = 0.1


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _axis_aligned_z(axis: list[float] | None) -> bool:
    """True when `axis` points mostly along +Z or -Z."""
    if not axis or len(axis) != 3:
        return False
    n = _norm(axis)
    if n < 1e-9:
        return False
    z_dot = abs(axis[2] / n)
    return z_dot >= AXIS_ALIGN_DOT_THRESHOLD


def _box_is_axis_aligned(axes: list[list[float]] | None) -> bool:
    """True when the box's 3 axes are close to world (X, Y, Z) — any sign."""
    if not axes or len(axes) != 3:
        return False
    world = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for ax in axes:
        n = _norm(ax)
        if n < 1e-9:
            return False
        unit = [c / n for c in ax]
        # Does this axis align with ANY world axis (up to sign)?
        if max(abs(_dot(unit, w)) for w in world) < AXIS_ALIGN_DOT_THRESHOLD:
            return False
    return True


def _next_id(prefix: str, counter: dict[str, int]) -> str:
    counter[prefix] = counter.get(prefix, 0) + 1
    return f"{prefix}{counter[prefix]}"


def _translate_box(
    op: dict,
    sketch_id: str,
    body_id: str,
) -> tuple[list[dict], str | None]:
    """Box → CreateSketch(rect, XY plane at z_bottom) + Extrude(size.z)."""
    params = op.get("params", {})
    transform = op.get("transform", {})
    size = params.get("size") or []
    center = transform.get("center") or [0.0, 0.0, 0.0]
    axes = transform.get("axes")

    if len(size) != 3 or len(center) != 3:
        return [], "box: missing size or center"
    if min(abs(float(s)) for s in size) < MIN_DIM_MM:
        return [], f"box: degenerate dimension {size} (< {MIN_DIM_MM}mm)"
    if not _box_is_axis_aligned(axes):
        return [], "box: non-axis-aligned (skipped in Phase 1)"

    w, h, d = size
    cx, cy, cz = center
    z_bottom = cz - d / 2.0

    return [
        {
            "op": "CreateSketch",
            "id": sketch_id,
            "plane": "XY",
            "offset": z_bottom,
            "shapes": [
                {"type": "rect", "center": {"x": cx, "y": cy}, "width": w, "height": h}
            ],
        },
        {
            "op": "Extrude",
            "id": body_id,
            "sketch_ref": sketch_id,
            "height": d,
        },
    ], None


def _translate_cylinder(
    op: dict,
    sketch_id: str,
    body_id: str,
) -> tuple[list[dict], str | None]:
    """Cylinder → CreateSketch(circle, XY) + Extrude(height). Z-axis only."""
    params = op.get("params", {})
    transform = op.get("transform", {})
    radius = params.get("radius")
    height = params.get("height")
    center = transform.get("center") or [0.0, 0.0, 0.0]
    axis = transform.get("axis")

    if radius is None or height is None:
        return [], "cylinder: missing radius or height"
    if len(center) != 3:
        return [], "cylinder: bad center"
    if float(radius) < MIN_DIM_MM / 2.0 or float(height) < MIN_DIM_MM:
        return [], f"cylinder: degenerate (r={radius}, h={height}, < {MIN_DIM_MM}mm)"
    if not _axis_aligned_z(axis):
        return [], "cylinder: non-Z axis (skipped in Phase 1)"

    cx, cy, cz = center
    z_bottom = cz - float(height) / 2.0

    return [
        {
            "op": "CreateSketch",
            "id": sketch_id,
            "plane": "XY",
            "offset": z_bottom,
            "shapes": [
                {"type": "circle", "center": {"x": cx, "y": cy}, "radius": float(radius)}
            ],
        },
        {
            "op": "Extrude",
            "id": body_id,
            "sketch_ref": sketch_id,
            "height": float(height),
        },
    ], None


def _translate_plane(
    op: dict,
    sketch_id: str,
    body_id: str,
) -> tuple[list[dict], str | None]:
    """Plane → thin slab (CreateSketch(rect) + small Extrude). Z-normal only."""
    params = op.get("params", {})
    transform = op.get("transform", {})
    normal = params.get("normal")
    extent = params.get("extent_uv") or []
    center = transform.get("center") or [0.0, 0.0, 0.0]

    if not _axis_aligned_z(normal):
        return [], "plane: non-Z normal (skipped in Phase 1)"
    if len(extent) != 2 or len(center) != 3:
        return [], "plane: missing extent_uv or center"
    if min(abs(float(e)) for e in extent) < MIN_DIM_MM:
        return [], f"plane: degenerate extent {extent} (< {MIN_DIM_MM}mm)"

    u, v = extent
    cx, cy, cz = center

    return [
        {
            "op": "CreateSketch",
            "id": sketch_id,
            "plane": "XY",
            "offset": cz - PLANE_SLAB_HEIGHT_MM / 2.0,
            "shapes": [
                {
                    "type": "rect",
                    "center": {"x": cx, "y": cy},
                    "width": float(u),
                    "height": float(v),
                }
            ],
        },
        {
            "op": "Extrude",
            "id": body_id,
            "sketch_ref": sketch_id,
            "height": PLANE_SLAB_HEIGHT_MM,
        },
    ], None


# Primitives we cannot represent in the current CoreOps schema yet.
# Each becomes a `skipped` note rather than failing the whole translation.
UNSUPPORTED_PRIMITIVES = {
    "sphere": "sphere: needs sketch-arc + Revolve (Phase 2)",
    "cone":   "cone: needs sketch-arc + Revolve (Phase 2)",
    "torus":  "torus: needs Loft/sweep (Phase 2)",
    "mesh":   "mesh: free-form, no primitive equivalent",
}


def translate_to_cad_coreops(ai_coreops: dict[str, Any]) -> dict[str, Any]:
    """
    Convert AI Model 4 primitive ops to CADAgent CoreOps.

    Returns a dict with:
        units, operations           — ready for CoreOpsDocument
        meta.translation_notes      — list of skip reasons
        meta.bodies_built           — count of bodies emitted
        meta.source_op_count        — primitives received
    """
    ops_in = ai_coreops.get("operations") or []
    ops_out: list[dict] = []
    counter: dict[str, int] = {}
    body_ids: list[str] = []
    notes: list[str] = []

    for in_op in ops_in:
        kind = in_op.get("op")
        if kind in UNSUPPORTED_PRIMITIVES:
            notes.append(UNSUPPORTED_PRIMITIVES[kind])
            continue

        sketch_id = _next_id("sketch", counter)
        body_id = _next_id("body", counter)

        if kind == "box":
            new_ops, err = _translate_box(in_op, sketch_id, body_id)
        elif kind == "cylinder":
            new_ops, err = _translate_cylinder(in_op, sketch_id, body_id)
        elif kind == "plane":
            new_ops, err = _translate_plane(in_op, sketch_id, body_id)
        else:
            new_ops, err = [], f"{kind}: unknown primitive"

        if err:
            notes.append(err)
            # Roll back the id counters we reserved so nothing references them.
            counter["sketch"] -= 1
            counter["body"] -= 1
            continue

        ops_out.extend(new_ops)
        body_ids.append(body_id)

    # Union everything if there are 2+ bodies — exporter wants one solid.
    if len(body_ids) >= 2:
        ops_out.append({
            "op": "Boolean",
            "id": _next_id("body", counter),
            "operation": "union",
            "body_refs": body_ids,
        })

    return {
        "units": "mm",
        "operations": ops_out,
        "meta": {
            "translation_notes": notes,
            "bodies_built": len(body_ids),
            "source_op_count": len(ops_in),
            "skipped_count": len(notes),
        },
    }
