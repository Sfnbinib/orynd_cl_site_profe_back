"""Workspace-level CAD document for semantic objects and simple assemblies.

Chat sessions are conversation threads. A CAD workspace is the engineering
document. This module keeps the document keyed by workspace_id and compiles
semantic objects back into the existing CoreOps pipeline.
"""
from __future__ import annotations

import copy
import threading
from typing import Any


_DOCS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _empty_doc(workspace_id: str) -> dict:
    return {
        "workspace_id": workspace_id,
        "objects": [],
        "constraints": [],
        "selection": None,
        "history": [],
        "last_result": None,
    }


def get_document(workspace_id: str) -> dict:
    with _LOCK:
        doc = _DOCS.setdefault(workspace_id, _empty_doc(workspace_id))
        return copy.deepcopy(doc)


def clear_document(workspace_id: str) -> None:
    with _LOCK:
        _DOCS[workspace_id] = _empty_doc(workspace_id)


def _counter_for(doc: dict, kind: str) -> int:
    prefix = _kind_prefix(kind)
    return sum(1 for obj in doc["objects"] if str(obj.get("id", "")).startswith(prefix + "_")) + 1


def _kind_prefix(kind: str) -> str:
    return {
        "spur_gear": "gear",
        "brake_disc": "disc",
        "box": "box",
        "cube": "box",
        "cylinder": "cyl",
    }.get(kind, kind.replace("-", "_"))


def _object_radius(obj: dict) -> float:
    kind = obj.get("kind")
    params = obj.get("params") or {}
    if kind == "spur_gear":
        return float(params.get("module", 2.0)) * float(params.get("teeth", 18)) / 2.0 + float(params.get("module", 2.0))
    if kind == "brake_disc":
        return float(params.get("diameter", 120.0)) / 2.0
    if kind in ("box", "cube"):
        return max(float(params.get("sx", 20.0)), float(params.get("sy", 20.0))) / 2.0
    if kind == "cylinder":
        return float(params.get("radius", 5.0))
    return 20.0


def _next_position(doc: dict, obj: dict, gap: float = 12.0) -> list[float]:
    objects = doc.get("objects") or []
    if not objects:
        return [0.0, 0.0, 0.0]
    right_edge = 0.0
    for existing in objects:
        pos = (existing.get("transform") or {}).get("position") or [0.0, 0.0, 0.0]
        right_edge = max(right_edge, float(pos[0]) + _object_radius(existing))
    return [right_edge + gap + _object_radius(obj), 0.0, 0.0]


def _connectors_for(kind: str, params: dict, position: list[float]) -> dict:
    axis = {"origin": position[:], "direction": [0.0, 0.0, 1.0]}
    if kind == "spur_gear":
        module = float(params.get("module", 2.0))
        teeth = int(params.get("teeth", 18))
        return {
            "axis_z": axis,
            "pitch_circle": {"radius": module * teeth / 2.0},
            "bore": {"diameter": float(params.get("bore", max(4.0, module * teeth / 6.0)))},
        }
    if kind == "brake_disc":
        return {
            "axis_z": axis,
            "bore": {"diameter": float(params.get("bore", 30.0))},
            "bolt_circle": {"diameter": float(params.get("bolt_circle", 70.0))},
        }
    if kind == "cylinder":
        return {"axis_z": axis}
    return {}


def add_object(workspace_id: str, kind: str, params: dict, *, mode: str = "append") -> dict:
    with _LOCK:
        doc = _DOCS.setdefault(workspace_id, _empty_doc(workspace_id))
        if mode == "replace":
            doc.clear()
            doc.update(_empty_doc(workspace_id))

        obj = {
            "id": f"{_kind_prefix(kind)}_{_counter_for(doc, kind)}",
            "kind": kind,
            "params": copy.deepcopy(params),
            "transform": {
                "position": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
            "connectors": {},
        }
        obj["transform"]["position"] = _next_position(doc, obj)
        obj["connectors"] = _connectors_for(kind, obj["params"], obj["transform"]["position"])
        doc["objects"].append(obj)
        doc["selection"] = obj["id"]
        doc["history"].append({"action": "object.added", "object_id": obj["id"], "kind": kind})
        return copy.deepcopy(obj)


def _scaled_params(kind: str, params: dict, factor: float) -> dict:
    out = copy.deepcopy(params)
    if kind == "spur_gear":
        # Preserve module so a half-size follow-up gear can still mesh with the
        # previous one; halve tooth count to halve pitch diameter.
        out["teeth"] = max(6, int(round(float(out.get("teeth", 18)) * factor)))
        if "bore" in out:
            out["bore"] = max(4.0, float(out["bore"]) * factor)
        return out
    for key in (
        "sx", "sy", "sz", "radius", "height", "length", "outer_radius",
        "diameter", "thickness", "bore", "bolt_circle", "bolt_dia",
    ):
        if isinstance(out.get(key), (int, float)):
            out[key] = float(out[key]) * factor
    return out


def clone_last_object(workspace_id: str, *, scale: float = 1.0) -> dict | None:
    with _LOCK:
        doc = _DOCS.setdefault(workspace_id, _empty_doc(workspace_id))
        source = doc["objects"][-1] if doc["objects"] else None
        if not source:
            return None
        kind = source.get("kind")
        params = _scaled_params(kind, source.get("params") or {}, scale)
        obj = {
            "id": f"{_kind_prefix(kind)}_{_counter_for(doc, kind)}",
            "kind": kind,
            "params": params,
            "transform": {
                "position": [0.0, 0.0, 0.0],
                "rotation": copy.deepcopy((source.get("transform") or {}).get("rotation") or [0.0, 0.0, 0.0]),
                "scale": [1.0, 1.0, 1.0],
            },
            "connectors": {},
        }
        obj["transform"]["position"] = _next_position(doc, obj)
        obj["connectors"] = _connectors_for(kind, params, obj["transform"]["position"])
        doc["objects"].append(obj)
        doc["selection"] = obj["id"]
        doc["history"].append({
            "action": "object.cloned",
            "source_id": source.get("id"),
            "object_id": obj["id"],
            "scale": scale,
        })
        return copy.deepcopy(obj)


def replace_primitives(workspace_id: str, primitive_ops: list[dict]) -> dict:
    """Mirror primitive fast-path state into the semantic workspace document."""
    clear_document(workspace_id)
    with _LOCK:
        doc = _DOCS.setdefault(workspace_id, _empty_doc(workspace_id))
        for op in primitive_ops:
            kind = op.get("type") or op.get("primitive_type") or op.get("op")
            if kind == "cube":
                kind = "box"
            if kind not in {"box", "cylinder"}:
                continue
            params = copy.deepcopy(op.get("parameters") or op.get("params") or {})
            ox = float(params.pop("_offset_x", 0.0))
            obj = {
                "id": f"{_kind_prefix(kind)}_{_counter_for(doc, kind)}",
                "kind": kind,
                "params": params,
                "transform": {
                    "position": [ox, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                },
                "connectors": {},
            }
            obj["connectors"] = _connectors_for(kind, params, obj["transform"]["position"])
            doc["objects"].append(obj)
            doc["selection"] = obj["id"]
        doc["history"].append({"action": "primitives.synced", "count": len(doc["objects"])})
        return copy.deepcopy(doc)


def last_object(workspace_id: str, kind: str | None = None) -> dict | None:
    doc = get_document(workspace_id)
    for obj in reversed(doc.get("objects") or []):
        if kind is None or obj.get("kind") == kind:
            return obj
    return None


def _shift_point(point: dict, dx: float, dy: float) -> dict:
    return {"x": float(point.get("x", 0.0)) + dx, "y": float(point.get("y", 0.0)) + dy}


def _namespace_ops(ops: list[dict], obj: dict) -> tuple[list[dict], str | None]:
    """Apply object transform and unique IDs. Returns ops and final body id."""
    out: list[dict] = []
    id_map: dict[str, str] = {}
    pos = (obj.get("transform") or {}).get("position") or [0.0, 0.0, 0.0]
    dx, dy = float(pos[0]), float(pos[1])
    final_body_id: str | None = None

    for raw in ops:
        op = copy.deepcopy(raw)
        raw_id = op.get("id")
        if raw_id:
            op["id"] = f"{obj['id']}__{raw_id}"
            id_map[raw_id] = op["id"]
        if op.get("sketch_ref") in id_map:
            op["sketch_ref"] = id_map[op["sketch_ref"]]

        if op.get("op") == "CreateSketch":
            for shape in op.get("shapes") or []:
                if "center" in shape:
                    shape["center"] = _shift_point(shape["center"], dx, dy)
                if "points" in shape:
                    shape["points"] = [_shift_point(p, dx, dy) for p in shape["points"]]
        elif op.get("op") == "CutHole" and "center" in op:
            op["center"] = _shift_point(op["center"], dx, dy)

        if op.get("op") in {"Extrude", "Cut", "CutHole", "CutSlot", "Fillet", "Chamfer", "Revolve", "Loft", "Boolean", "Mirror"}:
            final_body_id = op.get("id") or final_body_id
        out.append(op)
    return out, final_body_id


def _primitive_ops(obj: dict) -> tuple[list[dict], str | None]:
    params = obj.get("params") or {}
    pos = (obj.get("transform") or {}).get("position") or [0.0, 0.0, 0.0]
    x, y = float(pos[0]), float(pos[1])
    kind = obj.get("kind")
    if kind == "box":
        sx = float(params.get("sx", 20.0))
        sy = float(params.get("sy", 20.0))
        sz = float(params.get("sz", 20.0))
        return [
            {"op": "CreateSketch", "id": f"{obj['id']}__sk", "plane": "XY", "offset": 0.0,
             "shapes": [{"type": "rect", "center": {"x": x, "y": y}, "width": sx, "height": sy}]},
            {"op": "Extrude", "id": f"{obj['id']}__body", "sketch_ref": f"{obj['id']}__sk", "height": sz},
        ], f"{obj['id']}__body"
    if kind == "cylinder":
        radius = float(params.get("radius", 5.0))
        height = float(params.get("height", 20.0))
        return [
            {"op": "CreateSketch", "id": f"{obj['id']}__sk", "plane": "XY", "offset": 0.0,
             "shapes": [{"type": "circle", "center": {"x": x, "y": y}, "radius": radius}]},
            {"op": "Extrude", "id": f"{obj['id']}__body", "sketch_ref": f"{obj['id']}__sk", "height": height},
        ], f"{obj['id']}__body"
    return [], None


def compile_to_coreops(workspace_id: str) -> tuple[list[dict], dict]:
    from orynd_core.services.macro.disc import disc_coreops
    from orynd_core.services.macro.gear import gear_coreops

    doc = get_document(workspace_id)
    operations: list[dict] = []
    body_refs: list[str] = []
    for obj in doc.get("objects") or []:
        kind = obj.get("kind")
        params = obj.get("params") or {}
        if kind == "spur_gear":
            ops, _info = gear_coreops(**params)
            compiled, body_id = _namespace_ops(ops, obj)
        elif kind == "brake_disc":
            ops, _info = disc_coreops(**params)
            compiled, body_id = _namespace_ops(ops, obj)
        elif kind in {"box", "cylinder"}:
            compiled, body_id = _primitive_ops(obj)
        else:
            compiled, body_id = [], None
        operations.extend(compiled)
        if body_id:
            body_refs.append(body_id)

    if len(body_refs) >= 2:
        operations.append({
            "op": "Boolean",
            "id": "workspace_union",
            "operation": "union",
            "body_refs": body_refs,
        })
    return operations, doc


def add_gear_mesh_constraint(workspace_id: str, object_a: str | None = None, object_b: str | None = None) -> tuple[dict | None, str | None]:
    with _LOCK:
        doc = _DOCS.setdefault(workspace_id, _empty_doc(workspace_id))
        gears = [o for o in doc["objects"] if o.get("kind") == "spur_gear"]
        if len(gears) < 2:
            return None, "Need at least two gears in the workspace."

        a = next((o for o in gears if o.get("id") == object_a), None) if object_a else gears[-2]
        b = next((o for o in gears if o.get("id") == object_b), None) if object_b else gears[-1]
        if not a or not b or a["id"] == b["id"]:
            return None, "Could not resolve two different gears."

        ma = float((a.get("params") or {}).get("module", 2.0))
        mb = float((b.get("params") or {}).get("module", 2.0))
        if abs(ma - mb) > 1e-6:
            return None, f"Module mismatch: {a['id']}={ma:g}, {b['id']}={mb:g}."

        ra = float((a.get("connectors") or {}).get("pitch_circle", {}).get("radius", ma * float((a.get("params") or {}).get("teeth", 18)) / 2.0))
        rb = float((b.get("connectors") or {}).get("pitch_circle", {}).get("radius", mb * float((b.get("params") or {}).get("teeth", 18)) / 2.0))
        apos = (a.get("transform") or {}).get("position") or [0.0, 0.0, 0.0]
        b["transform"]["position"] = [float(apos[0]) + ra + rb, float(apos[1]), float(apos[2])]
        b["connectors"] = _connectors_for("spur_gear", b.get("params") or {}, b["transform"]["position"])
        constraint = {
            "type": "gear_mesh",
            "a": a["id"],
            "b": b["id"],
            "center_distance": ra + rb,
            "ratio": f"{int((a.get('params') or {}).get('teeth', 0))}:{int((b.get('params') or {}).get('teeth', 0))}",
        }
        doc["constraints"].append(constraint)
        doc["history"].append({"action": "constraint.added", **constraint})
        doc["selection"] = b["id"]
        return copy.deepcopy(constraint), None


def add_align_axis_constraint(workspace_id: str, object_a: str | None = None, object_b: str | None = None) -> tuple[dict | None, str | None]:
    """Align two Z-axis connectors by translating object B onto object A in XY."""
    with _LOCK:
        doc = _DOCS.setdefault(workspace_id, _empty_doc(workspace_id))
        axis_objects = [o for o in doc["objects"] if "axis_z" in (o.get("connectors") or {})]
        if len(axis_objects) < 2:
            return None, "Need at least two objects with axes."

        if object_a:
            a = next((o for o in axis_objects if o.get("id") == object_a), None)
        else:
            a = next((o for o in axis_objects if o.get("kind") == "cylinder"), None) or axis_objects[-2]
        if object_b:
            b = next((o for o in axis_objects if o.get("id") == object_b), None)
        else:
            b = next((o for o in reversed(axis_objects) if o is not a and o.get("kind") == "spur_gear"), None) or axis_objects[-1]
        if not a or not b or a["id"] == b["id"]:
            return None, "Could not resolve two different axis objects."

        apos = (a.get("transform") or {}).get("position") or [0.0, 0.0, 0.0]
        bpos = (b.get("transform") or {}).get("position") or [0.0, 0.0, 0.0]
        b["transform"]["position"] = [float(apos[0]), float(apos[1]), float(bpos[2])]
        b["connectors"] = _connectors_for(b.get("kind"), b.get("params") or {}, b["transform"]["position"])
        constraint = {
            "type": "align_axis",
            "a": a["id"],
            "b": b["id"],
            "connector_a": "axis_z",
            "connector_b": "axis_z",
        }
        doc["constraints"].append(constraint)
        doc["history"].append({"action": "constraint.added", **constraint})
        doc["selection"] = b["id"]
        return copy.deepcopy(constraint), None
