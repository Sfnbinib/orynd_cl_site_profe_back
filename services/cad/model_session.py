"""Per-session CoreOps model document — the accumulated build history of ONE
workspace model.

Until now every build was stateless: it overwrote the session's part.stl and the
viewport wiped the scene, so only ONE object could ever exist and tools/sketches
had nothing to modify. This module keeps the running list of primitive ops per
session so new ops build ONTO the existing model (multi-object, sketch-on-face,
ribbon tools) instead of replacing it.

In-memory and process-local — mirrors the in-memory nature of CadEngine. A real
deployment would persist this per workspace, but the shape stays the same.
"""
from __future__ import annotations

import threading

# session_id -> list of primitive-dialect ops ({"type"/"op", "parameters"/"params", ...})
_DOCS: dict[str, list[dict]] = {}
# session_id -> list of native CoreOps MODIFIERS (Fillet/Chamfer/…) applied AFTER the
# primitives are built. Kept separate because the primitive→CoreOps adapter skips
# native ops mixed with primitives, so modifiers are re-appended on every rebuild.
_MODS: dict[str, list[dict]] = {}
_LOCK = threading.Lock()


def get_ops(session_id: str) -> list[dict]:
    with _LOCK:
        return [dict(o) for o in _DOCS.get(session_id, [])]


def append_ops(session_id: str, new_ops: list[dict]) -> list[dict]:
    """Append ops to the session document, return the FULL accumulated list."""
    with _LOCK:
        ops = _DOCS.setdefault(session_id, [])
        ops.extend(dict(o) for o in new_ops)
        return [dict(o) for o in ops]


def set_ops(session_id: str, ops: list[dict]) -> None:
    with _LOCK:
        _DOCS[session_id] = [dict(o) for o in ops]


def count(session_id: str) -> int:
    with _LOCK:
        return len(_DOCS.get(session_id, []))


def get_mods(session_id: str) -> list[dict]:
    with _LOCK:
        return [dict(m) for m in _MODS.get(session_id, [])]


def append_mod(session_id: str, mod: dict) -> list[dict]:
    """Append a native CoreOps modifier (Fillet/Chamfer/…) applied after build."""
    with _LOCK:
        mods = _MODS.setdefault(session_id, [])
        mods.append(dict(mod))
        return [dict(m) for m in mods]


def clear(session_id: str) -> None:
    with _LOCK:
        _DOCS.pop(session_id, None)
        _MODS.pop(session_id, None)


def _prim_footprint(op: dict) -> float:
    """Rough XY size of a primitive — used to space objects so they don't overlap."""
    p = op.get("parameters") or op.get("params") or {}
    kind = op.get("type") or op.get("op")
    if kind in ("box", "cube"):
        return float(p.get("sx", p.get("size", [40])[0] if isinstance(p.get("size"), list) else 40) or 40)
    if kind in ("cylinder",):
        return float(p.get("radius", 20)) * 2.0
    return 40.0


def place_next(session_id: str, op: dict, gap: float = 12.0) -> dict:
    """Return a copy of `op` shifted in +X so it sits NEXT TO existing objects
    instead of on top of them. Used for 'build another one' style appends."""
    existing = get_ops(session_id)
    if not existing:
        return dict(op)
    # cursor = right edge of the last placed object + gap + half of the new one
    cursor = 0.0
    for e in existing:
        cursor += _prim_footprint(e) + gap
    op = dict(op)
    p = dict(op.get("parameters") or op.get("params") or {})
    p["_offset_x"] = cursor + _prim_footprint(op) / 2.0
    if "parameters" in op:
        op["parameters"] = p
    else:
        op["params"] = p
    return op
