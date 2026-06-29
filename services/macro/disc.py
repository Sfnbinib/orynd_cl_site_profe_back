"""Procedural brake-disc (rotor) generator → CoreOps.

Demonstrates the "complex object as a composed sequence of operations" idea (MechAgent-style):
  circle by size → extrude to thickness → centre bore → ring of bolt holes.
All native CoreOps the CadEngine already supports (CreateSketch circle → Extrude → CutHole×N).
A real vented/two-piece rotor is a later upgrade; this builds a recognisable, printable rotor.
"""
from __future__ import annotations

import math
import re as _re


def disc_coreops(
    diameter: float = 120.0,
    thickness: float = 10.0,
    bore: float = 30.0,
    bolts: int = 5,
    bolt_circle: float = 70.0,
    bolt_dia: float = 8.0,
) -> tuple[list[dict], dict]:
    """Return (coreops_operations, info) for a brake disc / rotor."""
    R = max(10.0, float(diameter)) / 2.0
    th = max(1.0, float(thickness))
    bolts = max(0, min(int(bolts), 24))

    ops: list[dict] = [
        {"op": "CreateSketch", "id": "dsk", "plane": "XY", "offset": 0.0,
         "shapes": [{"type": "circle", "center": {"x": 0.0, "y": 0.0}, "radius": R}]},
        {"op": "Extrude", "id": "rotor", "sketch_ref": "dsk", "height": th},
    ]
    if bore and bore > 0:
        ops.append({"op": "CutHole", "id": "bore", "center": {"x": 0.0, "y": 0.0},
                    "radius": float(bore) / 2.0, "through": True, "on_face": "top"})
    # ring of bolt holes on the bolt circle
    bc_r = float(bolt_circle) / 2.0
    for i in range(bolts):
        ang = 2.0 * math.pi * i / bolts
        ops.append({"op": "CutHole", "id": f"bolt{i+1}",
                    "center": {"x": round(bc_r * math.cos(ang), 3), "y": round(bc_r * math.sin(ang), 3)},
                    "radius": float(bolt_dia) / 2.0, "through": True, "on_face": "top"})

    info = {
        "diameter": round(2 * R, 1), "thickness": round(th, 1),
        "bore": round(float(bore), 1), "bolts": bolts,
        "bolt_circle": round(float(bolt_circle), 1), "bolt_dia": round(float(bolt_dia), 1),
    }
    return ops, info


_DIAM = _re.compile(r"\b(?:диаметр\w*|диам|diameter|d|Ø|ф)\s*=?\s*(\d+(?:\.\d+)?)", _re.I)
_THICK = _re.compile(r"\b(?:толщин\w*|thickness|h|высот\w*)\s*=?\s*(\d+(?:\.\d+)?)", _re.I)
_BOLTS = _re.compile(r"(\d+)\s*(?:болт\w*|отверст\w*|bolt\w*|hole\w*)", _re.I)


def parse_disc_params(text: str) -> dict:
    t = text or ""
    d = 120.0
    md = _DIAM.search(t)
    if md:
        d = float(md.group(1))
    th = 10.0
    mt = _THICK.search(t)
    if mt:
        th = float(mt.group(1))
    bolts = 5
    mb = _BOLTS.search(t)
    if mb:
        bolts = int(mb.group(1))
    return {
        "diameter": d, "thickness": th,
        "bore": max(10.0, d * 0.22), "bolts": bolts,
        "bolt_circle": d * 0.55, "bolt_dia": max(5.0, d * 0.06),
    }
