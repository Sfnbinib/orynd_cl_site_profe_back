"""Procedural spur-gear generator → CoreOps.

The local 3b model cannot compute a gear tooth profile, and a gear is not a
primitive — so "сделай шестерню" used to either hallucinate a box+hole or get an
honest refusal. This module generates a REAL parametric spur gear as a closed
polygon sketch + extrude + центральное отверстие, expressed in native CoreOps
(CreateSketch(polygon) → Extrude → CutHole). The CadEngine already supports all three.

Profile is a simplified (trapezoidal-tooth) spur gear — recognisable and printable,
not a true involute. Involute refinement is a later upgrade; this unblocks the
"build me a gear" path today with correct, deterministic geometry.
"""
from __future__ import annotations

import math


def gear_coreops(
    teeth: int = 18,
    module: float = 2.0,
    thickness: float = 8.0,
    bore: float = 6.0,
) -> tuple[list[dict], dict]:
    """Return (coreops_operations, info) for a spur gear.

    teeth     — number of teeth (z), clamped to a sane range
    module    — gear module m (mm); pitch diameter = m * z
    thickness — extrude height (mm)
    bore      — central hole diameter (mm); 0 = no bore
    """
    z = max(6, min(int(teeth), 200))
    m = max(0.4, float(module))
    th = max(1.0, float(thickness))

    r_pitch = m * z / 2.0          # pitch radius
    r_add = r_pitch + m            # addendum (tip) radius
    r_ded = r_pitch - 1.25 * m     # dedendum (root) radius
    r_ded = max(r_ded, m)          # keep positive

    # Build a closed polygon: per tooth, walk root → tip → tip → root within the
    # tooth's angular slot, leaving a root-gap before the next tooth.
    pts: list[dict] = []
    step = 2.0 * math.pi / z
    # fractions of the slot: tip occupies the middle ~40%, roots the edges
    profile = [(r_ded, 0.00), (r_add, 0.22), (r_add, 0.42), (r_ded, 0.64)]
    for i in range(z):
        base = i * step
        for radius, frac in profile:
            ang = base + step * frac
            pts.append({"x": round(radius * math.cos(ang), 4),
                        "y": round(radius * math.sin(ang), 4)})

    ops: list[dict] = [
        {"op": "CreateSketch", "id": "gsk", "plane": "XY", "offset": 0.0,
         "shapes": [{"type": "polygon", "points": pts}]},
        {"op": "Extrude", "id": "gear", "sketch_ref": "gsk", "height": th},
    ]
    if bore and bore > 0:
        ops.append({"op": "CutHole", "id": "bore", "center": {"x": 0.0, "y": 0.0},
                    "radius": float(bore) / 2.0, "through": True, "on_face": "top"})

    info = {
        "teeth": z, "module": round(m, 3), "thickness": round(th, 2),
        "bore": round(float(bore), 2),
        "pitch_diameter": round(2 * r_pitch, 2),
        "outer_diameter": round(2 * r_add, 2),
    }
    return ops, info


# Crude param extraction from free text: "шестерню на 20 зубьев модуль 2",
# "gear z=24 m=1.5", "шестерёнка 30 зубьев". Missing params → sensible defaults.
import re as _re

_TEETH = _re.compile(r"(\d+)\s*(?:зуб\w*|teeth|t\b|z\s*=?\s*\d+)", _re.I)
_TEETH2 = _re.compile(r"\b(?:z|зуб\w*)\s*=?\s*(\d+)", _re.I)
_MODULE = _re.compile(r"\b(?:модул\w*|module|m)\s*=?\s*(\d+(?:\.\d+)?)", _re.I)
_THICK = _re.compile(r"\b(?:толщин\w*|thickness|h|высот\w*)\s*=?\s*(\d+(?:\.\d+)?)", _re.I)
_RADIUS = _re.compile(r"\b(?:радиус\w*|диаметр\w*|\br\b|\bd\b)\s*=?\s*(\d+(?:\.\d+)?)", _re.I)


def parse_gear_params(text: str) -> dict:
    t = text or ""
    teeth = 18
    mt = _TEETH.search(t) or _TEETH2.search(t)
    if mt:
        teeth = int(mt.group(1))
    module = 2.0
    mm = _MODULE.search(t)
    if mm:
        module = float(mm.group(1))
    else:
        # If a radius/diameter is given, derive module so the outer size matches it.
        mr = _RADIUS.search(t)
        if mr:
            target_r = float(mr.group(1))
            if _re.search(r"диаметр", t, _re.I):
                target_r /= 2.0
            module = max(0.4, round(2.0 * target_r / (teeth + 2), 3))
    thickness = 8.0
    mh = _THICK.search(t)
    if mh:
        thickness = float(mh.group(1))
    return {"teeth": teeth, "module": module, "thickness": thickness, "bore": max(4.0, module * teeth / 6.0)}
