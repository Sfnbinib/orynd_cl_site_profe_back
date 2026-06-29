"""Standard Checker — validate CoreOps dimensions against standard sizes.

Checks (v1):
  * drill_hole radius  → standard drill Ø / fastener clearance hole
  * cylinder/bolt/shaft radius → metric fastener or bearing bore match
  * box dimensions     → print-bed sanity (informational)

Finding severity:
  ok      — matches a standard size (within tolerance)
  warn    — close to a standard size; suggest snapping
  info    — no standard applies / informational note
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .catalog import (
    BEARING_BORES,
    METRIC_SYSTEMS,
    STANDARD_DRILLS,
    SUPPORTED_SYSTEMS,
    nearest,
    nearest_fastener_for_clearance,
)

SNAP_TOLERANCE = 0.05   # ≤0.05mm → counts as the standard size
WARN_TOLERANCE = 0.5    # ≤0.5mm  → suggest snapping


@dataclass
class Finding:
    op_index: int
    op_type: str
    severity: str          # ok | warn | info
    message: str
    parameter: str = ""
    value: float | None = None
    nearest_standard: float | None = None
    designation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_index": self.op_index,
            "op_type": self.op_type,
            "severity": self.severity,
            "message": self.message,
            "parameter": self.parameter,
            "value": self.value,
            "nearest_standard": self.nearest_standard,
            "designation": self.designation,
        }


def _num(params: dict, *keys: str) -> float | None:
    for k in keys:
        v = params.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _check_hole(i: int, op_type: str, radius: float) -> Finding:
    diameter = radius * 2
    drill, d_delta = nearest(diameter, STANDARD_DRILLS)
    fastener, f_delta = nearest_fastener_for_clearance(diameter)

    if f_delta <= SNAP_TOLERANCE:
        return Finding(i, op_type, "ok",
                       f"Ø{diameter:g}mm is the {fastener.designation} clearance hole",
                       "radius", radius, fastener.clearance_hole / 2, fastener.designation)
    if d_delta <= SNAP_TOLERANCE:
        return Finding(i, op_type, "ok",
                       f"Ø{diameter:g}mm matches standard drill Ø{drill:g}mm",
                       "radius", radius, drill / 2)
    if f_delta <= WARN_TOLERANCE:
        return Finding(i, op_type, "warn",
                       f"Ø{diameter:g}mm is close to {fastener.designation} clearance "
                       f"(Ø{fastener.clearance_hole:g}mm) — snap for standard bolts",
                       "radius", radius, fastener.clearance_hole / 2, fastener.designation)
    if d_delta <= WARN_TOLERANCE:
        return Finding(i, op_type, "warn",
                       f"Ø{diameter:g}mm is close to standard drill Ø{drill:g}mm",
                       "radius", radius, drill / 2)
    return Finding(i, op_type, "info",
                   f"Ø{diameter:g}mm is non-standard (nearest drill Ø{drill:g}mm)",
                   "radius", radius, drill / 2)


def _check_shaft(i: int, op_type: str, radius: float) -> Finding:
    diameter = radius * 2
    bore, b_delta = nearest(diameter, BEARING_BORES)
    if b_delta <= SNAP_TOLERANCE:
        return Finding(i, op_type, "ok",
                       f"Ø{diameter:g}mm matches a standard bearing bore",
                       "radius", radius, bore / 2)
    if b_delta <= WARN_TOLERANCE:
        return Finding(i, op_type, "warn",
                       f"Ø{diameter:g}mm is close to bearing bore Ø{bore:g}mm — "
                       f"snap to fit standard bearings",
                       "radius", radius, bore / 2)
    return Finding(i, op_type, "info",
                   f"Ø{diameter:g}mm shaft (nearest bearing bore Ø{bore:g}mm)",
                   "radius", radius, bore / 2)


def check_operations(operations: list[dict], system: str = "ISO") -> dict[str, Any]:
    system = (system or "ISO").upper()
    if system not in SUPPORTED_SYSTEMS:
        raise ValueError(f"unknown system {system!r}; expected one of {sorted(SUPPORTED_SYSTEMS)}")

    findings: list[Finding] = []
    for i, op in enumerate(operations):
        op_type = str(op.get("primitive_type") or op.get("type") or "unknown")
        params = dict(op.get("parameters", {}) or {})

        if op_type == "drill_hole":
            radius = _num(params, "radius", "r")
            if radius:
                findings.append(_check_hole(i, op_type, radius))
        elif op_type in ("cylinder", "bolt", "shaft", "attach"):
            radius = _num(params, "radius", "outer_radius", "r")
            if radius:
                findings.append(_check_shaft(i, op_type, radius))
        elif op_type in ("box", "cube"):
            sx, sy, sz = (_num(params, "sx") or 0, _num(params, "sy") or 0, _num(params, "sz") or 0)
            big = max(sx, sy, sz)
            if big > 250:
                findings.append(Finding(i, op_type, "warn",
                                        f"{big:g}mm exceeds common 250mm print beds — consider splitting",
                                        "size", big))

    counts = {"ok": 0, "warn": 0, "info": 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1

    note = None
    if system == "ANSI":
        note = "ANSI inch tables not loaded in v1 — checked against metric equivalents"
    elif system in METRIC_SYSTEMS and system != "ISO":
        note = f"{system} checked via ISO metric tables (dimensionally equivalent subset)"

    return {
        "system": system,
        "checked_ops": len(operations),
        "findings": [f.to_dict() for f in findings],
        "counts": counts,
        "note": note,
    }
