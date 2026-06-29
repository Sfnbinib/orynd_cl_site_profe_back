"""Standard dimension tables (mm) — ISO metric base set.

v1 scope: metric fasteners, clearance/tapping drills, bearing bores.
ANSI/DIN/GOST map onto the same metric tables where dimensions coincide
(DIN 931≈ISO 4014 etc.); inch-based ANSI data is a later expansion —
the checker reports system coverage honestly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FastenerSpec:
    designation: str      # "M3"
    thread_diameter: float
    pitch: float          # coarse pitch
    clearance_hole: float # normal-fit clearance drill Ø (ISO 273 medium)
    tapping_drill: float  # for coarse thread


# ISO 273 clearance (medium) + ISO 262 coarse pitches
METRIC_FASTENERS: list[FastenerSpec] = [
    FastenerSpec("M2",   2.0, 0.40,  2.4,  1.6),
    FastenerSpec("M2.5", 2.5, 0.45,  2.9,  2.05),
    FastenerSpec("M3",   3.0, 0.50,  3.4,  2.5),
    FastenerSpec("M4",   4.0, 0.70,  4.5,  3.3),
    FastenerSpec("M5",   5.0, 0.80,  5.5,  4.2),
    FastenerSpec("M6",   6.0, 1.00,  6.6,  5.0),
    FastenerSpec("M8",   8.0, 1.25,  9.0,  6.8),
    FastenerSpec("M10", 10.0, 1.50, 11.0,  8.5),
    FastenerSpec("M12", 12.0, 1.75, 13.5, 10.2),
    FastenerSpec("M16", 16.0, 2.00, 17.5, 14.0),
    FastenerSpec("M20", 20.0, 2.50, 22.0, 17.5),
    FastenerSpec("M24", 24.0, 3.00, 26.0, 21.0),
]

# Standard bearing bore diameters (608, 625, 6000-series etc.)
BEARING_BORES: list[float] = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 25, 30, 35, 40]

# Preferred metric drill sizes (subset of ISO standard drill set)
STANDARD_DRILLS: list[float] = [
    1.0, 1.5, 2.0, 2.5, 3.0, 3.2, 3.4, 4.0, 4.2, 4.5, 5.0, 5.5,
    6.0, 6.5, 6.8, 7.0, 8.0, 8.5, 9.0, 10.0, 10.5, 11.0, 12.0, 13.0,
]

SUPPORTED_SYSTEMS = {"ISO", "DIN", "ANSI", "GOST"}

# Which systems the v1 metric tables genuinely cover
METRIC_SYSTEMS = {"ISO", "DIN", "GOST"}


def nearest(value: float, options: list[float]) -> tuple[float, float]:
    """Return (nearest_option, abs_delta)."""
    best = min(options, key=lambda o: abs(o - value))
    return best, abs(best - value)


def nearest_fastener_for_clearance(hole_diameter: float) -> tuple[FastenerSpec, float]:
    best = min(METRIC_FASTENERS, key=lambda f: abs(f.clearance_hole - hole_diameter))
    return best, abs(best.clearance_hole - hole_diameter)
