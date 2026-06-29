"""Seed theory patterns — built-in engineering best practices.

v1 theory store (blueprint 42/01): a small curated set so the comparison
engine works out of the box. Library articles become theory sources in a
later phase; callers can also pass their own patterns per request.
"""

from __future__ import annotations

from .engine import TheoryPattern

SEED_THEORIES: list[TheoryPattern] = [
    TheoryPattern(
        pattern_id="extrude-depth-ratio",
        action_type="cad_extrude",
        text="extrude depth should stay proportional to profile size; very deep "
             "thin extrusions warp in FDM printing and flex under load",
        recommended_params={"distance": [1, 50]},
        expected_outcome={"warp_risk": "low"},
    ),
    TheoryPattern(
        pattern_id="drill-standard-sizes",
        action_type="drill_hole",
        text="hole radii should match standard metric drill sizes; M3 bolts need "
             "1.6mm radius clearance holes, M4 needs 2.2mm, M5 needs 2.7mm",
        recommended_params={"radius": [1.0, 6.0]},
        expected_outcome={"fit": "clearance"},
    ),
    TheoryPattern(
        pattern_id="fillet-stress-relief",
        action_type="cad_fillet",
        text="inner corners need fillets for stress relief; radius 2-5mm covers "
             "most printed brackets, sharp inner corners crack first",
        recommended_params={"radius": [2, 5]},
        expected_outcome={"stress_concentration": "reduced"},
    ),
    TheoryPattern(
        pattern_id="wall-thickness-fdm",
        action_type="cad_shell",
        text="FDM wall thickness below 1.2mm (3 perimeters at 0.4 nozzle) is "
             "fragile; structural walls want 2.4mm or more",
        recommended_params={"thickness": [1.2, 8.0]},
        expected_outcome={"strength": "adequate"},
    ),
    TheoryPattern(
        pattern_id="cube-printable-size",
        action_type="box",
        text="boxes and plates print best with the largest face down; sizes "
             "under 250mm fit common print beds",
        recommended_params={"sx": [1, 250], "sy": [1, 250], "sz": [1, 250]},
        expected_outcome={"printable": True},
    ),
]
