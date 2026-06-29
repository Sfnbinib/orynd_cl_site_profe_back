"""Component catalog — seed parts for axes attachment.

Phase 0: hardcoded seed (~20 parts) per blueprint 99. Each part declares the
axis geometry it attaches to (diameter range, length) so the matcher can
filter candidates against a user-selected axis.

Phase 1+: catalogs loaded from Library (OPEN layer, skill type
"component_catalog") + community GitHub repos. This file stays the loader
entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

CatalogCategory = Literal[
    "suspension",
    "steering",
    "fastener",
    "wheel_axle",
    "electronics",
    "bracket",
]


@dataclass
class CatalogPart:
    """One catalog component that can attach to an axis."""

    part_id: str
    name: str
    category: CatalogCategory
    # Axis fit constraints (mm)
    axis_diameter_min: float
    axis_diameter_max: float
    axis_length_min: float = 0.0
    axis_length_max: float = 1e9
    # Default geometry to emit when attached (simplified primitive shape)
    primitive_type: str = "cylinder"
    default_parameters: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    description: str = ""

    def fits_axis(self, diameter: float, length: float) -> bool:
        return (
            self.axis_diameter_min <= diameter <= self.axis_diameter_max
            and self.axis_length_min <= length <= self.axis_length_max
        )

    def to_dict(self) -> dict:
        return {
            "part_id": self.part_id,
            "name": self.name,
            "category": self.category,
            "axis_diameter_range": [self.axis_diameter_min, self.axis_diameter_max],
            "axis_length_range": [self.axis_length_min, self.axis_length_max],
            "primitive_type": self.primitive_type,
            "default_parameters": dict(self.default_parameters),
            "tags": list(self.tags),
            "description": self.description,
        }


# ── Seed catalog (Phase 0) ──────────────────────────────────────────────

SEED_CATALOG: list[CatalogPart] = [
    # Fasteners — metric bolts (axis = shaft)
    CatalogPart("bolt-m3", "M3 Bolt", "fastener", 2.8, 3.2, 4, 40, "cylinder",
                {"radius": 1.5, "height": 16}, ["m3", "screw", "iso"], "Metric M3 hex bolt"),
    CatalogPart("bolt-m4", "M4 Bolt", "fastener", 3.8, 4.2, 5, 50, "cylinder",
                {"radius": 2.0, "height": 20}, ["m4", "screw"], "Metric M4 hex bolt"),
    CatalogPart("bolt-m5", "M5 Bolt", "fastener", 4.8, 5.2, 6, 60, "cylinder",
                {"radius": 2.5, "height": 25}, ["m5", "screw"], "Metric M5 hex bolt"),
    CatalogPart("bolt-m6", "M6 Bolt", "fastener", 5.8, 6.2, 8, 80, "cylinder",
                {"radius": 3.0, "height": 30}, ["m6", "screw"], "Metric M6 hex bolt"),
    CatalogPart("bolt-m8", "M8 Bolt", "fastener", 7.8, 8.2, 10, 100, "cylinder",
                {"radius": 4.0, "height": 40}, ["m8", "screw"], "Metric M8 hex bolt"),
    CatalogPart("nut-m3", "M3 Nut", "fastener", 2.8, 3.2, 0, 4, "cylinder",
                {"radius": 3.0, "height": 2.4}, ["m3", "nut"], "Metric M3 hex nut"),
    CatalogPart("nut-m5", "M5 Nut", "fastener", 4.8, 5.2, 0, 5, "cylinder",
                {"radius": 4.0, "height": 4.0}, ["m5", "nut"], "Metric M5 hex nut"),
    CatalogPart("washer-m5", "M5 Washer", "fastener", 5.0, 5.5, 0, 2, "cylinder",
                {"radius": 5.0, "height": 1.0}, ["m5", "washer"], "Metric M5 flat washer"),
    # Wheel / axle
    CatalogPart("bearing-608", "608 Bearing", "wheel_axle", 7.8, 8.2, 5, 9, "cylinder",
                {"radius": 11.0, "height": 7.0}, ["bearing", "608", "skate"], "608ZZ skate bearing (8mm bore)"),
    CatalogPart("bearing-625", "625 Bearing", "wheel_axle", 4.8, 5.2, 4, 7, "cylinder",
                {"radius": 8.0, "height": 5.0}, ["bearing", "625"], "625ZZ bearing (5mm bore)"),
    CatalogPart("hub-8mm", "8mm Wheel Hub", "wheel_axle", 7.8, 8.3, 10, 60, "cylinder",
                {"radius": 15.0, "height": 20.0}, ["hub", "wheel"], "Wheel hub for 8mm axle"),
    CatalogPart("flange-10mm", "10mm Flange", "wheel_axle", 9.8, 10.3, 3, 20, "cylinder",
                {"radius": 20.0, "height": 6.0}, ["flange"], "Mounting flange for 10mm shaft"),
    # Suspension
    CatalogPart("shock-absorber", "Shock Absorber", "suspension", 6.0, 12.0, 50, 250, "cylinder",
                {"radius": 8.0, "height": 120.0}, ["shock", "damper", "suspension"], "Adjustable coilover shock"),
    CatalogPart("rack-scooter", "Scooter Rack", "suspension", 8.0, 16.0, 20, 120, "box",
                {"sx": 40, "sy": 80, "sz": 8}, ["rack", "scooter"], "Adaptive scooter rack mount"),
    CatalogPart("control-arm", "Control Arm", "suspension", 8.0, 14.0, 60, 300, "box",
                {"sx": 30, "sy": 150, "sz": 12}, ["arm", "suspension"], "Lower control arm"),
    # Steering
    CatalogPart("steering-column", "Steering Column", "steering", 12.0, 25.0, 80, 400, "cylinder",
                {"radius": 12.0, "height": 200.0}, ["steering", "column"], "Telescoping steering column"),
    CatalogPart("tie-rod", "Tie Rod", "steering", 6.0, 12.0, 40, 200, "cylinder",
                {"radius": 5.0, "height": 100.0}, ["tie-rod", "steering"], "Adjustable tie rod end"),
    # Electronics
    CatalogPart("nema17", "NEMA17 Motor", "electronics", 4.8, 5.2, 15, 30, "box",
                {"sx": 42, "sy": 42, "sz": 40}, ["motor", "nema17", "stepper"], "NEMA17 stepper (5mm shaft)"),
    CatalogPart("nema23", "NEMA23 Motor", "electronics", 6.0, 6.5, 18, 35, "box",
                {"sx": 57, "sy": 57, "sz": 56}, ["motor", "nema23"], "NEMA23 stepper (6.35mm shaft)"),
    CatalogPart("servo-sg90", "SG90 Servo", "electronics", 3.8, 6.0, 5, 25, "box",
                {"sx": 23, "sy": 12, "sz": 29}, ["servo", "sg90"], "SG90 micro servo"),
    # Brackets
    CatalogPart("l-bracket", "L-Bracket", "bracket", 3.0, 10.0, 0, 100, "box",
                {"sx": 40, "sy": 40, "sz": 3}, ["bracket", "L", "corner"], "90° L mounting bracket"),
    CatalogPart("axle-clamp", "Axle Clamp", "bracket", 6.0, 20.0, 10, 60, "cylinder",
                {"radius": 14.0, "height": 25.0}, ["clamp", "axle"], "Split clamp for round shaft"),
]


def all_parts() -> list[CatalogPart]:
    return list(SEED_CATALOG)


def get_part(part_id: str) -> Optional[CatalogPart]:
    for p in SEED_CATALOG:
        if p.part_id == part_id:
            return p
    return None


def parts_by_category(category: str) -> list[CatalogPart]:
    return [p for p in SEED_CATALOG if p.category == category]


__all__ = [
    "CatalogCategory",
    "CatalogPart",
    "SEED_CATALOG",
    "all_parts",
    "get_part",
    "parts_by_category",
]
