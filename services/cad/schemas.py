"""
CoreOps JSON Schema — the ONLY way LLM controls CAD.

LLM generates CoreOps JSON → CADAgent validates via Pydantic → CadEngine executes.
LLM NEVER calls CadQuery directly.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class Point2D(BaseModel):
    x: float
    y: float


class Point3D(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class SketchRect(BaseModel):
    type: Literal["rect"] = "rect"
    center: Point2D = Field(default_factory=lambda: Point2D(x=0, y=0))
    width: float
    height: float


class SketchCircle(BaseModel):
    type: Literal["circle"] = "circle"
    center: Point2D = Field(default_factory=lambda: Point2D(x=0, y=0))
    radius: float


class SketchPolygon(BaseModel):
    type: Literal["polygon"] = "polygon"
    points: list[Point2D]


SketchShape = SketchRect | SketchCircle | SketchPolygon


class CreateSketch(BaseModel):
    op: Literal["CreateSketch"] = "CreateSketch"
    id: str
    plane: Literal["XY", "XZ", "YZ"] = "XY"
    offset: float = 0.0
    shapes: list[SketchShape]


class Extrude(BaseModel):
    op: Literal["Extrude"] = "Extrude"
    id: str
    sketch_ref: str
    height: float
    taper_angle: float = 0.0
    symmetric: bool = False


class Cut(BaseModel):
    op: Literal["Cut"] = "Cut"
    id: str
    sketch_ref: str
    depth: float
    through: bool = False


class CutHole(BaseModel):
    op: Literal["CutHole"] = "CutHole"
    id: str
    center: Point2D
    radius: float
    depth: float = 0.0
    through: bool = True
    on_face: str = "top"


class CutSlot(BaseModel):
    op: Literal["CutSlot"] = "CutSlot"
    id: str
    start: Point2D
    end: Point2D
    width: float
    depth: float


class Fillet(BaseModel):
    op: Literal["Fillet"] = "Fillet"
    id: str
    radius: float
    edges: list[str] = Field(default_factory=lambda: ["all"])


class Chamfer(BaseModel):
    op: Literal["Chamfer"] = "Chamfer"
    id: str
    distance: float
    edges: list[str] = Field(default_factory=lambda: ["all"])


class Revolve(BaseModel):
    op: Literal["Revolve"] = "Revolve"
    id: str
    sketch_ref: str
    axis: Literal["X", "Y", "Z"] = "Y"
    angle: float = 360.0


class Loft(BaseModel):
    op: Literal["Loft"] = "Loft"
    id: str
    sketch_refs: list[str]
    ruled: bool = False


class Boolean(BaseModel):
    op: Literal["Boolean"] = "Boolean"
    id: str
    operation: Literal["union", "subtract", "intersect"]
    body_refs: list[str]


class Mirror(BaseModel):
    op: Literal["Mirror"] = "Mirror"
    id: str
    body_ref: str
    plane: Literal["XY", "XZ", "YZ"] = "YZ"
    keep_original: bool = True


CoreOp = (
    CreateSketch | Extrude | Cut | CutHole | CutSlot |
    Fillet | Chamfer | Revolve | Loft | Boolean | Mirror
)

OP_REGISTRY: dict[str, type] = {
    "CreateSketch": CreateSketch,
    "Extrude": Extrude,
    "Cut": Cut,
    "CutHole": CutHole,
    "CutSlot": CutSlot,
    "Fillet": Fillet,
    "Chamfer": Chamfer,
    "Revolve": Revolve,
    "Loft": Loft,
    "Boolean": Boolean,
    "Mirror": Mirror,
}


class CoreOpsDocument(BaseModel):
    version: str = "1.0"
    units: Literal["mm", "inch"] = "mm"
    operations: list[dict] = Field(default_factory=list)

    def parse_operations(self) -> list[CoreOp]:
        parsed = []
        for raw in self.operations:
            op_type = raw.get("op")
            if op_type not in OP_REGISTRY:
                raise ValueError(f"Unknown operation: {op_type}")
            model_cls = OP_REGISTRY[op_type]
            parsed.append(model_cls.model_validate(raw))
        return parsed
