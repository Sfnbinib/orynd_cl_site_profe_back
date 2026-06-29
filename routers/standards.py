"""/standards/* — Standard Checker (ISO/DIN/ANSI/GOST).

Validates CoreOps operation dimensions against standard sizes
(fastener clearance holes, drills, bearing bores). Founder pick from
the MecAgent feature comparison.

* POST /standards/check    — {operations: [...], system?: "ISO"} → findings
* GET  /standards/catalog  — the dimension tables the checker uses
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from orynd_core.services.standards.catalog import (
    BEARING_BORES,
    METRIC_FASTENERS,
    STANDARD_DRILLS,
    SUPPORTED_SYSTEMS,
)
from orynd_core.services.standards.checker import check_operations

router = APIRouter(prefix="/standards", tags=["standards"])


@router.post("/check")
async def check(payload: dict = Body(...)) -> dict[str, Any]:
    operations = payload.get("operations")
    if not isinstance(operations, list):
        raise HTTPException(status_code=422, detail="'operations' must be a list")
    system = str(payload.get("system", "ISO"))
    try:
        return check_operations(operations, system=system)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/catalog")
async def catalog() -> dict[str, Any]:
    return {
        "systems": sorted(SUPPORTED_SYSTEMS),
        "metric_fasteners": [asdict(f) for f in METRIC_FASTENERS],
        "standard_drills": STANDARD_DRILLS,
        "bearing_bores": BEARING_BORES,
    }
