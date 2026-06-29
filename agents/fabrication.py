"""
FabricationAgent — analyzes a 3D model and generates a fabrication pack.

Phase 2: FDM pack (basic) + algorithm fallback.
Phase 5: full packs — CNC, Lathe, Laser, SLA, PCB, Supplier RFQ.

Input:  ctx.selected (dict with name, description, stl_url)
        ctx.extra["infill"]       (default 20)
        ctx.extra["printer"]      (default "prusa_mk4")
Output: ctx.extra["fabrication"] (dict with recommended_method, packs, notes)
"""

from __future__ import annotations
import json
import logging
import re

from orynd_core.agents.base import AgentContext, AgentResult, BaseAgent
from orynd_core.services.llm.base import LLMProvider

log = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a senior manufacturing engineer inside ORYND — an AI engineering workspace.
Given a 3D model with its properties, recommend the best fabrication approach.

Return ONLY valid JSON, no markdown, no explanation.

JSON schema:
{
  "recommended_method": "fdm",
  "alternative_methods": ["sla", "cnc"],
  "material": "PETG",
  "material_reason": "functional part with moderate loads, needs layer adhesion",
  "orientation_hint": "largest flat face down to minimize supports",
  "support_needed": false,
  "support_type": "tree|grid|none",
  "infill_pct": 20,
  "infill_pattern": "gyroid|grid|cubic|lightning",
  "wall_count": 3,
  "layer_height_mm": 0.2,
  "top_bottom_layers": 4,
  "estimated_print_time_min": 60,
  "estimated_material_g": 25,
  "estimated_cost_usd": 0.50,
  "critical_dimensions": ["hole diameters need -0.2mm compensation for FDM"],
  "post_processing": ["sand contact surfaces", "drill holes to final size"],
  "warnings": ["thin wall at 1.2mm may be fragile", "bridge span 30mm needs supports"],
  "notes": "concise manufacturing advice"
}

Decision rules:
- FDM: default for hobby/desktop, prototypes, enclosures, brackets. PLA for visual, PETG for mechanical, ABS for heat/outdoor, TPU for flexible.
- SLA: tiny parts (<30mm), high detail, jewelry, dental, miniatures. Always needs post-cure.
- CNC: precision metal, tight tolerances (<0.05mm), load-bearing, production runs.
- Lathe: axially symmetric parts ONLY (shafts, bushings, pulleys, knobs).
- Laser cut: flat sheet parts ONLY (panels, gaskets, stencils). Specify kerf compensation.
- Injection molding: suggest only for 500+ unit production runs.

Material selection:
- PLA: decorative, low-stress, indoor only. Easy to print.
- PETG: functional, moderate strength, chemical resistant. Standard for engineering.
- ABS: heat resistant (up to 100C), outdoor. Needs enclosure.
- ASA: outdoor UV resistant. ABS alternative.
- Nylon/PA: high wear resistance, living hinges, gears.
- TPU: flexible, vibration dampening, gaskets.
- PC: high impact, transparent options, high temp.

If CAD properties are provided (volume, bbox, surface area):
- Use volume to estimate material weight (density: PLA=1.24, PETG=1.27, ABS=1.04 g/cm3)
- Use bbox to estimate print time and orientation
- Flag thin features (<1.5mm wall for FDM)
"""


# ── Algorithm fallback ────────────────────────────────────────────────────────

_FUNCTIONAL_KEYWORDS = re.compile(
    r"\b(bracket|mount|holder|clamp|hinge|gear|bushing|shaft|bearing|clip|latch|hook|enclosure|case|housing)\b",
    re.IGNORECASE,
)
_LATHE_KEYWORDS = re.compile(
    r"\b(shaft|rod|axle|knob|bushing|cylinder|pulley|spool)\b", re.IGNORECASE
)
_LASER_KEYWORDS = re.compile(
    r"\b(flat|sheet|panel|plate|gasket|stencil)\b", re.IGNORECASE
)


def _algorithm_fabrication(name: str, description: str, infill: int, printer: str) -> dict:
    text = f"{name} {description}".lower()

    method = "fdm"
    if _LATHE_KEYWORDS.search(text):
        method = "lathe"
    elif _LASER_KEYWORDS.search(text):
        method = "laser"

    material = "PETG" if _FUNCTIONAL_KEYWORDS.search(text) else "PLA"

    return {
        "recommended_method": method,
        "material": material,
        "orientation_hint": "largest flat face down",
        "support_needed": any(w in text for w in ("overhang", "bridge", "arm", "canopy")),
        "infill_pct": infill,
        "wall_count": 3,
        "layer_height_mm": 0.2,
        "estimated_print_time_min": None,
        "notes": f"Algorithm analysis. Printer: {printer}. Material: {material} recommended for mechanical parts.",
        "_source": "algorithm",
    }


# ── Agent ─────────────────────────────────────────────────────────────────────

class FabricationAgent(BaseAgent):
    """
    Recommends fabrication method and generates a production pack.
    Uses LLM if available; falls back to deterministic algorithm.
    """

    name = "fabrication_agent"

    def __init__(self, provider: LLMProvider | None = None) -> None:
        super().__init__(provider=provider)

    async def run_logic(self, ctx: AgentContext) -> AgentResult:
        selected = ctx.selected or {}
        name = selected.get("name", ctx.raw_text or "unknown")
        description = selected.get("description", "")
        infill = int(ctx.extra.get("infill", 20))
        printer = ctx.extra.get("printer", "prusa_mk4")

        pack: dict = {}

        if self.provider:
            try:
                # Build detailed context including CAD properties if available
                cad_props = ctx.extra.get("cad", {}).get("properties", {})
                user_msg = (
                    f"Model name: {name}\n"
                    f"Description: {description[:400]}\n"
                    f"Target printer: {printer}\n"
                    f"Desired infill: {infill}%"
                )
                if cad_props:
                    bbox = cad_props.get("bbox", {})
                    user_msg += (
                        f"\n\nCAD Properties:"
                        f"\n  Volume: {cad_props.get('volume_mm3', 'unknown')} mm³"
                        f"\n  Surface area: {cad_props.get('surface_mm2', 'unknown')} mm²"
                        f"\n  Bounding box: {bbox.get('x_max', 0) - bbox.get('x_min', 0):.1f} x "
                        f"{bbox.get('y_max', 0) - bbox.get('y_min', 0):.1f} x "
                        f"{bbox.get('z_max', 0) - bbox.get('z_min', 0):.1f} mm"
                    )
                raw = await self.provider.complete(
                    system=_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
                # Strip markdown fences if present
                cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
                pack = json.loads(cleaned)
                pack["_source"] = "llm"
                log.info("[fabrication] LLM pack generated for '%s'", name)
            except Exception as e:
                log.warning("[fabrication] LLM failed (%s), using algorithm fallback", e)
                pack = _algorithm_fabrication(name, description, infill, printer)
        else:
            pack = _algorithm_fabrication(name, description, infill, printer)

        ctx.extra["fabrication"] = pack

        return AgentResult.success(
            self.name,
            {
                "recommended_method": pack.get("recommended_method"),
                "material": pack.get("material"),
                "source": pack.get("_source", "algorithm"),
            },
        )
