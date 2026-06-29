"""
VisionAgent — photo → structured object description.

Uses LLM vision (Claude) if provider injected.
Algorithm fallback: returns generic description from image_caption if present.

Phase 6: also support local Moondream2 (Apple Silicon, ~0.3s, $0 cost).

Input:  ctx.image_b64, ctx.image_caption (optional)
Output: ctx.extra["vision"] (dict with object_name, description, tags, confidence)
        ctx.extra["vision_description"] (str, short English description for search)
"""

from __future__ import annotations
import json
import logging
import re

from orynd_core.agents.base import AgentContext, AgentResult, BaseAgent
from orynd_core.services.llm.base import LLMMessage, LLMProvider

log = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a computer vision + engineering analysis assistant for ORYND — an AI CAD workspace.
Analyze the provided image and return ONLY valid JSON, no markdown, no explanation.

Your job is to extract BOTH high-level object info AND low-level geometry observations.
Think like a mechanical engineer: what shapes do you see? Where are the holes? How would you CAD this?

JSON schema:
{
  "object_name": "concise English name of the main object",
  "object_category": "mechanical_part|accessory|tool|bracket|enclosure|decor|plate|housing|other",
  "description": "1-2 sentence description of what this object is and its function",
  "geometry": {
    "overall_shape": "rectangular|cylindrical|L-shaped|U-shaped|complex|flat_plate|organic",
    "estimated_size_mm": [120, 80, 15],
    "features": [
      {"type": "hole", "count": 4, "diameter_mm": 5, "pattern": "rectangular_grid|circular|random"},
      {"type": "slot", "count": 1, "width_mm": 3, "length_mm": 20},
      {"type": "fillet", "radius_mm": 2, "edges": "all_vertical|all|specific"},
      {"type": "chamfer", "distance_mm": 1},
      {"type": "boss", "count": 2, "diameter_mm": 8, "height_mm": 5},
      {"type": "rib", "count": 3, "thickness_mm": 2}
    ],
    "has_holes": true,
    "has_slots": false,
    "has_fillets": true,
    "symmetry": "bilateral|radial|none",
    "wall_thickness_mm": 2,
    "estimated_volume_mm3": null
  },
  "material_guess": "plastic|metal|wood|composite|unknown",
  "manufacturing_guess": "fdm_printed|injection_molded|cnc_machined|laser_cut|cast|unknown",
  "tags": ["tag1", "tag2", "tag3"],
  "search_query": "best 3-5 word query to find this on 3D model sites",
  "is_broken": false,
  "broken_details": "what is broken and where, if applicable",
  "engineering_notes": "observations about fit, tolerance, mounting, load path",
  "confidence": 0.85
}

Rules:
- object_name: lowercase, English
- estimated_size_mm: [width, height, depth] — use context clues (hand, table, coin) to estimate
- features: list ALL geometric features you can see. Even if count/size is approximate, include them.
- If something is broken, describe WHERE and WHAT specifically in broken_details
- search_query should work well on Thingiverse/Printables/GrabCAD
- engineering_notes: mention any concerns (thin walls, overhang, snap-fit, tolerance needs)
"""


# ── Algorithm fallback ────────────────────────────────────────────────────────

def _algorithm_vision(caption: str | None) -> dict:
    """When no LLM — produce minimal result from caption if provided."""
    if caption:
        words = re.findall(r"\b\w{3,}\b", caption.lower())
        tags = words[:5]
        obj_name = " ".join(words[:2]) if words else "unknown object"
        search_q = " ".join(words[:4]) if words else caption[:40]
    else:
        tags = []
        obj_name = "unknown object"
        search_q = ""

    return {
        "object_name": obj_name,
        "object_category": "other",
        "description": caption or "No image description available.",
        "tags": tags,
        "search_query": search_q,
        "is_broken": False,
        "notes": "Algorithm fallback — no vision model available.",
        "confidence": 0.2 if not caption else 0.4,
        "_source": "algorithm",
    }


# ── Agent ─────────────────────────────────────────────────────────────────────

class VisionAgent(BaseAgent):
    """
    Analyzes an image and produces a structured object description.

    - If ctx.image_b64 is absent: skips gracefully (returns ok=True, no-op).
    - If provider is None: uses algorithm fallback from caption.
    - If provider has vision: calls LLM vision API.
    """

    name = "vision_agent"

    def __init__(self, provider: LLMProvider | None = None) -> None:
        super().__init__(provider=provider)

    async def run_logic(self, ctx: AgentContext) -> AgentResult:
        if not ctx.has_image():
            # No image — pass through silently
            return AgentResult.success(self.name, {"skipped": True, "reason": "no image"})

        vision_result: dict = {}

        if self.provider:
            try:
                caption = ctx.image_caption or "Describe this object for 3D printing search."
                messages = [
                    LLMMessage(role="user", content=caption, image_b64=ctx.image_b64)
                ]
                raw = await self.provider.complete_json(
                    messages, system=_SYSTEM, max_tokens=512
                )
                cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
                try:
                    vision_result = json.loads(cleaned)
                    vision_result["_source"] = "llm"
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                    if match:
                        vision_result = json.loads(match.group())
                        vision_result["_source"] = "llm"
                    else:
                        vision_result = _algorithm_vision(ctx.image_caption)
                        vision_result["_parse_error"] = cleaned[:100]

                log.info("[vision] LLM analyzed image → %s", vision_result.get("object_name"))
            except Exception as e:
                log.warning("[vision] LLM vision failed (%s), algorithm fallback", e)
                vision_result = _algorithm_vision(ctx.image_caption)
        else:
            vision_result = _algorithm_vision(ctx.image_caption)

        ctx.extra["vision"] = vision_result

        # If raw_text is empty but we have vision description, use it as search query
        if not ctx.raw_text and vision_result.get("search_query"):
            ctx.raw_text = vision_result["search_query"]

        # Also inject into intent if intent not yet set
        if not ctx.intent and vision_result.get("search_query"):
            ctx.intent = {
                "object_name": vision_result.get("object_name", ""),
                "keywords": vision_result.get("tags", []),
                "confidence": vision_result.get("confidence", 0.4),
                "_source": "vision_agent",
            }

        return AgentResult.success(
            self.name,
            {
                "object_name": vision_result.get("object_name"),
                "search_query": vision_result.get("search_query"),
                "confidence": vision_result.get("confidence"),
                "source": vision_result.get("_source", "algorithm"),
            },
        )
