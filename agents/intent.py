"""
IntentAgent — first agent in the search pipeline.

Input:  raw text or photo (in AgentContext)
Output: structured SearchIntent written to ctx.intent

Works with text or vision (photo → same output schema).
Uses whatever LLMProvider is injected — Claude, local, any.
"""
from __future__ import annotations
import json
import re

from .base import BaseAgent, AgentContext, AgentResult
from orynd_core.services.llm.base import LLMMessage

# ─────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────

_SYSTEM_TEXT = """
You are an intent extraction engine for ORYND — an AI engineering workspace.
The workspace can: search 3D models, build parts from scratch (CAD), analyze photos, research topics, recommend fabrication settings.

Analyze the user's request and return ONLY valid JSON, no markdown, no explanation.

Output schema:
{
  "intent_type": "search|create|modify|photo_to_3d|fabrication|research|question",
  "object_name": "short name of what they want (e.g. phone holder)",
  "object_type": "accessory|tool|part|decor|bracket|toy|replacement|enclosure|mechanical|other",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "dimensions": {"width": null, "height": null, "depth": null, "diameter": null, "units": "mm"},
  "mounting": "desk|wall|clamp|magnetic|screw|floor|null",
  "size_hint": "small|medium|large|null",
  "material_hint": "pla|petg|abs|resin|metal|wood|any|null",
  "confidence": 0.85
}

Intent type rules:
- "search" → user wants to FIND an existing 3D model (find, search, download, print, I need a...)
- "create" → user gives DIMENSIONS or says BUILD/CREATE/DESIGN/make me a box 50x30x10
- "modify" → user wants to CHANGE an existing model (add hole, make bigger, fillet edges)
- "photo_to_3d" → user uploaded a photo and wants a 3D model from it
- "fabrication" → user asks about PRINT SETTINGS, material choice, or how to manufacture
- "research" → user wants to UNDERSTAND something, compare options, find open-source solutions
- "question" → general question about engineering, 3D printing, or the workspace itself

Dimension extraction:
- If user says "50x30x10 mm" → dimensions: {width: 50, height: 30, depth: 10, units: "mm"}
- If user says "2 inch diameter" → dimensions: {diameter: 50.8, units: "mm"} (always convert to mm)
- If no dimensions mentioned → dimensions: all null

Keywords: 3–5 terms best for searching 3D model databases (English).
object_name: concise, English, lowercase.
If unclear, confidence < 0.6 and still return best guess.
""".strip()

_SYSTEM_VISION = """
You are an intent extraction engine for ORYND — an AI engineering workspace.
Analyze the image (and optional caption) and return ONLY valid JSON, no markdown, no explanation.

Output schema:
{
  "intent_type": "search|create|photo_to_3d|fabrication|question",
  "object_name": "what you see in the image",
  "object_type": "accessory|tool|part|decor|bracket|toy|replacement|enclosure|mechanical|other",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "geometry_hints": {
    "shape": "rectangular|cylindrical|complex|flat|organic",
    "has_holes": false,
    "has_slots": false,
    "estimated_size_mm": [0, 0, 0],
    "symmetry": "none|bilateral|radial"
  },
  "mounting": "desk|wall|clamp|magnetic|screw|floor|null",
  "size_hint": "small|medium|large|null",
  "material_hint": "pla|petg|abs|resin|metal|wood|any|null",
  "is_broken": false,
  "confidence": 0.85
}

Rules:
- intent_type: if photo shows a broken part → "photo_to_3d" (user wants replacement)
- intent_type: if photo shows a finished object and user asks to find similar → "search"
- geometry_hints: estimate shape, holes, slots, approximate size from visual cues
- estimated_size_mm: rough [width, height, depth] in mm based on context clues (hand, table, etc.)
- keywords: 3–5 terms for searching 3D model databases (English)
- object_name: concise, English, lowercase
""".strip()

# Fallback when LLM not available
_FALLBACK_INTENT = {
    "object_name": "3d model",
    "object_type": "other",
    "keywords": [],
    "mounting": None,
    "size_hint": None,
    "material_hint": None,
    "confidence": 0.0,
    "_source": "fallback",
}


class IntentAgent(BaseAgent):
    """
    Extracts structured intent from user input.
    Text input → keyword analysis.
    Image input → vision analysis.
    Both → fills ctx.intent dict.
    """

    name = "intent_agent"

    async def run_logic(self, ctx: AgentContext) -> AgentResult:
        if not ctx.has_text() and not ctx.has_image():
            return AgentResult.failure(self.name, "No input: provide text or image")

        if self.provider is None:
            # Algorithm fallback: extract keywords from raw text without LLM
            intent = self._extract_without_llm(ctx)
            ctx.intent = intent
            return AgentResult.success(self.name, {"intent": intent, "source": "algorithm"})

        try:
            intent = await self._extract_with_llm(ctx)
        except Exception as exc:
            # LLM failed → algorithm fallback, don't stop pipeline
            intent = self._extract_without_llm(ctx)
            intent["_llm_error"] = str(exc)

        ctx.intent = intent
        return AgentResult.success(self.name, {"intent": intent})

    # ─────────────────────────────────────────
    # LLM path
    # ─────────────────────────────────────────

    async def _extract_with_llm(self, ctx: AgentContext) -> dict:
        if ctx.has_image():
            return await self._vision(ctx)
        return await self._text(ctx)

    async def _text(self, ctx: AgentContext) -> dict:
        messages = [LLMMessage(role="user", content=ctx.raw_text or "")]
        raw = await self.provider.complete_json(messages, system=_SYSTEM_TEXT, max_tokens=512)
        return self._parse_json(raw, ctx)

    async def _vision(self, ctx: AgentContext) -> dict:
        caption = ctx.image_caption or ""
        messages = [
            LLMMessage(role="user", content=caption, image_b64=ctx.image_b64)
        ]
        raw = await self.provider.complete_json(messages, system=_SYSTEM_VISION, max_tokens=512)
        return self._parse_json(raw, ctx)

    def _parse_json(self, raw: str, ctx: AgentContext) -> dict:
        """Extract JSON from LLM response, robust to extra text."""
        # Strip markdown code blocks if present
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = self._extract_without_llm(ctx)
                data["_parse_error"] = raw[:100]
        return data

    # ─────────────────────────────────────────
    # Algorithm fallback (no LLM)
    # ─────────────────────────────────────────

    def _extract_without_llm(self, ctx: AgentContext) -> dict:
        text = (ctx.raw_text or ctx.image_caption or "").lower()
        words = re.findall(r"\b\w{3,}\b", text)
        stopwords = {"the", "and", "for", "with", "this", "that", "want", "need",
                     "make", "print", "find", "get", "can", "you", "как", "что",
                     "это", "мне", "для", "хочу", "нужен", "нужно", "можно"}
        keywords = [w for w in words if w not in stopwords][:5]

        # Determine intent_type from keywords
        intent_type = "search"  # default
        create_words = {"build", "create", "design", "make", "box", "plate", "bracket",
                        "создай", "построй", "сделай", "спроектируй"}
        research_words = {"how", "what", "compare", "research", "options", "alternatives",
                          "как", "что", "сравни", "исследуй", "варианты"}
        fabrication_words = {"print", "settings", "material", "infill", "fdm", "cnc",
                            "печать", "настройки", "материал", "заполнение"}
        modify_words = {"modify", "change", "add", "remove", "bigger", "smaller", "fillet",
                       "измени", "добавь", "убери", "больше", "меньше"}
        dimension_pattern = re.compile(r"\d+\s*[xх×]\s*\d+|\d+\s*mm|\d+\s*мм|\d+\s*inch")

        if ctx.has_image():
            intent_type = "photo_to_3d"
        elif dimension_pattern.search(text):
            intent_type = "create"  # explicit dimensions → definitely create
        elif any(w in words for w in fabrication_words):
            intent_type = "fabrication"
        elif any(w in words for w in research_words) and not any(w in words for w in create_words):
            intent_type = "research"
        elif any(w in words for w in modify_words):
            intent_type = "modify"
        elif any(w in words for w in create_words):
            intent_type = "create"

        # Extract dimensions if present
        dimensions = {"width": None, "height": None, "depth": None, "diameter": None, "units": "mm"}
        dim_match = re.search(r"(\d+(?:\.\d+)?)\s*[xх×]\s*(\d+(?:\.\d+)?)\s*(?:[xх×]\s*(\d+(?:\.\d+)?))?", text)
        if dim_match:
            dimensions["width"] = float(dim_match.group(1))
            dimensions["height"] = float(dim_match.group(2))
            if dim_match.group(3):
                dimensions["depth"] = float(dim_match.group(3))

        return {
            "intent_type": intent_type,
            "object_name": " ".join(keywords[:2]) if keywords else "3d model",
            "object_type": "other",
            "keywords": keywords,
            "dimensions": dimensions,
            "mounting": None,
            "size_hint": None,
            "material_hint": None,
            "confidence": 0.3,
            "_source": "algorithm",
        }
