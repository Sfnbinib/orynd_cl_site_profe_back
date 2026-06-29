"""Natural-language → CoreOps parser.

Phase 0 — deterministic keyword + numeric pattern extractor. Covers a
useful baseline of "create cube 20mm", "make cylinder r=5 h=20", "extrude
by 10", "rotate 45 deg around z", "drill hole 5mm". No LLM required.

Phase 1 (planned) — fall back to Claude through the model_router when this
parser cannot find a match. That makes the skill **manual + hint + agent**
in one path: simple inputs → free + instant; complex inputs → LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MacroParseResult:
    operations: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "parser"  # "parser" | "llm" later
    fallback_used: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operations": list(self.operations),
            "summary": _summarise(self.operations),
            "confidence": self.confidence,
            "source": self.source,
            "fallback_used": self.fallback_used,
            "notes": list(self.notes),
            "total": len(self.operations),
        }


def _summarise(ops: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for op in ops:
        t = str(op.get("type", "unknown"))
        out[t] = out.get(t, 0) + 1
    return out


# ---- Pattern helpers --------------------------------------------------------

_NUM = r"(\d+(?:\.\d+)?)"
_MM = r"(?:mm|\s*мм|millimeter[s]?)?"
_DEG = r"(?:deg|degrees?|\s*град|°)?"


def _f(match: re.Match, group: int) -> float:
    return float(match.group(group))


def _try_cube(text: str) -> Optional[dict]:
    # "20mm cube", "create cube 20", "куб 30 мм"
    patterns = [
        rf"\b(?:cube|куб|box|кубик)\s+{_NUM}\s*{_MM}",
        rf"{_NUM}\s*{_MM}\s+(?:cube|куб|box|кубик)",
        rf"(?:create|make|build|сделай|создай)\s+(?:a\s+)?(?:cube|box|куб)\s+(?:of\s+)?{_NUM}\s*{_MM}",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            size = _f(m, 1)
            return {
                "type": "box",
                "parameters": {"sx": size, "sy": size, "sz": size},
                "source": "macro:cube",
            }
    return None


def _try_box_xyz(text: str) -> Optional[dict]:
    # "box 10x20x30", "10 x 20 x 30 mm box"
    m = re.search(
        rf"\b(?:box|параллелепипед)?\s*{_NUM}\s*[x×]\s*{_NUM}\s*[x×]\s*{_NUM}\s*{_MM}",
        text,
        re.IGNORECASE,
    )
    if m:
        return {
            "type": "box",
            "parameters": {"sx": _f(m, 1), "sy": _f(m, 2), "sz": _f(m, 3)},
            "source": "macro:box_xyz",
        }
    return None


def _try_cylinder(text: str) -> Optional[dict]:
    # "cylinder r=5 h=20", "цилиндр радиус 5 высота 20"
    m = re.search(
        rf"(?:cylinder|cyl|цилиндр)\b[^0-9]*?(?:r(?:adius)?[=\s]+|радиус[=\s]+)?{_NUM}[^0-9]*?(?:h(?:eight)?[=\s]+|высот[аы]?[=\s]+)?{_NUM}",
        text,
        re.IGNORECASE,
    )
    if m:
        return {
            "type": "cylinder",
            "parameters": {"radius": _f(m, 1), "height": _f(m, 2)},
            "source": "macro:cylinder",
        }
    return None


def _try_sphere(text: str) -> Optional[dict]:
    m = re.search(
        rf"(?:sphere|шар|sphere)\s+(?:r(?:adius)?[=\s]+|радиус[=\s]+)?{_NUM}",
        text,
        re.IGNORECASE,
    )
    if m:
        return {
            "type": "sphere",
            "parameters": {"radius": _f(m, 1)},
            "source": "macro:sphere",
        }
    return None


def _try_extrude(text: str) -> Optional[dict]:
    # "extrude by 10", "выдавить на 15мм"
    m = re.search(
        rf"(?:extrude|выдавить|выдави)\s+(?:by\s+|на\s+)?{_NUM}\s*{_MM}",
        text,
        re.IGNORECASE,
    )
    if m:
        return {
            "type": "extrude",
            "parameters": {"distance": _f(m, 1)},
            "source": "macro:extrude",
        }
    return None


def _try_rotate(text: str) -> Optional[dict]:
    # "rotate 45 deg around z", "поверни на 90 градусов"
    m = re.search(
        rf"(?:rotate|поверни|вращ\w+)\s+(?:by\s+|на\s+)?{_NUM}\s*{_DEG}(?:\s+around\s+([xyz])|\s+по\s+оси\s+([xyz]))?",
        text,
        re.IGNORECASE,
    )
    if m:
        axis = (m.group(2) or m.group(3) or "z").lower()
        return {
            "type": "rotate",
            "parameters": {"angle_deg": _f(m, 1), "axis": axis},
            "source": "macro:rotate",
        }
    return None


def _try_translate(text: str) -> Optional[dict]:
    # "move by 10,20,30", "translate x=5 y=0 z=10"
    m = re.search(
        rf"(?:move|translate|сдвинь|переместить)\s+(?:by\s+)?{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}",
        text,
        re.IGNORECASE,
    )
    if m:
        return {
            "type": "translate",
            "parameters": {"dx": _f(m, 1), "dy": _f(m, 2), "dz": _f(m, 3)},
            "source": "macro:translate",
        }
    return None


def _try_drill(text: str) -> Optional[dict]:
    # "drill hole 5mm", "просверли отверстие 8 мм"
    m = re.search(
        rf"(?:drill|просверли|hole|отверст\w+)\s+(?:hole\s+)?(?:r(?:adius)?[=\s]+|radius\s+)?{_NUM}\s*{_MM}",
        text,
        re.IGNORECASE,
    )
    if m:
        return {
            "type": "drill_hole",
            "parameters": {"radius": _f(m, 1)},
            "source": "macro:drill",
        }
    return None


_TRY_FUNCS = [
    _try_box_xyz,  # before _try_cube — more specific
    _try_cube,
    _try_cylinder,
    _try_sphere,
    _try_extrude,
    _try_rotate,
    _try_translate,
    _try_drill,
]


# ---- Public entry --------------------------------------------------------


def parse_text_to_coreops(
    text: str,
    *,
    use_llm_fallback: bool = False,
) -> MacroParseResult:
    """Parse user text into a CoreOps document.

    Args:
        text: arbitrary command. May contain multiple ops separated by ``;``,
              ``,`` or ``and``/``и`` keywords.
        use_llm_fallback: if True AND nothing matched, try Anthropic via
            model_router. Off by default (no API cost in demo).
    """
    text = (text or "").strip()
    if not text:
        return MacroParseResult(notes=["empty input"])

    # Split on common multi-op separators (";" / ", then" / " and " / " и ")
    chunks = re.split(r"\s*(?:;|,\s+(?:then|then\s+)?|\s+and\s+|\s+и\s+)\s*", text)
    operations: list[dict] = []
    notes: list[str] = []
    matched_count = 0

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        matched = False
        for fn in _TRY_FUNCS:
            op = fn(chunk)
            if op:
                operations.append(op)
                matched_count += 1
                matched = True
                break
        if not matched:
            notes.append(f"unrecognised: {chunk!r}")

    confidence = matched_count / max(len(chunks), 1)

    fallback_used = False
    if not operations and use_llm_fallback:
        # Phase 1 hook: route to Claude/Ollama via model_router.
        # Stub for now — returns confidence 0 + note.
        notes.append("LLM fallback requested but not yet wired (Phase 1)")
        fallback_used = True

    return MacroParseResult(
        operations=operations,
        confidence=confidence,
        source="parser",
        fallback_used=fallback_used,
        notes=notes,
    )


__all__ = ["MacroParseResult", "parse_text_to_coreops"]
