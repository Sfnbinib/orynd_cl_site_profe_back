"""Learning Engine — theory-vs-practice comparison (blueprint 42/04).

Algorithm path (DSPy/embeddings optional, Phase 1 runs without them):

  compare(practice, theories) → ComparisonResult
    Axis 1 semantic   — token Jaccard between practice text and theory text
                        (embedding cosine when an embedder is wired later)
    Axis 2 structural — practice params vs theory recommended ranges
    Axis 3 outcome    — Jaccard between practice result and expected outcome

  aggregate = 0.4*semantic + 0.4*structural + 0.2*outcome   (founder defaults)

Suggestion generation is rate-limited to 3 per 10 minutes per session
(founder default, DECISIONS_LOG).
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_WEIGHTS = {"semantic": 0.4, "structural": 0.4, "outcome": 0.2}

SUGGESTION_RATE_LIMIT = 3
SUGGESTION_RATE_WINDOW_S = 600.0


@dataclass
class TheoryPattern:
    pattern_id: str
    action_type: str
    text: str
    recommended_params: dict[str, Any] = field(default_factory=dict)
    expected_outcome: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "action_type": self.action_type,
            "text": self.text,
            "recommended_params": dict(self.recommended_params),
            "expected_outcome": dict(self.expected_outcome),
        }


@dataclass
class ComparisonResult:
    match_score: float
    best_match: Optional[TheoryPattern]
    gap_areas: list[dict]
    confidence: float
    reasoning: str
    axis_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_score": round(self.match_score, 4),
            "best_match": self.best_match.to_dict() if self.best_match else None,
            "gap_areas": list(self.gap_areas),
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "axis_scores": {k: round(v, 4) for k, v in self.axis_scores.items()},
        }


# ── Axis 1: semantic ─────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-zа-яё0-9_]+", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 2}


def semantic_similarity(practice_text: str, theory_text: str) -> float:
    a, b = _tokens(practice_text), _tokens(theory_text)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Axis 2: structural (blueprint 42/04 verbatim semantics) ─────────────────


def structural_similarity(practice_params: dict, recommended: dict) -> float:
    matches = 0.0
    total = 0
    for key, expected in recommended.items():
        if key not in practice_params:
            continue
        value = practice_params[key]
        if not isinstance(value, (int, float)):
            continue
        total += 1
        if isinstance(expected, (list, tuple)) and len(expected) == 2:
            min_v, max_v = float(expected[0]), float(expected[1])
            if min_v <= value <= max_v:
                matches += 1
            else:
                gap = min(abs(value - min_v), abs(value - max_v))
                relative = gap / ((max_v - min_v) or 1)
                if relative < 0.5:
                    matches += 0.5
        else:
            if value == expected:
                matches += 1
    return matches / total if total > 0 else 0.0


# ── Axis 3: outcome ──────────────────────────────────────────────────────────


def outcome_similarity(practice_result: Optional[dict], expected: dict) -> float:
    if not expected:
        return 0.0
    practice_result = practice_result or {}
    a = {f"{k}={v}" for k, v in practice_result.items()}
    b = {f"{k}={v}" for k, v in expected.items()}
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Aggregation + compare ────────────────────────────────────────────────────


def aggregate(semantic: float, structural: float, outcome: float,
              weights: Optional[dict[str, float]] = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    return w["semantic"] * semantic + w["structural"] * structural + w["outcome"] * outcome


def _find_gaps(practice_params: dict, theory: TheoryPattern) -> list[dict]:
    gaps: list[dict] = []
    for key, expected in theory.recommended_params.items():
        if key not in practice_params:
            gaps.append({"area": key, "kind": "missing_param", "recommended": expected})
            continue
        value = practice_params[key]
        if isinstance(expected, (list, tuple)) and len(expected) == 2 and isinstance(value, (int, float)):
            if not (float(expected[0]) <= value <= float(expected[1])):
                gaps.append({
                    "area": key,
                    "kind": "out_of_range",
                    "value": value,
                    "recommended": list(expected),
                })
    return gaps


def compare(
    action_type: str,
    action_params: dict,
    practice_text: str,
    theories: list[TheoryPattern],
    result: Optional[dict] = None,
    weights: Optional[dict[str, float]] = None,
) -> ComparisonResult:
    """Score practice against every theory pattern, return the best match."""
    relevant = [t for t in theories if t.action_type in (action_type, "*")] or theories
    if not relevant:
        return ComparisonResult(0.0, None, [], 0.0, "no theory patterns available")

    best: tuple[float, TheoryPattern, dict[str, float]] | None = None
    for theory in relevant:
        sem = semantic_similarity(practice_text, theory.text)
        struct = structural_similarity(action_params, theory.recommended_params)
        out = outcome_similarity(result, theory.expected_outcome)
        score = aggregate(sem, struct, out, weights)
        axes = {"semantic": sem, "structural": struct, "outcome": out}
        if best is None or score > best[0]:
            best = (score, theory, axes)

    score, theory, axes = best
    gaps = _find_gaps(action_params, theory)
    # Confidence drops when scoring relied on a single axis
    populated = sum(1 for v in axes.values() if v > 0)
    confidence = min(1.0, 0.4 + 0.2 * populated)
    reasoning = (
        f"best theory '{theory.pattern_id}' for {action_type}: "
        f"semantic={axes['semantic']:.2f} structural={axes['structural']:.2f} "
        f"outcome={axes['outcome']:.2f} → {score:.2f}; {len(gaps)} gap(s)"
    )
    return ComparisonResult(score, theory, gaps, confidence, reasoning, axes)


# ── Suggestion generation (rate-limited) ─────────────────────────────────────

_suggestion_times: dict[str, list[float]] = {}
_recent_comparisons: list[dict] = []
_lock = threading.Lock()


def _rate_ok(session_id: str) -> bool:
    now = time.monotonic()
    with _lock:
        times = [t for t in _suggestion_times.get(session_id, []) if now - t < SUGGESTION_RATE_WINDOW_S]
        if len(times) >= SUGGESTION_RATE_LIMIT:
            _suggestion_times[session_id] = times
            return False
        times.append(now)
        _suggestion_times[session_id] = times
        return True


def generate_suggestions(comparison: ComparisonResult, session_id: str = "default") -> list[dict]:
    """Turn gaps into actionable hints. Empty when rate-limited or no gaps."""
    if not comparison.gap_areas:
        return []
    if not _rate_ok(session_id):
        return []
    suggestions = []
    for gap in comparison.gap_areas[:3]:
        if gap["kind"] == "out_of_range":
            tip = (
                f"Parameter '{gap['area']}' = {gap['value']} is outside the recommended "
                f"range {gap['recommended'][0]}–{gap['recommended'][1]}."
            )
        else:
            tip = f"Consider setting '{gap['area']}' (recommended: {gap['recommended']})."
        suggestions.append({
            "format": "side",
            "content": {"tip_title": f"Check {gap['area']}", "tip_body": tip},
            "match_score": round(comparison.match_score, 4),
            "gap": gap,
        })
    return suggestions


def record_comparison(entry: dict) -> None:
    with _lock:
        _recent_comparisons.append(entry)
        del _recent_comparisons[:-50]


def recent_comparisons(limit: int = 20) -> list[dict]:
    with _lock:
        return list(_recent_comparisons[-limit:])


# ── Event bus subscriber (action.recorded → compare → suggest) ───────────────

# workspace tool name → learning action_type
_TOOL_TO_ACTION_TYPE: dict[str, str] = {
    "search_models": "search",
    "build_3d_model": "cad_build",
    "deep_research": "research",
    "analyze_mesh": "mesh_analyze",
    "analyze_image": "vision",
    "get_fabrication": "fabricate",
    "select_model": "select",
}


async def on_action_recorded(entry: dict) -> None:
    """Called by event_bus when workspace emits 'action.recorded'.

    Runs theory-vs-practice comparison and publishes 'learning.hint' if
    the engine has useful suggestions. Errors are swallowed — learning
    must never break the main tool loop.
    """
    try:
        from orynd_core.services.learning.theory_seed import SEED_THEORIES
        from orynd_core.services.event_bus import bus

        tool = entry.get("tool", "")
        action_type = _TOOL_TO_ACTION_TYPE.get(tool, tool)
        params = dict(entry.get("params") or {})
        result = entry.get("result") if isinstance(entry.get("result"), dict) else None
        session_id = entry.get("session_id", "default")
        practice_text = f"{action_type} {params}"

        comparison = compare(
            action_type=action_type,
            action_params=params,
            practice_text=practice_text,
            theories=SEED_THEORIES,
            result=result,
        )
        suggestions = generate_suggestions(comparison, session_id=session_id)

        record_comparison({
            "action_type": action_type,
            "match_score": comparison.match_score,
            "gap_areas": comparison.gap_areas,
            "reasoning": comparison.reasoning,
            "session_id": session_id,
            "computed_at": time.time(),
        })

        if suggestions:
            await bus.publish("learning.hint", {
                "session_id": session_id,
                "tool": tool,
                "suggestions": suggestions,
                "match_score": comparison.match_score,
            })
    except Exception:
        pass  # never block the event loop
