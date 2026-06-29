"""Predict next action from observed signal tail.

For each pattern, try every prefix length ``i`` against the tail of the
recent action list. The longest matching prefix that has a next action
in the pattern wins. Confidence = support / total_pattern_support, capped
to ``[0, 1]``.

This is intentionally simple and fully deterministic — easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from orynd_core.services.movement.pattern_miner import (
    MovementPattern,
    MovementPatternMiner,
)
from orynd_core.services.movement.signals import MovementSignal


@dataclass
class NextActionPrediction:
    action_type: str
    confidence: float
    matched_prefix: tuple[str, ...]
    derived_from_pattern: tuple[str, ...]


class MovementPredictor:
    def __init__(self, *, miner: Optional[MovementPatternMiner] = None) -> None:
        self.miner = miner or MovementPatternMiner(min_support=2, max_len=5)

    def predict_next(
        self,
        history: Iterable[MovementSignal],
        recent: Iterable[MovementSignal],
    ) -> Optional[NextActionPrediction]:
        patterns = self.miner.mine(history)
        if not patterns:
            return None
        recent_actions = [s.action_type for s in recent]
        if not recent_actions:
            return None

        total_support = sum(p.support for p in patterns)
        best: Optional[tuple[float, NextActionPrediction]] = None
        for pattern in patterns:
            actions = pattern.actions
            for prefix_len in range(min(len(actions) - 1, len(recent_actions)), 0, -1):
                if tuple(recent_actions[-prefix_len:]) == actions[:prefix_len]:
                    next_action = actions[prefix_len]
                    confidence = pattern.support / total_support if total_support else 0
                    candidate = (
                        prefix_len + confidence,
                        NextActionPrediction(
                            action_type=next_action,
                            confidence=min(1.0, confidence),
                            matched_prefix=tuple(recent_actions[-prefix_len:]),
                            derived_from_pattern=actions,
                        ),
                    )
                    if best is None or candidate[0] > best[0]:
                        best = candidate
                    break

        return best[1] if best else None


__all__ = ["MovementPredictor", "NextActionPrediction"]
