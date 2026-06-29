"""Sequence pattern miner — finds repeated n-grams of action types.

Approach: sliding window over ordered signals, count tuples of length
``2..max_len``, return tuples whose count ≥ ``min_support``. Lightweight
enough to run on-device. Phase 1 swaps in PrefixSpan when real data lands.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from orynd_core.services.movement.signals import MovementSignal


@dataclass
class MovementPattern:
    actions: tuple[str, ...]
    support: int


class MovementPatternMiner:
    def __init__(self, *, min_support: int = 3, max_len: int = 5) -> None:
        self.min_support = min_support
        self.max_len = max_len

    def mine(self, signals: Iterable[MovementSignal]) -> list[MovementPattern]:
        actions = [s.action_type for s in signals]
        if len(actions) < 2:
            return []

        counts: Counter[tuple[str, ...]] = Counter()
        max_len = min(self.max_len, len(actions))
        for length in range(2, max_len + 1):
            for i in range(len(actions) - length + 1):
                counts[tuple(actions[i : i + length])] += 1

        return [
            MovementPattern(actions=key, support=cnt)
            for key, cnt in counts.most_common()
            if cnt >= self.min_support
        ]


__all__ = ["MovementPattern", "MovementPatternMiner"]
