"""Movement Engine — Phase 0 = signal collection only.

UI suggestions intentionally not wired (per 43_movement_engine.md phase
rollout). This module captures the data that Phase 1 will use for ghost
suggestions and Phase 2 for cross-user federated patterns.
"""

from orynd_core.services.movement.local_store import (
    MovementStore,
    get_movement_store,
    reset_movement_store,
)
from orynd_core.services.movement.pattern_miner import (
    MovementPattern,
    MovementPatternMiner,
)
from orynd_core.services.movement.predictor import (
    MovementPredictor,
    NextActionPrediction,
)
from orynd_core.services.movement.signals import MovementSignal

__all__ = [
    "MovementSignal",
    "MovementStore",
    "MovementPattern",
    "MovementPatternMiner",
    "MovementPredictor",
    "NextActionPrediction",
    "get_movement_store",
    "reset_movement_store",
]
