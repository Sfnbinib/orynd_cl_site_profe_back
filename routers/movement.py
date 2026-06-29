"""/movement/* — signal recording + pattern listing + next-action prediction.

Phase 0 contract: signals stored in-memory; prediction returned only when
a pattern with min_support=2 fires. UI consumes /movement/predict to render
ghost suggestions in Phase 1.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from orynd_core.services.movement import (
    MovementPatternMiner,
    MovementPredictor,
    MovementSignal,
    get_movement_store,
)

router = APIRouter(prefix="/movement", tags=["movement"])


@router.post("/signal", response_model=MovementSignal)
async def record_signal(signal: MovementSignal):
    return get_movement_store().record(signal)


@router.get("/session/{session_id}", response_model=list[MovementSignal])
async def list_session_signals(
    session_id: UUID, limit: int = Query(200, ge=1, le=2000)
):
    return get_movement_store().list_session(session_id, limit=limit)


@router.get("/user/{user_id}", response_model=list[MovementSignal])
async def list_user_signals(
    user_id: UUID, limit: int = Query(500, ge=1, le=5000)
):
    return get_movement_store().list_user(user_id, limit=limit)


@router.get("/patterns/{user_id}")
async def list_user_patterns(
    user_id: UUID,
    min_support: int = Query(3, ge=2),
    max_len: int = Query(5, ge=2, le=10),
) -> list[dict]:
    store = get_movement_store()
    miner = MovementPatternMiner(min_support=min_support, max_len=max_len)
    patterns = miner.mine(store.list_user(user_id))
    return [{"actions": list(p.actions), "support": p.support} for p in patterns]


@router.get("/predict/{session_id}")
async def predict_next(
    session_id: UUID,
    user_id: Optional[UUID] = None,
    tail: int = Query(3, ge=1, le=10),
) -> dict:
    store = get_movement_store()
    history = (
        store.list_user(user_id) if user_id is not None else store.list_session(session_id)
    )
    if not history:
        raise HTTPException(status_code=404, detail="no history for prediction")
    recent = store.list_session(session_id)[-tail:]
    predictor = MovementPredictor()
    prediction = predictor.predict_next(history, recent)
    if prediction is None:
        return {"prediction": None}
    return {
        "prediction": {
            "action_type": prediction.action_type,
            "confidence": prediction.confidence,
            "matched_prefix": list(prediction.matched_prefix),
            "derived_from_pattern": list(prediction.derived_from_pattern),
        }
    }
