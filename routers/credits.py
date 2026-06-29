"""Credits endpoints — quote (estimate) and commit (atomic consume).

Flow:
  UI → POST /api/credits/quote { action, params }
      → { action, cost, breakdown, balance_after }   ← UI shows confirmation
  UI confirms → backend action runs (mesh_analyze / cad_execute / ...)
  On success → POST /api/credits/commit { action, params, idempotency_key }
      → { remaining, charged }                       ← atomic via Supabase RPC

If a request is replayed with the same idempotency_key, the second call
returns the original charge without double-billing.

Insufficient credits → 402 Payment Required.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from orynd_core.auth import UserContext, current_user
from orynd_core.auth import supabase_client as sb
from orynd_core.auth import users as users_repo
from orynd_core.services.credits import (
    PriceQuote,
    PricingError,
    list_actions,
    quote_action,
)

log = logging.getLogger("orynd.credits")

router = APIRouter(prefix="/api/credits", tags=["credits"])


# ---------------------------------------------------------------------------
# /api/credits/actions — public(ish) pricing table for UI
# ---------------------------------------------------------------------------
@router.get("/actions")
async def actions(_user: UserContext = Depends(current_user)) -> dict[str, Any]:
    return {
        "actions": [
            {
                "action": a.action,
                "base": a.base,
                "unit_label": a.unit_label,
                "per_unit": a.per_unit,
                "description": a.description,
            }
            for a in list_actions()
        ]
    }


# ---------------------------------------------------------------------------
# /api/credits/quote — non-charging estimate
# ---------------------------------------------------------------------------
class QuoteRequest(BaseModel):
    action: str = Field(..., description="Action identifier, e.g. 'mesh_analyze'")
    params: dict[str, Any] = Field(default_factory=dict)


class QuoteResponse(BaseModel):
    action: str
    cost: int
    breakdown: dict[str, Any]
    balance: int
    balance_after: int
    sufficient: bool


@router.post("/quote", response_model=QuoteResponse)
async def quote(
    body: QuoteRequest,
    user: UserContext = Depends(current_user),
) -> QuoteResponse:
    try:
        q: PriceQuote = quote_action(body.action, body.params)
    except PricingError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e

    row = await users_repo.get_user(user.id)
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not synced")
    balance = int(row.get("credits", 0))
    balance_after = balance - q.cost

    return QuoteResponse(
        action=q.action,
        cost=q.cost,
        breakdown=q.breakdown,
        balance=balance,
        balance_after=balance_after,
        sufficient=balance_after >= 0,
    )


# ---------------------------------------------------------------------------
# /api/credits/commit — atomic consume via Supabase RPC
# ---------------------------------------------------------------------------
class CommitRequest(BaseModel):
    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Client-generated unique key for this action attempt.",
    )


class CommitResponse(BaseModel):
    charged: int
    remaining: int
    idempotent_replay: bool


@router.post("/commit", response_model=CommitResponse)
async def commit(
    body: CommitRequest,
    user: UserContext = Depends(current_user),
) -> CommitResponse:
    # 1) Recompute cost server-side from action+params (DO NOT trust client cost).
    try:
        q: PriceQuote = quote_action(body.action, body.params)
    except PricingError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e

    # 2) Idempotency check — has this key already produced a charge?
    existing = await sb.select_one(
        "usage_logs",
        filters={
            "user_id": user.id,
            "idempotency_key": body.idempotency_key,
        },
    )
    if existing:
        row = await users_repo.get_user(user.id)
        remaining = int(row.get("credits", 0)) if row else 0
        return CommitResponse(
            charged=int(existing.get("credits_consumed", q.cost)),
            remaining=remaining,
            idempotent_replay=True,
        )

    # 3) Atomic consume via Postgres function.
    try:
        new_balance = await users_repo.consume_credits(
            user_id=user.id,
            amount=q.cost,
            action=body.action,
            metadata={
                "params": body.params,
                "breakdown": q.breakdown,
                "idempotency_key": body.idempotency_key,
            },
        )
    except sb.SupabaseError as e:
        msg = str(e).lower()
        if "insufficient_credits" in msg:
            raise HTTPException(
                status.HTTP_402_PAYMENT_REQUIRED,
                "insufficient_credits",
            ) from e
        log.error("consume_credits failed user=%s action=%s: %s", user.id, body.action, e)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "credit ledger error") from e

    # 4) Backfill idempotency_key into the just-written usage_log row.
    # consume_credits inserts the row inside the same transaction without the key
    # (the RPC signature is fixed); we tag it after the fact by latest match.
    # This is a best-effort tag — if it fails, the consume already happened.
    try:
        latest = await sb.select_one(
            "usage_logs",
            filters={
                "user_id": user.id,
                "action": body.action,
            },
        )
        if latest and latest.get("idempotency_key") is None:
            await sb.update(
                "usage_logs",
                filters={"id": latest["id"]},
                values={"idempotency_key": body.idempotency_key},
            )
    except Exception as e:  # noqa: BLE001 — non-fatal
        log.warning("could not tag idempotency_key on usage_log: %s", e)

    return CommitResponse(
        charged=q.cost,
        remaining=int(new_balance),
        idempotent_replay=False,
    )
