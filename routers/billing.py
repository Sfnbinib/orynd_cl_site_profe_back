"""Billing endpoints — checkout, webhook (IPN), account status."""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from orynd_core.auth import UserContext, current_user
from orynd_core.auth import supabase_client as sb
from orynd_core.auth import users as users_repo
from orynd_core.services.payments import (
    CryptoBotClient,
    CryptoBotError,
    verify_webhook_signature,
)

log = logging.getLogger("orynd.billing")

router = APIRouter(prefix="/api/billing", tags=["billing"])

Plan = Literal["pro", "team"]

# Statuses that grant credits. Other statuses (waiting, failed, refunded, etc.)
# leave the payment record but do not credit the user.
FINISHED_STATUSES = {"paid"}


def _plan_config(plan: Plan) -> tuple[float, int]:
    """Return (price_usd, credits) for the requested plan."""
    if plan == "pro":
        return (
            float(os.getenv("PLAN_PRO_PRICE_USD", "20")),
            int(os.getenv("PLAN_PRO_CREDITS", "1500")),
        )
    if plan == "team":
        return (
            float(os.getenv("PLAN_TEAM_PRICE_USD", "99")),
            int(os.getenv("PLAN_TEAM_CREDITS", "6000")),
        )
    raise HTTPException(status.HTTP_400_BAD_REQUEST, f"unknown plan: {plan}")


# ---------------------------------------------------------------------------
# /api/billing/me
# ---------------------------------------------------------------------------
class MeBillingResponse(BaseModel):
    plan: str
    credits: int
    pending_payment: dict[str, Any] | None = None


@router.get("/me", response_model=MeBillingResponse)
async def me(user: UserContext = Depends(current_user)) -> MeBillingResponse:
    # Auto-provision a free-tier row on first authenticated visit (users who
    # signed up via Supabase Auth directly never hit the backend signup path).
    row = await users_repo.ensure_user(user_id=user.id, email=user.email or "")
    # Latest pending payment, if any (lets UI resume "waiting for confirmation").
    pending = await sb.select_one(
        "payments",
        filters={"user_id": user.id, "status": "pending"},
    )
    return MeBillingResponse(
        plan=row.get("plan", "free"),
        credits=int(row.get("credits", 0)),
        pending_payment=pending,
    )


# ---------------------------------------------------------------------------
# /api/billing/checkout
# ---------------------------------------------------------------------------
class CheckoutRequest(BaseModel):
    plan: Plan
    asset: str = Field(default="USDT", description="Crypto asset: USDT, BTC, ETH, TON")


class CheckoutResponse(BaseModel):
    pay_url: str
    invoice_id: str
    order_id: str
    amount: float
    asset: str


@router.post("/checkout", response_model=CheckoutResponse)
async def checkout(
    body: CheckoutRequest,
    user: UserContext = Depends(current_user),
) -> CheckoutResponse:
    price_usd, _credits = _plan_config(body.plan)
    order_id = f"{user.id}:{body.plan}:{int(time.time())}:{uuid.uuid4().hex[:8]}"

    success_url = os.getenv("BILLING_SUCCESS_URL", os.getenv("FRONTEND_URL", "") + "/account.html")

    client = CryptoBotClient()
    try:
        invoice = await client.create_invoice(
            amount=price_usd,
            asset=body.asset.upper(),
            order_id=order_id,
            description=f"ORYND {body.plan.title()} plan",
            paid_btn_url=success_url,
        )
    except CryptoBotError as e:
        log.error("create_invoice failed for user=%s plan=%s: %s", user.id, body.plan, e)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "payment provider error") from e

    await sb.insert(
        "payments",
        {
            "user_id": user.id,
            "amount": price_usd,
            "currency": body.asset.upper(),
            "status": "pending",
            "provider": "cryptobot",
            "provider_payment_id": invoice.invoice_id,
            "metadata": {"plan": body.plan, "order_id": order_id},
        },
    )

    return CheckoutResponse(
        pay_url=invoice.pay_url,
        invoice_id=invoice.invoice_id,
        order_id=order_id,
        amount=price_usd,
        asset=body.asset.upper(),
    )


# ---------------------------------------------------------------------------
# /api/billing/webhook (IPN)
# ---------------------------------------------------------------------------
def _parse_order_id(order_id: str) -> tuple[str | None, str | None]:
    """order_id format: '{user_uuid}:{plan}:{ts}:{nonce}'."""
    parts = order_id.split(":", 3)
    if len(parts) < 2:
        return None, None
    user_id, plan = parts[0], parts[1]
    if plan not in ("pro", "team"):
        return user_id, None
    return user_id, plan


@router.post("/webhook")
async def webhook(request: Request) -> dict[str, str]:
    raw = await request.body()
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid json") from e

    sig = request.headers.get("crypto-pay-api-signature", "")
    if not sig or not verify_webhook_signature(raw, sig):
        log.warning("CryptoBot webhook signature mismatch from %s", request.client.host if request.client else "?")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad signature")

    # CryptoBot payload: {"update_id": ..., "update_type": "invoice_paid", "payload": {...}}
    if payload.get("update_type") != "invoice_paid":
        return {"status": "ok", "note": "ignored"}

    invoice_data = payload.get("payload", {})
    invoice_id = str(invoice_data.get("invoice_id") or "")
    order_id = str(invoice_data.get("payload") or "")  # our order_id stored in payload field
    asset = str(invoice_data.get("asset") or "")
    amount = invoice_data.get("amount", "0")

    if not invoice_id or not order_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "missing invoice_id/order_id")

    user_id, plan = _parse_order_id(order_id)
    if not user_id or not plan:
        log.error("cannot parse order_id=%s", order_id)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "bad order_id")

    existing = await sb.select_one(
        "payments",
        filters={"provider": "cryptobot", "provider_payment_id": invoice_id},
    )

    if existing:
        if existing.get("status") == "confirmed":
            return {"status": "ok", "note": "already processed"}
        await sb.update(
            "payments",
            filters={"id": existing["id"]},
            values={
                "status": "confirmed",
                "currency": asset,
                "metadata": {**(existing.get("metadata") or {}), "webhook": payload},
            },
        )
    else:
        await sb.insert(
            "payments",
            {
                "user_id": user_id,
                "amount": float(amount),
                "currency": asset,
                "status": "confirmed",
                "provider": "cryptobot",
                "provider_payment_id": invoice_id,
                "metadata": {"order_id": order_id, "plan": plan, "webhook": payload},
            },
        )

    _price, credits = _plan_config(plan)  # type: ignore[arg-type]
    await users_repo.add_credits(user_id, credits)
    await users_repo.set_plan(user_id, plan)
    log.info("credited user=%s plan=%s credits=%s invoice=%s", user_id, plan, credits, invoice_id)

    return {"status": "ok"}
