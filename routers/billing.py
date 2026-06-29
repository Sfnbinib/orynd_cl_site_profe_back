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
    NOWPaymentsClient,
    NOWPaymentsError,
    verify_ipn_signature,
)

log = logging.getLogger("orynd.billing")

router = APIRouter(prefix="/api/billing", tags=["billing"])

Plan = Literal["pro", "team"]

# Statuses that grant credits. Other statuses (waiting, failed, refunded, etc.)
# leave the payment record but do not credit the user.
FINISHED_STATUSES = {"finished", "confirmed"}


def _plan_config(plan: Plan) -> tuple[float, int]:
    """Return (price_usd, credits) for the requested plan."""
    if plan == "pro":
        return (
            float(os.getenv("PLAN_PRO_PRICE_USD", "29")),
            int(os.getenv("PLAN_PRO_CREDITS", "1500")),
        )
    if plan == "team":
        return (
            float(os.getenv("PLAN_TEAM_PRICE_USD", "99")),
            int(os.getenv("PLAN_TEAM_CREDITS", "6000")),
        )
    raise HTTPException(status.HTTP_400_BAD_REQUEST, f"unknown plan: {plan}")


def _ipn_callback_url() -> str:
    base = os.getenv("PUBLIC_API_URL", "http://localhost:8000").rstrip("/")
    return f"{base}/api/billing/webhook"


# ---------------------------------------------------------------------------
# /api/billing/me
# ---------------------------------------------------------------------------
class MeBillingResponse(BaseModel):
    plan: str
    credits: int
    pending_payment: dict[str, Any] | None = None


@router.get("/me", response_model=MeBillingResponse)
async def me(user: UserContext = Depends(current_user)) -> MeBillingResponse:
    row = await users_repo.get_user(user.id)
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not synced")
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
    pay_currency: str | None = Field(
        default=None,
        description="Optional preferred crypto, e.g. 'usdttrc20', 'btc', 'eth'. Omit to let user pick.",
    )


class CheckoutResponse(BaseModel):
    invoice_url: str
    invoice_id: str
    order_id: str
    amount_usd: float


@router.post("/checkout", response_model=CheckoutResponse)
async def checkout(
    body: CheckoutRequest,
    user: UserContext = Depends(current_user),
) -> CheckoutResponse:
    price_usd, _credits = _plan_config(body.plan)
    order_id = f"{user.id}:{body.plan}:{int(time.time())}:{uuid.uuid4().hex[:8]}"

    client = NOWPaymentsClient()
    try:
        invoice = await client.create_invoice(
            price_amount=price_usd,
            price_currency="usd",
            order_id=order_id,
            order_description=f"ORYND {body.plan.title()} plan",
            ipn_callback_url=_ipn_callback_url(),
            success_url=os.getenv("BILLING_SUCCESS_URL"),
            cancel_url=os.getenv("BILLING_CANCEL_URL"),
            pay_currency=body.pay_currency,
        )
    except NOWPaymentsError as e:
        log.error("create_invoice failed for user=%s plan=%s: %s", user.id, body.plan, e)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "payment provider error") from e

    # Pre-record a pending payment row so the UI can show "waiting".
    # The IPN webhook will move it to 'confirmed' (or 'failed') and credit the user.
    await sb.insert(
        "payments",
        {
            "user_id": user.id,
            "amount": price_usd,
            "currency": "USD",
            "status": "pending",
            "provider": "nowpayments",
            "provider_payment_id": f"invoice:{invoice.invoice_id}",
            "metadata": {
                "plan": body.plan,
                "order_id": order_id,
                "pay_currency": body.pay_currency,
            },
        },
    )

    return CheckoutResponse(
        invoice_url=invoice.invoice_url,
        invoice_id=invoice.invoice_id,
        order_id=order_id,
        amount_usd=price_usd,
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

    secret = os.getenv("NOWPAYMENTS_IPN_SECRET")
    if not secret:
        log.error("NOWPAYMENTS_IPN_SECRET not set — refusing webhook")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "ipn not configured")

    sig = request.headers.get("x-nowpayments-sig", "")
    if not verify_ipn_signature(payload=payload, header_sig=sig, secret=secret):
        log.warning("IPN signature mismatch from %s; body=%s", request.client.host if request.client else "?", raw[:200])
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad signature")

    payment_id = str(payload.get("payment_id") or "")
    payment_status = str(payload.get("payment_status") or "")
    order_id = str(payload.get("order_id") or "")
    pay_currency = str(payload.get("pay_currency") or "")
    pay_address = str(payload.get("pay_address") or "")
    actually_paid = payload.get("actually_paid")

    if not payment_id or not order_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "missing payment_id/order_id")

    user_id, plan = _parse_order_id(order_id)
    if not user_id or not plan:
        log.error("cannot parse order_id=%s", order_id)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "bad order_id")

    # Idempotency: if we already recorded this payment_id as confirmed, no-op.
    existing = await sb.select_one(
        "payments",
        filters={"provider": "nowpayments", "provider_payment_id": payment_id},
    )

    is_finished = payment_status in FINISHED_STATUSES

    if existing:
        if existing.get("status") == "confirmed" and is_finished:
            return {"status": "ok", "note": "already processed"}
        # Update status / metadata in place.
        await sb.update(
            "payments",
            filters={"id": existing["id"]},
            values={
                "status": "confirmed" if is_finished else payment_status,
                "crypto_address": pay_address or existing.get("crypto_address"),
                "currency": pay_currency.upper() or existing.get("currency"),
                "metadata": {**(existing.get("metadata") or {}), "ipn": payload},
                **({"confirmed_at": "now()"} if is_finished else {}),
            },
        )
    else:
        await sb.insert(
            "payments",
            {
                "user_id": user_id,
                "amount": float(actually_paid or payload.get("pay_amount") or 0),
                "currency": pay_currency.upper() or "USD",
                "crypto_address": pay_address or None,
                "status": "confirmed" if is_finished else payment_status,
                "provider": "nowpayments",
                "provider_payment_id": payment_id,
                "metadata": {"order_id": order_id, "plan": plan, "ipn": payload},
            },
        )

    if is_finished:
        _price, credits = _plan_config(plan)  # type: ignore[arg-type]
        await users_repo.add_credits(user_id, credits)
        await users_repo.set_plan(user_id, plan)
        log.info("credited user=%s plan=%s credits=%s payment=%s", user_id, plan, credits, payment_id)

    return {"status": "ok"}
