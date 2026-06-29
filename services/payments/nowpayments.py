"""NOWPayments REST API client.

Docs: https://documenter.getpostman.com/view/7907941/2s93JusNJt

We only use what we need: create invoice, fetch payment status. Webhook (IPN)
handling lives in the billing router; signature verification in `signature.py`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx


class NOWPaymentsError(Exception):
    pass


PaymentStatus = Literal[
    "waiting",
    "confirming",
    "confirmed",
    "sending",
    "partially_paid",
    "finished",
    "failed",
    "refunded",
    "expired",
]


@dataclass(frozen=True)
class Invoice:
    invoice_id: str
    invoice_url: str
    order_id: str
    price_amount: float
    price_currency: str


def _api_key() -> str:
    k = os.getenv("NOWPAYMENTS_API_KEY")
    if not k:
        raise NOWPaymentsError("NOWPAYMENTS_API_KEY not configured")
    return k


def _base_url() -> str:
    return os.getenv("NOWPAYMENTS_BASE_URL", "https://api.nowpayments.io/v1").rstrip("/")


class NOWPaymentsClient:
    def __init__(self, *, timeout: float = 15.0) -> None:
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": _api_key(), "Content-Type": "application/json"}

    async def create_invoice(
        self,
        *,
        price_amount: float,
        price_currency: str,
        order_id: str,
        order_description: str,
        ipn_callback_url: str,
        success_url: str | None = None,
        cancel_url: str | None = None,
        pay_currency: str | None = None,
    ) -> Invoice:
        """Create a hosted invoice. User completes payment on NOWPayments-hosted page.

        `pay_currency` (optional) forces a single accepted currency (e.g. 'usdttrc20').
        """
        body: dict[str, Any] = {
            "price_amount": price_amount,
            "price_currency": price_currency,
            "order_id": order_id,
            "order_description": order_description,
            "ipn_callback_url": ipn_callback_url,
        }
        if success_url:
            body["success_url"] = success_url
        if cancel_url:
            body["cancel_url"] = cancel_url
        if pay_currency:
            body["pay_currency"] = pay_currency

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(f"{_base_url()}/invoice", json=body, headers=self._headers())
        if r.status_code >= 400:
            raise NOWPaymentsError(f"create_invoice failed: {r.status_code} {r.text}")
        data = r.json()
        return Invoice(
            invoice_id=str(data["id"]),
            invoice_url=data["invoice_url"],
            order_id=data["order_id"],
            price_amount=float(data["price_amount"]),
            price_currency=data["price_currency"],
        )

    async def get_payment_status(self, payment_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.get(
                f"{_base_url()}/payment/{payment_id}",
                headers=self._headers(),
            )
        if r.status_code >= 400:
            raise NOWPaymentsError(f"get_payment_status failed: {r.status_code} {r.text}")
        return r.json()

    async def get_available_currencies(self) -> list[str]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.get(f"{_base_url()}/currencies", headers=self._headers())
        if r.status_code >= 400:
            raise NOWPaymentsError(f"get_currencies failed: {r.status_code} {r.text}")
        return r.json().get("currencies", [])
