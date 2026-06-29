"""CryptoBot Crypto Pay API client.

Docs: https://help.crypt.bot/crypto-pay-api
Token env: CRYPTOBOT_TOKEN (format: {app_id}:{secret})
"""
from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Any

import httpx


class CryptoBotError(Exception):
    pass


@dataclass(frozen=True)
class Invoice:
    invoice_id: str
    pay_url: str
    order_id: str
    amount: str
    asset: str
    status: str


def _token() -> str:
    t = os.getenv("CRYPTOBOT_TOKEN")
    if not t:
        raise CryptoBotError("CRYPTOBOT_TOKEN not configured")
    return t


def _base_url() -> str:
    return os.getenv("CRYPTOBOT_API_URL", "https://pay.crypt.bot/api").rstrip("/")


class CryptoBotClient:
    def __init__(self, *, timeout: float = 15.0) -> None:
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Crypto-Pay-API-Token": _token(),
            "Content-Type": "application/json",
        }

    async def create_invoice(
        self,
        *,
        amount: float,
        asset: str = "USDT",
        order_id: str,
        description: str = "",
        paid_btn_url: str | None = None,
        expires_in: int = 3600,
    ) -> Invoice:
        body: dict[str, Any] = {
            "asset": asset,
            "amount": f"{amount:.2f}",
            "payload": order_id,
            "expires_in": expires_in,
        }
        if description:
            body["description"] = description
        if paid_btn_url:
            body["paid_btn_name"] = "viewItem"
            body["paid_btn_url"] = paid_btn_url

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(
                f"{_base_url()}/createInvoice",
                json=body,
                headers=self._headers(),
            )
        if r.status_code >= 400:
            raise CryptoBotError(f"createInvoice failed: {r.status_code} {r.text}")
        data = r.json()
        if not data.get("ok"):
            raise CryptoBotError(f"createInvoice error: {data}")
        result = data["result"]
        return Invoice(
            invoice_id=str(result["invoice_id"]),
            pay_url=result["pay_url"],
            order_id=result.get("payload", order_id),
            amount=result["amount"],
            asset=result["asset"],
            status=result["status"],
        )

    async def get_invoice(self, invoice_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.get(
                f"{_base_url()}/getInvoices",
                params={"invoice_ids": invoice_id},
                headers=self._headers(),
            )
        if r.status_code >= 400:
            raise CryptoBotError(f"getInvoices failed: {r.status_code} {r.text}")
        data = r.json()
        items = data.get("result", {}).get("items", [])
        return items[0] if items else {}


def verify_webhook_signature(body: bytes, header_sig: str) -> bool:
    """Verify CryptoBot webhook: HMAC-SHA256(SHA256(token), body)."""
    secret = hashlib.sha256(_token().encode()).digest()
    expected = hmac.new(secret, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header_sig.lower())
