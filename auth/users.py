"""Business logic for web user billing.

Plan + credits live in `public.subscriptions` (keyed by user_id, FK to
auth.users). Email/name in `public.profiles`. The legacy `public.users` table
belongs to the Telegram bot and is NOT used here.
"""
from __future__ import annotations

import os
from typing import Any

from . import supabase_client as sb


def _free_credits() -> int:
    try:
        return int(os.getenv("PLAN_FREE_CREDITS", "100"))
    except ValueError:
        return 100


async def get_subscription(user_id: str) -> dict[str, Any] | None:
    return await sb.select_one("subscriptions", filters={"user_id": user_id})


async def ensure_subscription(user_id: str, email: str | None = None) -> dict[str, Any]:
    """Return the user's subscription row, creating a free one on first visit."""
    existing = await get_subscription(user_id)
    if existing:
        return existing

    row = await sb.insert(
        "subscriptions",
        {
            "user_id": user_id,
            "plan": "free",
            "status": "active",
            "credits_balance": _free_credits(),
        },
    )

    # Best-effort: keep a profile row with the email (non-fatal if it fails).
    if email:
        try:
            await sb.upsert(
                "profiles",
                {"user_id": user_id, "email": email},
                on_conflict="user_id",
            )
        except Exception:
            pass

    return row


async def add_credits(user_id: str, amount: int) -> dict[str, Any] | None:
    if amount <= 0:
        raise ValueError("amount must be positive")
    sub = await get_subscription(user_id)
    if not sub:
        return None
    new_balance = int(sub.get("credits_balance", 0) or 0) + amount
    granted = int(sub.get("credits_granted_total", 0) or 0) + amount
    return await sb.update(
        "subscriptions",
        filters={"user_id": user_id},
        values={"credits_balance": new_balance, "credits_granted_total": granted},
    )


async def set_plan(user_id: str, plan: str) -> dict[str, Any] | None:
    if plan not in ("free", "pro", "team"):
        raise ValueError(f"invalid plan: {plan}")
    return await sb.update(
        "subscriptions",
        filters={"user_id": user_id},
        values={"plan": plan},
    )
