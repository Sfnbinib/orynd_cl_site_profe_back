"""Business logic for the `users` table — ensure/get/update."""
from __future__ import annotations

import os
from typing import Any

from . import supabase_client as sb


def _free_credits() -> int:
    try:
        return int(os.getenv("PLAN_FREE_CREDITS", "100"))
    except ValueError:
        return 100


async def get_user(user_id: str) -> dict[str, Any] | None:
    return await sb.select_one("users", filters={"id": user_id})


async def ensure_user(*, user_id: str, email: str, display_name: str | None = None) -> dict[str, Any]:
    """Upsert by id. Existing users keep their plan/credits; new users get free tier."""
    existing = await get_user(user_id)
    if existing:
        # Only refresh fields that might change in Supabase Auth.
        if existing.get("email") != email or (display_name and existing.get("display_name") != display_name):
            updated = await sb.update(
                "users",
                filters={"id": user_id},
                values={
                    "email": email,
                    **({"display_name": display_name} if display_name else {}),
                },
            )
            return updated or existing
        return existing

    return await sb.upsert(
        "users",
        {
            "id": user_id,
            "email": email,
            "display_name": display_name,
            "plan": "free",
            "credits": _free_credits(),
        },
        on_conflict="id",
    )


async def add_credits(user_id: str, amount: int) -> dict[str, Any] | None:
    if amount <= 0:
        raise ValueError("amount must be positive")
    user = await get_user(user_id)
    if not user:
        return None
    return await sb.update(
        "users",
        filters={"id": user_id},
        values={"credits": int(user["credits"]) + amount},
    )


async def set_plan(user_id: str, plan: str) -> dict[str, Any] | None:
    if plan not in ("free", "pro", "team"):
        raise ValueError(f"invalid plan: {plan}")
    return await sb.update("users", filters={"id": user_id}, values={"plan": plan})


async def consume_credits(user_id: str, amount: int, action: str, metadata: dict | None = None) -> int:
    """Atomic credit consumption via RPC. Raises SupabaseError on insufficient_credits."""
    return await sb.rpc(
        "consume_credits",
        {
            "p_user_id": user_id,
            "p_amount": amount,
            "p_action": action,
            "p_metadata": metadata or {},
        },
    )
