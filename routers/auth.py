"""Auth endpoints — /api/auth/me, /api/auth/sync, /api/auth/consents."""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field

from orynd_core.auth import UserContext, current_user
from orynd_core.auth import supabase_client as sb
from orynd_core.auth import users as users_repo

router = APIRouter(prefix="/api/auth", tags=["auth"])


class MeResponse(BaseModel):
    id: str
    email: str | None
    display_name: str | None
    plan: str
    credits: int


class SyncRequest(BaseModel):
    email: EmailStr
    display_name: str | None = Field(default=None, max_length=120)


class ConsentRequest(BaseModel):
    training_opt_in: bool = True
    marketing_opt_in: bool = False


def _user_to_me(row: dict[str, Any]) -> MeResponse:
    return MeResponse(
        id=row["id"],
        email=row.get("email"),
        display_name=row.get("display_name"),
        plan=row.get("plan", "free"),
        credits=int(row.get("credits", 0)),
    )


@router.get("/me", response_model=MeResponse)
async def me(user: UserContext = Depends(current_user)) -> MeResponse:
    row = await users_repo.get_user(user.id)
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not synced — call /api/auth/sync first")
    return _user_to_me(row)


@router.post("/sync", response_model=MeResponse)
async def sync(body: SyncRequest, user: UserContext = Depends(current_user)) -> MeResponse:
    # JWT email is the source of truth — body.email must match to prevent spoofing.
    if user.email and user.email.lower() != body.email.lower():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "email mismatch with token")
    row = await users_repo.ensure_user(
        user_id=user.id,
        email=user.email or body.email,
        display_name=body.display_name,
    )
    return _user_to_me(row)


@router.post("/consents")
async def upsert_consents(
    body: ConsentRequest,
    request: Request,
    user: UserContext = Depends(current_user),
) -> dict[str, str]:
    await sb.upsert(
        "consents",
        {
            "user_id": user.id,
            "training_opt_in": body.training_opt_in,
            "marketing_opt_in": body.marketing_opt_in,
            "terms_version": os.getenv("TERMS_VERSION", "unversioned"),
            "privacy_version": os.getenv("PRIVACY_VERSION", "unversioned"),
            "ip_at_accept": request.client.host if request.client else None,
            "user_agent_at_accept": request.headers.get("user-agent"),
        },
        on_conflict="user_id",
    )
    return {"status": "ok"}


@router.get("/consents")
async def get_consents(user: UserContext = Depends(current_user)) -> dict[str, Any]:
    row = await sb.select_one("consents", filters={"user_id": user.id})
    return row or {
        "user_id": user.id,
        "training_opt_in": None,
        "marketing_opt_in": None,
        "terms_version": None,
        "privacy_version": None,
    }
