"""Thin Supabase REST client for backend operations using the service role key.

We don't pull the full supabase-py SDK — we hit PostgREST directly with httpx
to stay light and avoid version churn.
"""
from __future__ import annotations

import os
from typing import Any

import httpx


class SupabaseError(Exception):
    pass


def _base_url() -> str:
    url = os.getenv("SUPABASE_URL")
    if not url:
        raise SupabaseError("SUPABASE_URL not configured")
    return url.rstrip("/")


def _service_key() -> str:
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        raise SupabaseError("SUPABASE_SERVICE_ROLE_KEY not configured")
    return key


def _headers() -> dict[str, str]:
    key = _service_key()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


async def select_one(table: str, *, filters: dict[str, str]) -> dict[str, Any] | None:
    """SELECT one row from `table` matching filters. None if not found."""
    params = {k: f"eq.{v}" for k, v in filters.items()}
    params["limit"] = "1"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            f"{_base_url()}/rest/v1/{table}",
            params=params,
            headers=_headers(),
        )
    if r.status_code >= 400:
        raise SupabaseError(f"select_one {table} failed: {r.status_code} {r.text}")
    rows = r.json()
    return rows[0] if rows else None


async def upsert(table: str, row: dict[str, Any], *, on_conflict: str) -> dict[str, Any]:
    """UPSERT and return the resulting row."""
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(
            f"{_base_url()}/rest/v1/{table}",
            params={"on_conflict": on_conflict},
            json=row,
            headers={**_headers(), "Prefer": "return=representation,resolution=merge-duplicates"},
        )
    if r.status_code >= 400:
        raise SupabaseError(f"upsert {table} failed: {r.status_code} {r.text}")
    rows = r.json()
    if not rows:
        raise SupabaseError(f"upsert {table} returned empty")
    return rows[0]


async def insert(table: str, row: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(
            f"{_base_url()}/rest/v1/{table}",
            json=row,
            headers=_headers(),
        )
    if r.status_code >= 400:
        raise SupabaseError(f"insert {table} failed: {r.status_code} {r.text}")
    rows = r.json()
    return rows[0] if rows else {}


async def update(table: str, *, filters: dict[str, str], values: dict[str, Any]) -> dict[str, Any] | None:
    params = {k: f"eq.{v}" for k, v in filters.items()}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.patch(
            f"{_base_url()}/rest/v1/{table}",
            params=params,
            json=values,
            headers=_headers(),
        )
    if r.status_code >= 400:
        raise SupabaseError(f"update {table} failed: {r.status_code} {r.text}")
    rows = r.json()
    return rows[0] if rows else None


async def rpc(fn: str, args: dict[str, Any]) -> Any:
    """Call a Postgres function via PostgREST."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{_base_url()}/rest/v1/rpc/{fn}",
            json=args,
            headers=_headers(),
        )
    if r.status_code >= 400:
        raise SupabaseError(f"rpc {fn} failed: {r.status_code} {r.text}")
    return r.json()
