"""supabase_sink — persists pipeline events to Supabase.

Wire #7: subscribes to event_bus topics and writes to Supabase tables.

Topics:
  action.recorded  → table: orynd_action_logs
  credits.consumed → table: orynd_credit_events

Activated in api/main.py lifespan only when SUPABASE_URL is set.
Never raises — errors are logged, main flow is never blocked.

Tables use prefix "orynd_" to avoid conflicts with existing Supabase schema.
"""
from __future__ import annotations

import logging
import os
import time

log = logging.getLogger("orynd.supabase_sink")

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_ANON_KEY", "")
    if not url or not key or "your-project" in url:
        return None
    try:
        from supabase import create_client
        _client = create_client(url, key)
        log.info("[supabase_sink] connected to %s", url)
        return _client
    except Exception as e:
        log.warning("[supabase_sink] cannot init client: %s", e)
        return None


async def on_action_recorded(entry: dict) -> None:
    """event_bus 'action.recorded' → table orynd_action_logs"""
    client = _get_client()
    if not client:
        return
    try:
        import asyncio
        row = {
            "tool": entry.get("tool", ""),
            "session_id": entry.get("session_id", "anonymous"),
            "result_summary": str(entry.get("result", {}))[:500],
            "ts": entry.get("ts", time.time()),
        }
        await asyncio.to_thread(
            lambda: client.table("orynd_action_logs").insert(row).execute()
        )
    except Exception as e:
        log.debug("[supabase_sink] orynd_action_logs insert failed: %s", e)


async def on_credits_consumed(entry: dict) -> None:
    """event_bus 'credits.consumed' → table orynd_credit_events"""
    client = _get_client()
    if not client:
        return
    try:
        import asyncio
        row = {
            "session_id": entry.get("session_id", "anonymous"),
            "tool": entry.get("tool", ""),
            "cost": entry.get("cost", 0),
            "session_total": entry.get("session_total", 0),
            "ts": time.time(),
        }
        await asyncio.to_thread(
            lambda: client.table("orynd_credit_events").insert(row).execute()
        )
    except Exception as e:
        log.debug("[supabase_sink] orynd_credit_events insert failed: %s", e)
