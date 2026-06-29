"""Groq provider — OpenAI-compatible REST API.

Free tier: llama-3.1-70b-versatile, very fast inference, no geo restrictions.
Falls back gracefully if key missing or call fails.

No extra dependencies — uses httpx (already in requirements).
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .base import LLMMessage, LLMProvider, LLMResponse

_BASE = "https://api.groq.com/openai/v1"


def _build_messages(messages: list[LLMMessage], system: str) -> list[dict]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    for m in messages:
        role = "assistant" if m.role == "assistant" else "user"
        msgs.append({"role": role, "content": m.content})
    return msgs


class GroqProvider(LLMProvider):
    """Groq cloud provider (OpenAI-compatible, free tier)."""

    name = "groq"

    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile") -> None:
        self.api_key = api_key
        self.model = model

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def complete(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        msgs = _build_messages(messages, system)
        body = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{_BASE}/chat/completions", json=body, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            content=text,
            model=self.model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        msgs = _build_messages(messages, system)
        body = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", f"{_BASE}/chat/completions", json=body, headers=self._headers()
            ) as resp:
                resp.raise_for_status()
                async for raw in resp.aiter_lines():
                    if not raw.startswith("data:"):
                        continue
                    payload = raw[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            yield text
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
