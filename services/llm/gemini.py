"""Gemini provider — Google Generative AI via REST API.

Uses gemini-1.5-flash (free tier) by default.
Falls back gracefully if the key is missing or the call fails.

No extra dependencies — uses httpx which is already in requirements.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .base import LLMMessage, LLMProvider, LLMResponse

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _build_contents(messages: list[LLMMessage], system: str) -> tuple[list[dict], str | None]:
    """Convert LLMMessage list to Gemini `contents` format."""
    contents: list[dict] = []
    sys_instruction: str | None = system or None

    for m in messages:
        role = "user" if m.role in ("user", "system") else "model"
        parts: list[dict] = [{"text": m.content}]
        if m.image_b64:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": m.image_b64,
                }
            })
        contents.append({"role": role, "parts": parts})

    return contents, sys_instruction


class GeminiProvider(LLMProvider):
    """Google Gemini provider (REST, no SDK dependency)."""

    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        self.api_key = api_key
        self.model = model

    async def complete(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        contents, sys_instruction = _build_contents(messages, system)

        body: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if sys_instruction:
            body["systemInstruction"] = {"parts": [{"text": sys_instruction}]}

        url = f"{_BASE}/{self.model}:generateContent?key={self.api_key}"
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()

        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            content=text,
            model=self.model,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        contents, sys_instruction = _build_contents(messages, system)

        body: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if sys_instruction:
            body["systemInstruction"] = {"parts": [{"text": sys_instruction}]}

        url = f"{_BASE}/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=body) as resp:
                resp.raise_for_status()
                async for raw in resp.aiter_lines():
                    if not raw.startswith("data:"):
                        continue
                    payload = raw[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(payload)
                        text = (
                            chunk.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                        )
                        if text:
                            yield text
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
