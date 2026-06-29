"""
Claude provider — Anthropic API.
Default model: claude-haiku-4-5 (fast, cheap, vision-capable).
Swap to claude-opus-4 for complex reasoning — one line change.

Anti-piracy / IP protection:
  When ORYND_ANTHROPIC_PROXY is set (e.g. https://oryndai.com/api/anthropic-proxy)
  ALL traffic goes through ORYND's server. The real API key never lives in
  the client binary. Server enforces per-license quota + rate limit.

  Direct BYOK is allowed only for Pro+ tier (handled via license decorator
  on the caller); when proxy is set, BYOK still tunnels through proxy.
"""
from __future__ import annotations
import os
from typing import AsyncIterator

import anthropic

from .base import LLMProvider, LLMMessage, LLMResponse


# Cost table (per 1M tokens, USD)
_COST = {
    "claude-haiku-4-5":        (0.80,  4.00),
    "claude-haiku-4-5-20251001":(0.80,  4.00),
    "claude-sonnet-4-5":       (3.00, 15.00),
    "claude-opus-4":           (15.0, 75.00),
}


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
        base_url: str | None = None,
    ) -> None:
        """Construct the provider.

        Args:
            api_key: Anthropic key. When proxy is used, this can be a
                license-issued short-lived token instead of a raw key.
            model: claude model id.
            base_url: override Anthropic endpoint. Set automatically from
                ``ORYND_ANTHROPIC_PROXY`` env when present, so the client
                cannot bypass our proxy by editing config.
        """
        self.model = model
        resolved_base = base_url or os.getenv("ORYND_ANTHROPIC_PROXY") or None
        self.base_url = resolved_base  # exposed for tests + diagnostics

        # When the proxy is in front, we want the license JWT in the headers
        # so the server can attribute usage. The proxy verifies + forwards.
        default_headers: dict[str, str] = {}
        license_jwt = os.getenv("ORYND_LICENSE_JWT")
        if license_jwt and resolved_base:
            default_headers["X-Orynd-License"] = license_jwt
        client_id = os.getenv("ORYND_CLIENT_ID")
        if client_id and resolved_base:
            default_headers["X-Orynd-Client-Id"] = client_id

        kwargs: dict = dict(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", ""),
        )
        if resolved_base:
            kwargs["base_url"] = resolved_base
        if default_headers:
            kwargs["default_headers"] = default_headers

        self._client = anthropic.AsyncAnthropic(**kwargs)

    def _build_messages(self, messages: list[LLMMessage]) -> list[dict]:
        out = []
        for m in messages:
            if m.role == "system":
                continue  # system goes via system= param
            if m.image_b64:
                # Vision message
                out.append({
                    "role": m.role,
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": m.image_b64,
                            },
                        },
                        {"type": "text", "text": m.content or ""},
                    ],
                })
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    async def complete(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=self._build_messages(messages),
        )
        if system:
            kwargs["system"] = system

        resp = await self._client.messages.create(**kwargs)
        text = resp.content[0].text if resp.content else ""

        inp, out = resp.usage.input_tokens, resp.usage.output_tokens
        in_cost, out_cost = _COST.get(self.model, (0, 0))
        cost = (inp * in_cost + out * out_cost) / 1_000_000

        result = LLMResponse(
            content=text,
            model=self.model,
            input_tokens=inp,
            output_tokens=out,
            stop_reason=resp.stop_reason or "end_turn",
        )
        result.__dict__["_cost_usd"] = cost
        return result

    async def stream(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=self._build_messages(messages),
        )
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
