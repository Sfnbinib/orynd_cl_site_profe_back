"""
Local model provider stub.
Drop-in replacement for ClaudeProvider when running own model
(Ollama, llama.cpp, own fine-tune, etc.)

Usage:
    provider = LocalProvider(base_url="http://localhost:11434", model="llama3")
    # agents work identically — zero code changes
"""
from __future__ import annotations
import json
from typing import AsyncIterator

import httpx

from .base import LLMProvider, LLMMessage, LLMResponse


class LocalProvider(LLMProvider):
    """
    Ollama-compatible local provider.
    Works with: Ollama, LM Studio, llama-cpp-python server, vLLM.
    """
    name = "local"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def complete(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        for m in messages:
            msgs.append({"role": m.role, "content": m.content})

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": msgs,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            )
            data = resp.json()

        text = data.get("message", {}).get("content", "")
        return LLMResponse(content=text, model=self.model)

    async def stream(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        for m in messages:
            msgs.append({"role": m.role, "content": m.content})

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": msgs,
                    "stream": True,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        text = chunk.get("message", {}).get("content", "")
                        if text:
                            yield text
