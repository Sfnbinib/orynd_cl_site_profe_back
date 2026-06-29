"""
Base LLM provider interface.
Any model (Claude, GPT, Ollama, own) implements this → agents never depend on vendor.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class LLMMessage:
    role: str          # "user" | "assistant" | "system"
    content: str
    image_b64: str | None = None   # base64-encoded image (vision)


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = "end_turn"

    @property
    def cost_usd(self) -> float:
        """Approximate cost — override per provider."""
        return 0.0


class LLMProvider(ABC):
    """
    Abstract LLM provider.
    Implement this to add any model: Claude, GPT, Gemini, Ollama, own fine-tune.
    """

    name: str = "base"

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Single-turn completion. Returns full response."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Streaming completion. Yields text chunks."""
        ...

    async def complete_json(
        self,
        messages: list[LLMMessage],
        system: str = "",
        max_tokens: int = 1024,
    ) -> str:
        """
        Convenience: call complete(), return content string.
        Agents parse JSON themselves — keeps provider thin.
        """
        resp = await self.complete(messages, system=system, max_tokens=max_tokens)
        return resp.content
