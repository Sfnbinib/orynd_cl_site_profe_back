"""
LLM Provider abstraction layer.
Swap models by changing one line — never tied to a single vendor.
"""
from .base import LLMProvider, LLMMessage, LLMResponse
from .claude import ClaudeProvider
from .local import LocalProvider

__all__ = ["LLMProvider", "LLMMessage", "LLMResponse", "ClaudeProvider", "LocalProvider"]
