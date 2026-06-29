"""
Model Router — picks the right LLM for the task.

Philosophy (local-first, founder decision — наша модель основная):
  Simple / fast task     → local model (Ollama, free, private)
  Medium / standard      → local model → Claude Haiku fallback
  Complex / synthesis    → local model → Claude Sonnet fallback
  Critical / reasoning   → Claude Sonnet (best available) → local fallback

Falls back across tiers when the preferred one is unavailable.
"""
from __future__ import annotations
import logging
import os
import time
from enum import Enum

from orynd_core.services.llm.base import LLMProvider

log = logging.getLogger(__name__)

# Ollama availability is checked at most once per TTL — the old code paid a
# 1s HTTP timeout on every routing decision when Ollama was down.
_OLLAMA_CHECK_TTL = 30.0
_ollama_cache: dict[str, tuple[float, bool]] = {}


class TaskComplexity(str, Enum):
    SIMPLE   = "simple"    # keyword extraction, classification, routing
    MEDIUM   = "medium"    # summarization, synthesis, Q&A
    COMPLEX  = "complex"   # multi-step reasoning, research synthesis
    CRITICAL = "critical"  # architecture decisions, full CAD reasoning


def get_provider(complexity: TaskComplexity = TaskComplexity.MEDIUM) -> LLMProvider | None:
    """
    Return the most appropriate LLMProvider for the given complexity.
    Returns None if no model available at requested tier (caller uses algorithm fallback).

    Priority order:
      Ollama (local, free) → Groq (cloud, free, fast) → Gemini (cloud, free) → Claude (cloud, paid) → None
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    local_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    # Heavy local tier — opt-in. Measured on this hardware: qwen2.5-coder:7b gives
    # marginally better synthesis but ~100s+/call (exceeds the research safety
    # budget), while llama3.2:3b does research at conf≈0.8 in ~67s. So default to
    # the fast model; set OLLAMA_HEAVY_MODEL=qwen2.5-coder:7b to trade speed for depth.
    local_heavy = os.getenv("OLLAMA_HEAVY_MODEL", local_model)
    # gemini-1.5-* were retired by Google (404). 2.0-flash is the current free tier.
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

    if complexity == TaskComplexity.SIMPLE:
        if _ollama_available(ollama_url):
            return _local_provider(ollama_url, model=local_model)
        if groq_key:
            return _groq_provider(groq_key, groq_model)
        if gemini_key:
            return _gemini_provider(gemini_key, gemini_model)
        if anthropic_key:
            return _claude_provider(anthropic_key, "claude-haiku-4-5-20251001")
        return None

    if complexity == TaskComplexity.MEDIUM:
        if _ollama_available(ollama_url):
            return _local_provider(ollama_url, model=local_model)
        if groq_key:
            return _groq_provider(groq_key, groq_model)
        if gemini_key:
            return _gemini_provider(gemini_key, gemini_model)
        if anthropic_key:
            return _claude_provider(anthropic_key, "claude-haiku-4-5-20251001")
        return None

    if complexity == TaskComplexity.COMPLEX:
        if _ollama_available(ollama_url):
            return _local_provider(ollama_url, model=local_heavy)
        if groq_key:
            return _groq_provider(groq_key, "llama-3.1-70b-versatile")
        if gemini_key:
            return _gemini_provider(gemini_key, "gemini-2.0-flash")
        if anthropic_key:
            return _claude_provider(anthropic_key, "claude-sonnet-4-5")
        return None

    if complexity == TaskComplexity.CRITICAL:
        # Prefer strongest cloud model for critical decisions
        if anthropic_key:
            return _claude_provider(anthropic_key, "claude-sonnet-4-5")
        if groq_key:
            return _groq_provider(groq_key, "llama-3.1-70b-versatile")
        if gemini_key:
            return _gemini_provider(gemini_key, "gemini-2.0-flash")
        if _ollama_available(ollama_url):
            return _local_provider(ollama_url, model=local_heavy)
        return None

    return None


def assess_complexity(task_description: str) -> TaskComplexity:
    """
    Heuristic complexity assessment from task description.
    Used when no explicit complexity is provided.
    """
    desc = task_description.lower()

    # Keywords that signal complexity
    complex_signals = [
        "research", "analyze", "synthesize", "design", "architect",
        "compare", "evaluate", "comprehensive", "deep", "full",
        "ресёрч", "анализ", "синтез", "проект", "архитектур",
    ]
    simple_signals = [
        "extract", "classify", "route", "keyword", "tag", "label",
        "категори", "ключевы", "маршрут",
    ]

    if any(s in desc for s in complex_signals):
        return TaskComplexity.COMPLEX
    if any(s in desc for s in simple_signals):
        return TaskComplexity.SIMPLE
    return TaskComplexity.MEDIUM


def _claude_provider(api_key: str, model: str) -> LLMProvider:
    from orynd_core.services.llm.claude import ClaudeProvider
    return ClaudeProvider(api_key=api_key, model=model)


def _local_provider(base_url: str, model: str) -> LLMProvider:
    from orynd_core.services.llm.local import LocalProvider
    return LocalProvider(base_url=base_url, model=model)


def _groq_provider(api_key: str, model: str) -> LLMProvider:
    from orynd_core.services.llm.groq import GroqProvider
    return GroqProvider(api_key=api_key, model=model)


def _gemini_provider(api_key: str, model: str) -> LLMProvider:
    from orynd_core.services.llm.gemini import GeminiProvider
    return GeminiProvider(api_key=api_key, model=model)


def _ollama_available(url: str) -> bool:
    """Check if Ollama is running locally (cached for _OLLAMA_CHECK_TTL s)."""
    if os.environ.get("ORYND_SKIP_OLLAMA_CHECK"):
        return False
    now = time.monotonic()
    cached = _ollama_cache.get(url)
    if cached and now - cached[0] < _OLLAMA_CHECK_TTL:
        return cached[1]

    import httpx
    try:
        r = httpx.get(f"{url}/api/tags", timeout=1.0)
        ok = r.status_code == 200
    except Exception:
        ok = False
    _ollama_cache[url] = (now, ok)
    return ok
