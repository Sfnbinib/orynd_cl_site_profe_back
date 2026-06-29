"""Observability — metrics + traces + middleware."""

from orynd_core.services.observability.langfuse_client import (
    langfuse_enabled,
    traced_llm_call,
)
from orynd_core.services.observability.metrics import (
    ai_model_4_invocations_total,
    ai_model_4_quality,
    background_tasks_active,
    circuit_failures_total,
    circuit_state,
    external_api_calls_total,
    external_api_latency_seconds,
    http_request_duration_seconds,
    http_requests_total,
    library_articles_total,
    library_searches_total,
    library_skill_invocations_total,
    library_stage_transitions_total,
    research_duration_seconds,
    research_sessions_total,
    research_tokens_consumed,
)
from orynd_core.services.observability.middleware import (
    install_middleware,
    orynd_exception_handler,
    unexpected_exception_handler,
)

__all__ = [
    "install_middleware",
    "orynd_exception_handler",
    "unexpected_exception_handler",
    "langfuse_enabled",
    "traced_llm_call",
    # metrics
    "http_requests_total",
    "http_request_duration_seconds",
    "library_articles_total",
    "library_searches_total",
    "library_skill_invocations_total",
    "library_stage_transitions_total",
    "research_sessions_total",
    "research_duration_seconds",
    "research_tokens_consumed",
    "ai_model_4_invocations_total",
    "ai_model_4_quality",
    "external_api_calls_total",
    "external_api_latency_seconds",
    "circuit_state",
    "circuit_failures_total",
    "background_tasks_active",
]
