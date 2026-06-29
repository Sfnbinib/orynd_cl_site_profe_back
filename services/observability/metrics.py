"""Prometheus metric definitions.

Spec: MONITORING_OBSERVABILITY.md § Metrics catalog.

All metrics live in the default :class:`prometheus_client.CollectorRegistry`
and are exposed via ``/system/metrics``.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---- HTTP request metrics --------------------------------------------------

http_requests_total = Counter(
    "orynd_http_requests_total",
    "Total HTTP requests",
    ["method", "route", "status"],
)
http_request_duration_seconds = Histogram(
    "orynd_http_request_duration_seconds",
    "HTTP request duration",
    ["route"],
)

# ---- Library metrics -------------------------------------------------------

library_articles_total = Counter(
    "orynd_library_articles_total",
    "Articles created",
    ["layer", "authored_by"],
)
library_searches_total = Counter(
    "orynd_library_searches_total",
    "Search queries",
    ["type"],
)
library_skill_invocations_total = Counter(
    "orynd_library_skill_invocations_total",
    "Skill calls",
    ["slug"],
)
library_stage_transitions_total = Counter(
    "orynd_library_stage_transitions_total",
    "Stage promotions",
    ["from_stage", "to_stage"],
)

# ---- Deep Research metrics ------------------------------------------------

research_sessions_total = Counter(
    "orynd_research_sessions_total",
    "Research sessions",
    ["mode"],
)
research_duration_seconds = Histogram(
    "orynd_research_duration_seconds",
    "Research session duration",
    ["mode"],
)
research_tokens_consumed = Counter(
    "orynd_research_tokens_total",
    "Tokens consumed",
    ["model"],
)

# ---- AI Model 4 metrics ---------------------------------------------------

ai_model_4_invocations_total = Counter(
    "orynd_ai_model_4_invocations_total",
    "AI Model 4 calls",
    ["pass", "result"],
)
ai_model_4_quality = Gauge(
    "orynd_ai_model_4_quality",
    "Quality score of latest run (0..1)",
)

# ---- External services -----------------------------------------------------

external_api_calls_total = Counter(
    "orynd_external_api_calls_total",
    "External API calls",
    ["service", "status"],
)
external_api_latency_seconds = Histogram(
    "orynd_external_api_latency_seconds",
    "External API latency",
    ["service"],
)

# ---- Circuit breakers ------------------------------------------------------

circuit_state = Gauge(
    "orynd_circuit_state",
    "Circuit state (0=closed, 1=half_open, 2=open)",
    ["name"],
)
circuit_failures_total = Counter(
    "orynd_circuit_failures_total",
    "Circuit failures",
    ["name"],
)

# ---- Background tasks ------------------------------------------------------

background_tasks_active = Gauge(
    "orynd_background_tasks_active",
    "Active background tasks",
    ["type"],
)
background_tasks_completed_total = Counter(
    "orynd_background_tasks_completed_total",
    "Completed background tasks",
    ["type", "status"],
)

# ---- User metrics ----------------------------------------------------------

active_users_gauge = Gauge("orynd_active_users", "Currently active users")
user_session_duration_seconds = Histogram(
    "orynd_user_session_duration_seconds",
    "User session duration",
)


_CIRCUIT_STATE_VALUE = {"closed": 0, "half_open": 1, "open": 2}


def refresh_circuit_metrics() -> None:
    """Pull current circuit-breaker state into the Gauge labels.

    Call before scraping ``/system/metrics`` so values are fresh.
    """
    from orynd_core.services.resilience.circuit_breaker import breakers

    for name, breaker in breakers.items():
        circuit_state.labels(name=name).set(_CIRCUIT_STATE_VALUE[breaker.state.value])


__all__ = [
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
    "background_tasks_completed_total",
    "active_users_gauge",
    "user_session_duration_seconds",
    "refresh_circuit_metrics",
]
