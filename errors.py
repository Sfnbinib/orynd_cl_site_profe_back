"""ORYND error taxonomy.

Every domain raises a subclass of :class:`OryndError`. The global FastAPI
exception handler (see ``services/observability/middleware.py``) converts these
to RFC-style JSON envelopes with stable ``code`` strings the frontend can switch on.
"""

from __future__ import annotations

from typing import Any


class OryndError(Exception):
    """Base for all app errors.

    Attributes:
        code: stable machine-readable identifier (e.g. ``library.topic_not_found``)
        http_status: HTTP status code the global handler should return
        user_message: safe-to-display message (no PII, no internals)
        details: extra structured context for logs and dev-mode responses
    """

    code: str = "orynd.unknown"
    http_status: int = 500
    user_message: str = "Something went wrong"

    def __init__(
        self,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.details = details or {}
        super().__init__(message or self.user_message)


# ---- generic ---------------------------------------------------------------


class ValidationFailedError(OryndError):
    code = "orynd.validation_failed"
    http_status = 422
    user_message = "Input failed validation"


class UnauthorizedError(OryndError):
    code = "orynd.unauthorized"
    http_status = 401
    user_message = "Not authenticated"


class ForbiddenError(OryndError):
    code = "orynd.forbidden"
    http_status = 403
    user_message = "Forbidden"


class NotFoundError(OryndError):
    code = "orynd.not_found"
    http_status = 404
    user_message = "Resource not found"


class RateLimitedError(OryndError):
    code = "orynd.rate_limited"
    http_status = 429
    user_message = "Rate limit exceeded"


# ---- library ---------------------------------------------------------------


class TopicNotFoundError(OryndError):
    code = "library.topic_not_found"
    http_status = 404
    user_message = "Topic not found"


class InsufficientStageError(OryndError):
    code = "library.insufficient_stage"
    http_status = 422
    user_message = "Topic stage too low for this operation"


class ArticleNotFoundError(OryndError):
    code = "library.article_not_found"
    http_status = 404
    user_message = "Article not found"


# ---- skills ----------------------------------------------------------------


class SkillNotFoundError(OryndError):
    code = "skills.not_found"
    http_status = 404
    user_message = "Skill not found"


class SkillExecutionError(OryndError):
    code = "skills.execution_failed"
    http_status = 500
    user_message = "Skill execution failed"


# ---- resilience ------------------------------------------------------------


class CircuitOpenError(OryndError):
    code = "resilience.circuit_open"
    http_status = 503
    user_message = "Upstream service unavailable"


class TimeoutExceededError(OryndError):
    code = "resilience.timeout"
    http_status = 504
    user_message = "Operation timed out"


# ---- mesh / model 4 -------------------------------------------------------


class MeshLoadError(OryndError):
    code = "mesh.load_failed"
    http_status = 422
    user_message = "Could not read mesh file"


class DecompositionError(OryndError):
    code = "mesh.decomposition_failed"
    http_status = 500
    user_message = "Mesh decomposition failed"


# ---- external --------------------------------------------------------------


class ExternalAPIError(OryndError):
    code = "external.api_error"
    http_status = 502
    user_message = "External service error"


__all__ = [
    "OryndError",
    "ValidationFailedError",
    "UnauthorizedError",
    "ForbiddenError",
    "NotFoundError",
    "RateLimitedError",
    "TopicNotFoundError",
    "InsufficientStageError",
    "ArticleNotFoundError",
    "SkillNotFoundError",
    "SkillExecutionError",
    "CircuitOpenError",
    "TimeoutExceededError",
    "MeshLoadError",
    "DecompositionError",
    "ExternalAPIError",
]
