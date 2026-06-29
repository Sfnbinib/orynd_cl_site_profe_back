"""structlog + stdlib logging configuration.

Call :func:`configure_logging` exactly once at process start (see
``api/main.py``). Subsequent loggers are obtained via :func:`get_logger`.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

import structlog
from pythonjsonlogger import jsonlogger

_CONFIGURED = False


def _log_dir() -> Path:
    base = Path(os.environ.get("ORYND_LOG_DIR", "~/.orynd/logs")).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    return base


def configure_logging(
    level: str | None = None,
    *,
    enable_file: bool | None = None,
) -> None:
    """Idempotent. Sets up JSON logging to stdout (+ rotating file when enabled)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = (level or os.environ.get("ORYND_LOG_LEVEL", "INFO")).upper()
    enable_file = (
        enable_file
        if enable_file is not None
        else os.environ.get("ORYND_LOG_FILE", "1") not in {"0", "false", "False"}
    )

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if enable_file:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                _log_dir() / "backend.log",
                maxBytes=50_000_000,
                backupCount=5,
            )
            handlers.append(file_handler)
        except OSError:
            # Read-only fs or sandbox — stdout only.
            pass

    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    for handler in handlers:
        handler.setFormatter(formatter)

    root = logging.getLogger()
    # Wipe and re-attach so re-runs in pytest behave.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level, logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name) if name else structlog.get_logger()
