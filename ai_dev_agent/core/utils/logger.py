"""Logger configuration utilities with correlation ID support."""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime

_CORRELATION_ID = contextvars.ContextVar("correlation_id", default="-")


class CorrelationIdFilter(logging.Filter):
    """Inject the active correlation ID into each log record."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - delegation
        record.correlation_id = _CORRELATION_ID.get()
        return True


class StructuredFormatter(logging.Formatter):
    """Emit structured JSON log records."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting
        payload = {
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
        }
        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(level: str = "INFO", *, structured: bool = False) -> None:
    """Configure root logging for the CLI agent."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.addFilter(CorrelationIdFilter())
    if structured:
        handler.setFormatter(StructuredFormatter())
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | cid=%(correlation_id)s | %(message)s"
        )
        handler.setFormatter(formatter)
    root.addHandler(handler)


def set_correlation_id(value: str | None) -> None:
    """Set the active correlation ID for subsequent log records."""
    _CORRELATION_ID.set(value or "-")


def get_correlation_id() -> str:
    """Return the active correlation ID for the current context."""
    return _CORRELATION_ID.get()


def get_logger(name: str | None = None) -> logging.Logger:
    """Retrieve a module-level logger."""
    return logging.getLogger(name if name else "ai_dev_agent")


__all__ = [
    "configure_logging",
    "get_correlation_id",
    "get_logger",
    "set_correlation_id",
]
