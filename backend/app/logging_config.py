"""
logging_config.py — Structured Logging Setup
=============================================
CONCEPT: Structured logging vs plain text logging

  PLAIN TEXT (hard to parse):
    "2024-01-01 12:00:00 ERROR Failed to process request"

  STRUCTURED JSON (machine-readable, searchable):
    {
      "timestamp": "2024-01-01T12:00:00Z",
      "level": "error",
      "event": "Failed to process request",
      "request_id": "abc-123",
      "user_id": 42,
      "endpoint": "/api/chat",
      "duration_ms": 1523
    }

WHY STRUCTURED LOGGING?
  - Each field is queryable in log aggregators (Elasticsearch, Grafana Loki)
  - Easy to correlate logs across services using request_id
  - Agents can log their reasoning steps as structured events
  - Production debugging becomes much easier

We use `structlog` which gives us:
  - Automatic context binding (add user_id once, it appears in all subsequent logs)
  - Console renderer for dev (pretty, colored)
  - JSON renderer for production (machine-readable)
"""

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

from app.config import get_settings

settings = get_settings()


def setup_logging() -> None:
    """
    Configure structlog and standard library logging to work together.

    Call this ONCE at application startup (in main.py lifespan).

    Structlog wraps stdlib logging, so any third-party library that uses
    standard logging (FastAPI, SQLAlchemy, etc.) will also output structured logs.
    """

    # ── Step 1: Configure standard library logging ────────────────────────────
    # structlog integrates with stdlib logging so all loggers go through structlog
    log_level = getattr(logging, settings.log_level.upper(), logging.DEBUG)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    # Add file handler if log file path is configured
    if settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        format="%(message)s",   # structlog handles formatting
        level=log_level,
        handlers=handlers,
        force=True,             # Override any existing config
    )

    # Silence noisy third-party loggers in development
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    # ── Step 2: Choose renderer based on environment ──────────────────────────
    # Development: Pretty colored console output
    # Production: JSON output for log aggregators
    if settings.log_format == "json" or settings.is_production:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # ── Step 3: Configure structlog processors pipeline ───────────────────────
    # Processors are applied left-to-right to each log event.
    # Think of them as middleware for your log messages.
    structlog.configure(
        processors=[
            # 1. Add log level to the event dict
            structlog.stdlib.add_log_level,
            # 2. Add logger name (module path)
            structlog.stdlib.add_logger_name,
            # 3. Add timestamp in ISO 8601 format
            structlog.processors.TimeStamper(fmt="iso"),
            # 4. Format any exceptions with full traceback
            structlog.processors.format_exc_info,
            # 5. Convert bytes to strings
            structlog.processors.UnicodeDecoder(),
            # 6. Final render (JSON or colored console)
            renderer,
        ],
        # Use stdlib logging as the output backend
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a named logger instance with structured logging.

    Usage:
        logger = get_logger(__name__)
        logger.info("Ticket created", ticket_id=42, user_id=1)
        logger.error("DB error", error=str(e), query="SELECT ...")

    Args:
        name: Usually __name__ from the calling module
    """
    return structlog.get_logger(name)


# ─── Request Context Logger ───────────────────────────────────────────────────
class RequestLogger:
    """
    Context-aware logger that automatically includes request metadata.

    CONCEPT: Context binding — bind once, log everywhere.
    Instead of passing request_id to every log call, you bind it once
    and all subsequent logs in that request automatically include it.

    Usage (in FastAPI middleware):
        req_logger = RequestLogger(request_id="abc-123", user_id=42)
        req_logger.info("Processing chat request", endpoint="/api/chat")
        # Output: {..., "request_id": "abc-123", "user_id": 42, "endpoint": "/api/chat"}
    """

    def __init__(self, request_id: str, user_id: int | None = None, **extra: Any):
        self._logger = get_logger("request").bind(
            request_id=request_id,
            user_id=user_id,
            **extra,
        )

    def info(self, event: str, **kwargs: Any) -> None:
        self._logger.info(event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._logger.error(event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._logger.warning(event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._logger.debug(event, **kwargs)


# ─── Agent Step Logger ────────────────────────────────────────────────────────
def log_agent_step(
    agent_name: str,
    step: str,
    session_id: str,
    **details: Any,
) -> None:
    """
    Log an agent reasoning/action step with structured context.

    CONCEPT: This is critical for debugging multi-agent systems.
    When you have 5 agents taking turns, you need to know exactly
    which agent did what, in what order, and with what data.

    Usage:
        log_agent_step(
            agent_name="incident_agent",
            step="tool_call",
            session_id="sess-123",
            tool="logs_tool",
            args={"service": "payment-api"},
            result_preview="Found 3 OOM errors..."
        )
    """
    logger = get_logger("agent")
    logger.info(
        "agent_step",
        agent=agent_name,
        step=step,
        session_id=session_id,
        **details,
    )
