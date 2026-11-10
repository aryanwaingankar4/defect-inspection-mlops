"""Structured JSON logs for serving (TASK-018): one JSON object per line.

Fields go in ``extra={"fields": {...}}``. Callers pass only safe values: the request id, the
image SHA-256 and size, results and timings; never image bytes, file names or the API key.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

LOGGER_NAME = "src.serving"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        out.update(getattr(record, "fields", {}) or {})
        if record.exc_info and record.exc_info[0] is not None:
            out["exc_type"] = record.exc_info[0].__name__
        return json.dumps(out, default=str, ensure_ascii=False)


def configure(level: int = logging.INFO) -> logging.Logger:
    """Attach one JSON stream handler to the serving logger (idempotent)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger
