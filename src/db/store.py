"""Prediction log store (TASK-019, ADR-029). The engine comes from ``DATABASE_URL``.

The URL can hold a password (Postgres on Azure): it is never printed or logged; use
``safe_url`` for messages.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, func, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from src.db.models import Base, Prediction

DATABASE_URL_ENV = "DATABASE_URL"


class DatabaseConfigError(RuntimeError):
    pass


def database_url_from_env() -> str:
    url = os.environ.get(DATABASE_URL_ENV, "").strip()
    if not url:
        raise DatabaseConfigError(f"{DATABASE_URL_ENV} is not set")
    return url


def safe_url(url: str) -> str:
    """The URL with any password hidden."""
    return make_url(url).render_as_string(hide_password=True)


def make_engine(url: str) -> Engine:
    """Engine for ``url``; SQLite files get their folder created and cross-thread use allowed
    (the API runs requests in a thread pool)."""
    parsed = make_url(url)
    kwargs: dict[str, Any] = {}
    if parsed.get_backend_name() == "sqlite":
        kwargs["connect_args"] = {"check_same_thread": False}
        if parsed.database and parsed.database != ":memory:":
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, **kwargs)


class PredictionLog:
    """Write and read prediction rows."""

    def __init__(self, url: str) -> None:
        self.engine = make_engine(url)

    def create_tables(self) -> None:
        """Create missing tables (Alembic migrations come at TASK-037)."""
        Base.metadata.create_all(self.engine)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001 - any failure means "not reachable"
            return False

    def add(self, row: Mapping[str, Any]) -> None:
        with Session(self.engine) as session, session.begin():
            session.add(Prediction(**row))

    def count(self) -> int:
        with Session(self.engine) as session:
            return int(session.scalar(select(func.count()).select_from(Prediction)) or 0)

    def review_queue_size(self) -> int:
        """Predictions routed to review. Until feedback is stored (TASK-026) every routed
        prediction counts as waiting; then it becomes "routed and not yet labelled"."""
        with Session(self.engine) as session:
            query = select(func.count()).select_from(Prediction).where(Prediction.routed_to_review)
            return int(session.scalar(query) or 0)

    def recent(self, n: int = 10) -> list[dict[str, Any]]:
        """The ``n`` newest rows as dicts, newest first (``created_at`` as UTC)."""
        with Session(self.engine) as session:
            rows = session.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(n))
            return [as_dict(r) for r in rows]

    def table_names(self) -> list[str]:
        return inspect(self.engine).get_table_names()

    def dispose(self) -> None:
        self.engine.dispose()


def as_dict(row: Prediction) -> dict[str, Any]:
    out = {c.key: getattr(row, c.key) for c in Prediction.__table__.columns}
    created = out["created_at"]
    if created is not None and created.tzinfo is None:  # SQLite drops the zone; it is UTC
        out["created_at"] = created.replace(tzinfo=UTC)
    return out
