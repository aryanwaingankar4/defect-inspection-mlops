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

from sqlalchemy import Engine, case, create_engine, func, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from src.db.models import Feedback, Prediction

DATABASE_URL_ENV = "DATABASE_URL"


class DatabaseConfigError(RuntimeError):
    pass


class SchemaError(RuntimeError):
    """The database is not at the schema revision this code needs (ADR-038)."""


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
    if parsed.drivername == "postgresql":  # the form Azure gives; psycopg 3 is installed
        parsed = parsed.set(drivername="postgresql+psycopg")
    kwargs: dict[str, Any] = {}
    if parsed.get_backend_name() == "sqlite":
        kwargs["connect_args"] = {"check_same_thread": False}
        if parsed.database and parsed.database != ":memory:":
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(parsed, **kwargs)


class PredictionLog:
    """Write and read prediction rows."""

    def __init__(self, url: str) -> None:
        self.engine = make_engine(url)

    def require_head(self) -> str:
        """Refuse unless the schema is at the Alembic head (ADR-038); never migrates."""
        from src.db.migrate import check_at_head

        return check_at_head(self.engine)

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

    def _queue_select(self, reason: str | None = None):
        """Routed predictions without a feedback row (ADR-031)."""
        query = (
            select(Prediction)
            .outerjoin(Feedback, Feedback.request_id == Prediction.request_id)
            .where(Prediction.routed_to_review, Feedback.id.is_(None))
        )
        if reason is not None:
            column = {
                "low_confidence": Prediction.reason_low_confidence,
                "ood": Prediction.reason_ood,
                "audit": Prediction.reason_audit,
            }[reason]
            query = query.where(column)
        return query

    def review_queue_size(self, reason: str | None = None) -> int:
        """Predictions routed to review and not yet labelled."""
        with Session(self.engine) as session:
            sub = self._queue_select(reason).subquery()
            return int(session.scalar(select(func.count()).select_from(sub)) or 0)

    def review_queue(self, limit: int | None = 50, reason: str | None = None) -> list[dict]:
        """The queue, oldest first."""
        with Session(self.engine) as session:
            query = self._queue_select(reason).order_by(Prediction.id)
            if limit is not None:
                query = query.limit(limit)
            return [as_dict(r) for r in session.scalars(query)]

    def prediction(self, request_id: str) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            row = session.scalar(select(Prediction).where(Prediction.request_id == request_id))
            return None if row is None else as_dict(row)

    def add_feedback(self, row: Mapping[str, Any]) -> None:
        with Session(self.engine) as session, session.begin():
            session.add(Feedback(**row))

    def feedback_for(self, request_id: str) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            row = session.scalar(select(Feedback).where(Feedback.request_id == request_id))
            return None if row is None else as_dict(row)

    def feedback_count(self) -> int:
        with Session(self.engine) as session:
            return int(session.scalar(select(func.count()).select_from(Feedback)) or 0)

    def labelled_images(self) -> list[dict[str, Any]]:
        """Every feedback row with its image hash, oldest first (feedback export, ADR-032).

        Selects only request id, label, reviewer kind and hash: the simulated-noise columns
        (``simulated_flipped``, ``simulated_error_rate``) are never read for training."""
        query = (
            select(
                Feedback.request_id,
                Feedback.label,
                Feedback.reviewer_kind,
                Prediction.image_sha256,
            )
            .join(Prediction, Prediction.request_id == Feedback.request_id)
            .order_by(Feedback.id)
        )
        with Session(self.engine) as session:
            return [dict(r._mapping) for r in session.execute(query)]

    def loop_counts(self) -> dict[str, Any]:
        """Counts for the loop report (ADR-036): predictions and routed per model version, and
        label rows per reviewer kind / simulated error rate / flipped. The flip count is for
        reporting only; the export never reads it (ADR-032)."""
        with Session(self.engine) as session:
            preds = session.execute(
                select(
                    Prediction.model_version,
                    func.count(),
                    func.sum(case((Prediction.routed_to_review, 1), else_=0)),
                ).group_by(Prediction.model_version)
            )
            labels = session.execute(
                select(
                    Feedback.reviewer_kind,
                    Feedback.simulated_error_rate,
                    func.count(),
                    func.sum(case((Feedback.simulated_flipped, 1), else_=0)),
                ).group_by(Feedback.reviewer_kind, Feedback.simulated_error_rate)
            )
            return {
                "predictions": {
                    str(v): {"n": int(n), "routed_to_review": int(r or 0)} for v, n, r in preds
                },
                "labels": [
                    {
                        "reviewer_kind": kind,
                        "simulated_error_rate": rate,
                        "n": int(n),
                        "simulated_flipped": int(f or 0),
                    }
                    for kind, rate, n, f in labels
                ],
            }

    def recent(self, n: int = 10) -> list[dict[str, Any]]:
        """The ``n`` newest rows as dicts, newest first (``created_at`` as UTC)."""
        with Session(self.engine) as session:
            rows = session.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(n))
            return [as_dict(r) for r in rows]

    def table_names(self) -> list[str]:
        return inspect(self.engine).get_table_names()

    def dispose(self) -> None:
        self.engine.dispose()


def as_dict(row: Prediction | Feedback) -> dict[str, Any]:
    out = {c.key: getattr(row, c.key) for c in row.__table__.columns}
    created = out["created_at"]
    if created is not None and created.tzinfo is None:  # SQLite drops the zone; it is UTC
        out["created_at"] = created.replace(tzinfo=UTC)
    return out
