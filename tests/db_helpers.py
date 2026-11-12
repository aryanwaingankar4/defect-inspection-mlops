"""Test databases (ADR-038): a temporary SQLite file, or the local TEST Postgres when
``TEST_DATABASE_URL`` is set (``make test-db-postgres``, CI's Postgres job)."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import make_url

from src.db import migrate
from src.db.store import make_engine

TEST_DATABASE_URL_ENV = "TEST_DATABASE_URL"
LOCAL_HOSTS = ("localhost", "127.0.0.1")


def sqlite_url(folder: Path, name: str = "p.db") -> str:
    return "sqlite:///" + (Path(folder) / name).as_posix()


def migrated(url: str) -> str:
    """``url`` upgraded to head (what `make db-migrate` does); returns the URL."""
    engine = make_engine(url)
    try:
        migrate.upgrade(engine)
    finally:
        engine.dispose()
    return url


def empty_test_database(tmp_path: Path) -> str:
    """A database with nothing in it. Postgres: the local TEST server only, schema dropped."""
    url = os.environ.get(TEST_DATABASE_URL_ENV, "").strip()
    if not url:
        return sqlite_url(tmp_path / "db")
    parsed = make_url(url)
    if parsed.get_backend_name() != "postgresql" or parsed.host not in LOCAL_HOSTS:
        raise RuntimeError(
            f"{TEST_DATABASE_URL_ENV} must be a local Postgres (tests drop its schema)"
        )
    engine = make_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
    finally:
        engine.dispose()
    return url
