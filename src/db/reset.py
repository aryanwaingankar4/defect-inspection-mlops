"""Archive the local SQLite prediction log and start a fresh one (ADR-032).

    make db-reset          # DATABASE_URL from .env, archive folder from params (data.archive_dir)

Moves the database file and any ``-wal``/``-shm``/``-journal`` files to
``<archive_dir>/predictions-<UTC time>.db`` and creates an empty schema in its place, so labels
entered while testing the UI never reach a feedback export. SQLite only: a server database
(Postgres) is never touched. Stop ``make serve`` and ``make review-ui`` first.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.engine import make_url

from src.config import load_params
from src.db.store import PredictionLog, database_url_from_env

SIDE_SUFFIXES = ("-wal", "-shm", "-journal")


class ResetError(RuntimeError):
    pass


def sqlite_file(url: str) -> Path:
    """The database file of a SQLite URL; refuses any other backend or an in-memory database."""
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite":
        raise ResetError("db-reset only handles a local SQLite file, never a server database")
    if not parsed.database or parsed.database == ":memory:":
        raise ResetError("db-reset needs a SQLite file, not an in-memory database")
    return Path(parsed.database)


def archive_sqlite(url: str, archive_dir: Path | str, now: datetime | None = None) -> Path | None:
    """Move the SQLite file (and side files) into ``archive_dir``; ``None`` if there was none."""
    db_file = sqlite_file(url)
    if not db_file.exists():
        return None
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    target = Path(archive_dir) / f"{db_file.stem}-{stamp}{db_file.suffix}"
    if target.exists():
        raise ResetError(f"{target} already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        db_file.replace(target)
        for suffix in SIDE_SUFFIXES:
            side = db_file.with_name(db_file.name + suffix)
            if side.exists():
                side.replace(target.with_name(target.name + suffix))
    except PermissionError as exc:  # Windows: the file is open (API or UI still running)
        raise ResetError(f"{db_file} is in use: stop make serve / make review-ui first") from exc
    return target


def reset(url: str, archive_dir: Path | str) -> dict[str, str | None]:
    archived = archive_sqlite(url, archive_dir)
    db = PredictionLog(url)
    try:
        db.create_tables()
    finally:
        db.dispose()
    return {"archived_to": None if archived is None else archived.as_posix(), "fresh": True}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Archive the local SQLite log, start fresh")
    parser.add_argument("--config", default="configs/params.yaml")
    args = parser.parse_args(argv)
    params = load_params(args.config)
    try:
        out = reset(database_url_from_env(), params["data"]["archive_dir"])
    except ResetError as exc:
        raise SystemExit(f"db-reset: {exc}") from exc
    print("db-reset:", json.dumps(out))


if __name__ == "__main__":
    main()
