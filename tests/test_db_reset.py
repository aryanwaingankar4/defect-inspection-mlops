"""ADR-032: `make db-reset` archives the local SQLite log and starts an empty one."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

from src.db import migrate  # noqa: E402
from src.db import reset as rs  # noqa: E402
from src.db.store import PredictionLog  # noqa: E402
from tests import review_helpers as rh  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 11, 9, 30, 0, tzinfo=UTC)


def url_for(path: Path) -> str:
    return "sqlite:///" + path.as_posix()


def test_reset_archives_the_log_and_side_files_and_starts_empty(tmp_path):
    db_file = tmp_path / "data" / "predictions.db"
    log = PredictionLog(url_for(db_file))
    migrate.upgrade(log.engine)
    log.add(rh.prediction("a" * 32, reasons=("audit",)))
    log.dispose()
    (tmp_path / "data" / "predictions.db-wal").write_bytes(b"wal")
    archive = tmp_path / "data" / "archive"

    target = rs.archive_sqlite(url_for(db_file), archive, now=NOW)
    assert target == archive / "predictions-20261111T093000Z.db"
    assert target.is_file() and not db_file.exists()
    assert (archive / "predictions-20261111T093000Z.db-wal").read_bytes() == b"wal"
    old = PredictionLog(url_for(target))
    assert old.count() == 1  # the old rows are kept in the archive
    old.dispose()

    out = rs.reset(url_for(db_file), archive)
    assert out["archived_to"] is None  # nothing left to archive
    fresh = PredictionLog(url_for(db_file))
    assert {"predictions", "feedback"} <= set(fresh.table_names())
    assert fresh.count() == 0 and fresh.feedback_count() == 0
    fresh.dispose()


def test_reset_never_overwrites_an_archive(tmp_path):
    db_file = tmp_path / "p.db"
    db_file.write_bytes(b"x")
    (tmp_path / "arc").mkdir()
    (tmp_path / "arc" / "p-20261111T093000Z.db").write_bytes(b"old")
    with pytest.raises(rs.ResetError, match="already exists"):
        rs.archive_sqlite(url_for(db_file), tmp_path / "arc", now=NOW)
    assert db_file.exists()


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg://user:secret@db.example.com/defects",
        "sqlite://",
        "sqlite:///:memory:",
    ],
)
def test_reset_refuses_anything_but_a_sqlite_file(url):
    with pytest.raises(rs.ResetError):
        rs.sqlite_file(url)


def test_archive_folder_is_git_ignored():
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/data/archive/" in ignore
