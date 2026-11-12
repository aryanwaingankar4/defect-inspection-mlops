"""TASK-019: the database layer (ADR-003, ADR-014, ADR-029). A temporary SQLite file, or the local
TEST Postgres with ``TEST_DATABASE_URL`` (ADR-038). Schema changes: tests/test_migrations.py."""

from datetime import UTC, datetime

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import LargeBinary  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402

from src.db import store  # noqa: E402
from src.db.models import Prediction  # noqa: E402
from tests.db_helpers import empty_test_database, migrated  # noqa: E402

EXPECTED_COLUMNS = {
    "id",
    "request_id",
    "created_at",
    "image_sha256",
    "prediction",
    "calibrated_confidence",
    "energy",
    "routed_to_review",
    "reason_low_confidence",
    "reason_ood",
    "reason_audit",
    "model_name",
    "model_version",
    "model_precision",
    "latency_ms",
    "inference_ms",
    "image_bytes",
    "image_width",
    "image_height",
    "image_format",
    "brightness",
    "contrast",
    "sharpness",
    "logits",
}


def row(request_id: str = "a" * 32, **changes) -> dict:
    return {
        "request_id": request_id,
        "created_at": datetime(2026, 11, 9, 12, 0, tzinfo=UTC),
        "image_sha256": "f" * 64,
        "prediction": "scratches",
        "calibrated_confidence": 0.97,
        "energy": -11.5,
        "routed_to_review": False,
        "reason_low_confidence": False,
        "reason_ood": False,
        "reason_audit": False,
        "model_name": "steel-defect-classifier",
        "model_version": "2",
        "model_precision": "fp32",
        "latency_ms": 41.0,
        "inference_ms": 30.0,
        "image_bytes": 12345,
        "image_width": 200,
        "image_height": 200,
        "image_format": "JPEG",
        "brightness": 120.5,
        "contrast": 22.1,
        "sharpness": 310.0,
        "logits": [1.0, -2.0, 0.5, 3.0, -1.0, 9.0],
        **changes,
    }


@pytest.fixture
def db(tmp_path):
    log = store.PredictionLog(migrated(empty_test_database(tmp_path / "new" / "dir")))
    log.require_head()
    yield log
    log.dispose()


def test_schema_is_exactly_the_adr_029_columns_and_holds_no_bytes():
    cols = Prediction.__table__.columns
    assert {c.key for c in cols} == EXPECTED_COLUMNS
    assert not [c.key for c in cols if isinstance(c.type, LargeBinary)]
    assert not [c.key for c in cols if any(w in c.key for w in ("pixel", "filename", "content"))]


def test_round_trip_and_folder_creation(db):
    db.add(row())
    assert sorted(db.table_names()) == ["alembic_version", "feedback", "predictions"]
    assert db.count() == 1
    got = db.recent(1)[0]
    expected = row()
    for key, value in expected.items():
        assert got[key] == value, key
    assert got["created_at"].tzinfo is not None


def test_request_id_is_unique(db):
    db.add(row())
    with pytest.raises(IntegrityError):
        db.add(row())
    assert db.count() == 1  # the failed insert was rolled back


def test_recent_is_newest_first(db):
    for i in range(3):
        db.add(row(f"{i:032d}"))
    assert [r["request_id"] for r in db.recent(2)] == [f"{2:032d}", f"{1:032d}"]


def test_migrating_again_changes_nothing(db):
    db.add(row())
    migrated(db.engine.url.render_as_string(hide_password=False))
    assert db.count() == 1 and db.require_head() == "0001"


def test_ping(db, tmp_path):
    assert db.ping()
    bad = store.PredictionLog("sqlite:///" + tmp_path.as_posix())  # a folder: cannot open
    assert not bad.ping()


def test_database_url_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(store.DatabaseConfigError):
        store.database_url_from_env()
    monkeypatch.setenv("DATABASE_URL", "  ")
    with pytest.raises(store.DatabaseConfigError):
        store.database_url_from_env()
    monkeypatch.setenv("DATABASE_URL", "sqlite:///x.db")
    assert store.database_url_from_env() == "sqlite:///x.db"


def test_safe_url_hides_the_password():
    shown = store.safe_url("postgresql://user:s3cret@host:5432/db")
    assert "s3cret" not in shown and "user" in shown and "host" in shown


def test_review_queue_size_counts_routed_rows(db):
    db.add(row("a" * 32))
    db.add(row("b" * 32, routed_to_review=True, reason_low_confidence=True))
    db.add(row("c" * 32, routed_to_review=True, reason_ood=True))
    assert db.review_queue_size() == 2


def test_postgresql_urls_use_psycopg():
    engine = store.make_engine("postgresql://u:p@db.example:5432/x")
    assert engine.url.drivername == "postgresql+psycopg"
    engine.dispose()
