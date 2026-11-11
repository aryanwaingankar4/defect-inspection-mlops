"""TASK-019: the database layer (ADR-003, ADR-014, ADR-029), on SQLite in a temp folder."""

from datetime import UTC, datetime

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import LargeBinary  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402

from src.db import store  # noqa: E402
from src.db.models import Prediction  # noqa: E402

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
    log = store.PredictionLog("sqlite:///" + (tmp_path / "new" / "dir" / "p.db").as_posix())
    log.create_tables()
    yield log
    log.dispose()


def test_schema_is_exactly_the_adr_029_columns_and_holds_no_bytes():
    cols = Prediction.__table__.columns
    assert {c.key for c in cols} == EXPECTED_COLUMNS
    assert not [c.key for c in cols if isinstance(c.type, LargeBinary)]
    assert not [c.key for c in cols if any(w in c.key for w in ("pixel", "filename", "content"))]


def test_round_trip_and_folder_creation(db):
    db.add(row())
    assert sorted(db.table_names()) == ["feedback", "predictions"] and db.count() == 1
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


def test_create_tables_is_idempotent(db):
    db.add(row())
    db.create_tables()
    assert db.count() == 1


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


OLD_TABLE = """CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, request_id VARCHAR(32) NOT NULL UNIQUE,
    created_at DATETIME, image_sha256 VARCHAR(64), prediction VARCHAR(64),
    calibrated_confidence FLOAT, energy FLOAT, routed_to_review BOOLEAN,
    reason_low_confidence BOOLEAN, reason_ood BOOLEAN, model_name VARCHAR(128),
    model_version VARCHAR(32), model_precision VARCHAR(8), latency_ms FLOAT, inference_ms FLOAT,
    image_bytes INTEGER, image_width INTEGER, image_height INTEGER, image_format VARCHAR(16),
    brightness FLOAT, contrast FLOAT, sharpness FLOAT, logits JSON)"""


def test_an_older_log_gets_reason_audit_false(tmp_path):
    """A TASK-019 database: the new column is added and earlier rows were never audited."""
    from sqlalchemy import text

    url = "sqlite:///" + (tmp_path / "old.db").as_posix()
    old = store.PredictionLog(url)
    with old.engine.begin() as conn:
        conn.execute(text(OLD_TABLE))
        conn.execute(
            text(
                "INSERT INTO predictions (request_id, prediction, routed_to_review) "
                "VALUES ('" + "o" * 32 + "', 'crazing', 1)"
            )
        )
    old.dispose()
    log = store.PredictionLog(url)
    log.create_tables()
    got = log.recent(1)[0]
    assert got["request_id"] == "o" * 32 and got["reason_audit"] is False
    log.add(row("n" * 32, routed_to_review=True, reason_audit=True))
    assert log.recent(1)[0]["reason_audit"] is True
    assert log.upgrade_schema() == []  # nothing left to add
    log.dispose()


def test_unknown_or_undefaulted_schema_changes_stop(tmp_path):
    from sqlalchemy import text

    url = "sqlite:///" + (tmp_path / "odd.db").as_posix()
    odd = store.PredictionLog(url)
    with odd.engine.begin() as conn:
        conn.execute(text(OLD_TABLE.replace("logits JSON)", "logits JSON, extra TEXT)")))
    with pytest.raises(store.SchemaError, match="does not know"):
        odd.create_tables()
    odd.dispose()
    url2 = "sqlite:///" + (tmp_path / "missing.db").as_posix()
    missing = store.PredictionLog(url2)
    with missing.engine.begin() as conn:
        conn.execute(text(OLD_TABLE.replace(", logits JSON)", ")")))
    with pytest.raises(store.SchemaError, match="no default"):
        missing.create_tables()
    missing.dispose()
