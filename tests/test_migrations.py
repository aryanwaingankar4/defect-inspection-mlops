"""TASK-037 (ADR-038): Alembic migrations, the head check, and adopting pre-Alembic SQLite logs.
Runs on a temporary SQLite file, or on the local TEST Postgres with ``TEST_DATABASE_URL``."""

import shutil
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("alembic")

from alembic.autogenerate import compare_metadata  # noqa: E402
from alembic.runtime.migration import MigrationContext  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import MetaData, Table, inspect, text  # noqa: E402

from src.db import migrate  # noqa: E402
from src.db.models import Base  # noqa: E402
from src.db.store import PredictionLog, SchemaError, make_engine  # noqa: E402
from tests.db_helpers import empty_test_database, migrated, sqlite_url  # noqa: E402
from tests.review_helpers import prediction  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def url(tmp_path):
    return empty_test_database(tmp_path)


def test_one_linear_history_starting_at_the_baseline():
    assert migrate.BASELINE == "0001" and migrate.head() == "0001"


def test_upgrade_creates_the_schema_at_head(url):
    engine = make_engine(url)
    try:
        assert migrate.current(engine) is None
        out = migrate.upgrade(engine)
        assert out == {"from": None, "adopted_pre_alembic": False, "to": migrate.head()}
        assert migrate.check_at_head(engine) == migrate.head()
        assert migrate.upgrade(engine)["from"] == migrate.head()  # again: a no-op
    finally:
        engine.dispose()


def test_head_matches_the_models_exactly(url):
    """A model change without a migration fails here (Alembic's autogenerate comparison)."""
    engine = make_engine(migrated(url))
    try:
        with engine.connect() as conn:
            diffs = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    finally:
        engine.dispose()
    assert diffs == []


def test_apps_refuse_a_database_that_is_not_at_head(url):
    engine = make_engine(url)
    try:
        with pytest.raises(SchemaError, match="none \\(never migrated\\).*make db-migrate"):
            migrate.check_at_head(engine)
        migrate.upgrade(engine)
        with engine.begin() as conn:  # a database from some other (e.g. newer) code
            conn.execute(text("UPDATE alembic_version SET version_num = 'ffff'"))
        with pytest.raises(SchemaError, match="revision ffff.*needs 0001"):
            PredictionLog(url).require_head()
    finally:
        engine.dispose()


def test_the_api_never_migrates_and_refuses_to_start(tmp_path, monkeypatch):
    pytest.importorskip("onnx")  # the test bundle is built with onnx (train group; main CI job)
    from src.config import load_params
    from src.serving import app as api
    from tests import serving_helpers as sh

    params = load_params(REPO_ROOT / "configs" / "params.yaml")
    url = sqlite_url(tmp_path / "db")
    app = api.create_app(
        params,
        api_key="k" * 16,
        bundle_dir=sh.make_bundle(tmp_path / "bundle", params["preprocess"]),
        database_url=url,
    )
    with pytest.raises(SchemaError, match="make db-migrate"), TestClient(app):
        pass
    engine = make_engine(url)
    try:
        assert migrate.current(engine) is None  # the API did not create or stamp anything
        assert "predictions" not in inspect(engine).get_table_names()
    finally:
        engine.dispose()


# ---------------------------------------------------------------- pre-Alembic SQLite logs


def legacy_log(path: Path, *, append_reason_audit: bool = False, extra: str = "") -> str:
    """A log as the old code made it, with one row and no alembic_version: ``create_all`` of
    today's models, or (``append_reason_audit``) the TASK-019 tables plus the ADR-030
    ``ALTER TABLE ... ADD COLUMN reason_audit`` (the shape of data/archive's log)."""
    url = sqlite_url(path.parent, path.name)
    engine = make_engine(url)
    if append_reason_audit:
        md = MetaData()
        old = Base.metadata.tables["predictions"]
        Table("predictions", md, *[c._copy() for c in old.columns if c.name != "reason_audit"])
        Base.metadata.tables["feedback"].to_metadata(md)
        md.create_all(engine)
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE predictions ADD COLUMN reason_audit BOOLEAN DEFAULT 0 NOT NULL")
            )
    else:
        Base.metadata.create_all(engine)
    if extra:
        with engine.begin() as conn:
            conn.execute(text(extra))
    engine.dispose()
    PredictionLog(url).add(prediction("a" * 32, reasons=("low_confidence",)))
    return url


@pytest.mark.parametrize("append", [False, True], ids=["create_all", "additive_alter"])
def test_a_pre_alembic_log_is_stamped_and_keeps_its_rows(tmp_path, append):
    url = legacy_log(tmp_path / "old.db", append_reason_audit=append)
    engine = make_engine(url)
    try:
        assert migrate.current(engine) is None
        with pytest.raises(SchemaError):
            PredictionLog(url).require_head()  # apps refuse it until db-migrate ran
        out = migrate.upgrade(engine)
        assert out["adopted_pre_alembic"] and out["to"] == migrate.head()
        log = PredictionLog(url)
        assert log.count() == 1 and log.require_head() == migrate.head()
        assert log.recent(1)[0]["reason_low_confidence"] is True
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "extra",
    [
        "ALTER TABLE predictions ADD COLUMN surprise TEXT",
        "CREATE INDEX ix_other ON predictions (prediction)",
        "DROP INDEX ix_feedback_created_at",
    ],
)
def test_a_pre_alembic_log_with_another_schema_is_refused_unchanged(tmp_path, extra):
    url = legacy_log(tmp_path / "odd.db", extra=extra)
    path = tmp_path / "odd.db"
    before = path.read_bytes()
    engine = make_engine(url)
    try:
        with pytest.raises(SchemaError, match="differs from the baseline"):
            migrate.upgrade(engine)
        assert migrate.current(engine) is None
    finally:
        engine.dispose()
    assert path.read_bytes() == before  # nothing changed


def test_the_schema_signature_ignores_column_order(tmp_path):
    a = make_engine(legacy_log(tmp_path / "a.db"))
    b = make_engine(legacy_log(tmp_path / "b.db", append_reason_audit=True))
    try:
        assert migrate.schema_signature(a) == migrate.schema_signature(b)
        assert migrate.schema_signature(a) == migrate.baseline_signature()
    finally:
        a.dispose()
        b.dispose()


@pytest.mark.needs_data
def test_the_real_local_logs_are_stamped_on_copies(tmp_path):
    """data/predictions.db and data/archive/*.db (git-ignored): copies only, never the files."""
    found = sorted((REPO_ROOT / "data").glob("predictions.db"))
    found += sorted((REPO_ROOT / "data" / "archive").glob("*.db"))
    if not found:
        pytest.skip("no local prediction logs")
    for i, src in enumerate(found):
        copy = tmp_path / f"copy{i}.db"
        shutil.copyfile(src, copy)
        with sqlite3.connect(copy) as conn:
            rows = conn.execute("SELECT count(*) FROM predictions").fetchone()[0]
        engine = make_engine(sqlite_url(tmp_path, copy.name))
        try:
            out = migrate.upgrade(engine)
            assert out["to"] == migrate.head(), src.name
            assert PredictionLog(sqlite_url(tmp_path, copy.name)).count() == rows
        finally:
            engine.dispose()


def test_a_server_database_without_a_revision_is_never_adopted(tmp_path, monkeypatch):
    engine = make_engine(legacy_log(tmp_path / "x.db"))
    monkeypatch.setattr(engine.dialect, "name", "postgresql")
    try:
        with pytest.raises(SchemaError, match="only pre-Alembic local SQLite"):
            migrate.adopt_legacy(engine)
    finally:
        engine.dispose()


def test_migrations_never_import_the_models():
    import ast

    for path in (migrate.MIGRATIONS_DIR / "versions").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any(m and m.startswith("src") for m in modules), path.name
