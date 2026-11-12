"""Schema migrations (TASK-037, ADR-038). Alembic owns the prediction-log schema.

    make db-migrate                                  # upgrade DATABASE_URL to head
    python -m src.db.migrate current                 # print the database's revision
    python -m src.db.migrate revision -m "add x" --autogenerate   # new revision (dev only)

Only this module (``make db-migrate``) and ``make db-reset`` change a schema. The API, the review
UI, the simulated reviewer and the feedback export call ``check_at_head`` and refuse to run on
any other revision: they never migrate.

A pre-Alembic SQLite log (made by the old ``create_all`` + additive code) has the app tables but
no ``alembic_version``. ``upgrade`` stamps it with the baseline only if its schema equals a fresh
baseline database exactly; otherwise it refuses and changes nothing.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, inspect

from src.db.store import SchemaError, database_url_from_env, make_engine, safe_url

MIGRATIONS_DIR = Path(__file__).resolve().with_name("migrations")
BASELINE = "0001"
APP_TABLES = ("predictions", "feedback")


def config(engine: Engine | None = None) -> Config:
    """Alembic config built in code (no alembic.ini); ``env.py`` uses ``engine``."""
    cfg = Config()
    cfg.set_main_option("script_location", MIGRATIONS_DIR.as_posix())
    cfg.attributes["engine"] = engine
    return cfg


def head() -> str:
    found = ScriptDirectory.from_config(config()).get_current_head()
    if found is None:
        raise SchemaError("no migration revisions found")
    return found


def current(engine: Engine) -> str | None:
    """The database's revision (None: never migrated). Raises if unreachable."""
    with engine.connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def check_at_head(engine: Engine) -> str:
    """Refuse unless the database is at head (ADR-038). Returns the revision."""
    found, want = current(engine), head()
    if found != want:
        raise SchemaError(
            f"database {engine.url.render_as_string(hide_password=True)} is at "
            f"revision {found or 'none (never migrated)'}, this code needs {want}: run "
            "`make db-migrate` (apps never migrate by themselves, ADR-038)"
        )
    return found


def schema_signature(engine: Engine) -> dict[str, Any]:
    """Tables, columns (type, nullability), primary keys, indexes, unique constraints and
    foreign keys of the app tables, order-independent."""
    insp = inspect(engine)
    out: dict[str, Any] = {}
    for table in sorted(set(insp.get_table_names()) & set(APP_TABLES)):
        out[table] = {
            "columns": sorted(
                (c["name"], str(c["type"]), bool(c["nullable"])) for c in insp.get_columns(table)
            ),
            "primary_key": sorted(insp.get_pk_constraint(table)["constrained_columns"]),
            "indexes": sorted(
                (i["name"], tuple(i["column_names"]), bool(i["unique"]))
                for i in insp.get_indexes(table)
            ),
            "unique": sorted(tuple(u["column_names"]) for u in insp.get_unique_constraints(table)),
            "foreign_keys": sorted(
                (tuple(f["constrained_columns"]), f["referred_table"], tuple(f["referred_columns"]))
                for f in insp.get_foreign_keys(table)
            ),
        }
    return out


def baseline_signature() -> dict[str, Any]:
    """The schema of a fresh SQLite database at the baseline revision."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        ref = make_engine("sqlite:///" + (Path(tmp) / "baseline.db").as_posix())
        try:
            command.upgrade(config(ref), BASELINE)
            return schema_signature(ref)
        finally:
            ref.dispose()


def differences(got: dict[str, Any], want: dict[str, Any]) -> list[str]:
    diffs = []
    for table in sorted(set(got) | set(want)):
        if table not in got or table not in want:
            diffs.append(f"table {table}: {'missing' if table not in got else 'unexpected'}")
            continue
        for key in want[table]:
            if got[table][key] != want[table][key]:
                extra = sorted(set(got[table][key]) - set(want[table][key]))
                missing = sorted(set(want[table][key]) - set(got[table][key]))
                diffs.append(f"{table}.{key}: unexpected {extra}, missing {missing}")
    return diffs


def adopt_legacy(engine: Engine) -> None:
    """Stamp a pre-Alembic SQLite log with the baseline, only if its schema is the baseline."""
    if engine.dialect.name != "sqlite":
        raise SchemaError(
            f"{engine.dialect.name} database has app tables but no Alembic revision: refused "
            "(only pre-Alembic local SQLite logs are adopted, ADR-038)"
        )
    diffs = differences(schema_signature(engine), baseline_signature())
    if diffs:
        raise SchemaError(
            "pre-Alembic database differs from the baseline schema, nothing changed: "
            + "; ".join(diffs)
        )
    command.stamp(config(engine), BASELINE)


def upgrade(engine: Engine) -> dict[str, Any]:
    """Upgrade to head; a pre-Alembic log is first checked and stamped (ADR-038)."""
    before = current(engine)
    adopted = False
    if before is None and set(inspect(engine).get_table_names()) & set(APP_TABLES):
        adopt_legacy(engine)
        adopted = True
    command.upgrade(config(engine), "head")
    return {"from": before, "adopted_pre_alembic": adopted, "to": current(engine)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prediction-log migrations (ADR-038)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("upgrade", help="upgrade DATABASE_URL to head")
    sub.add_parser("current", help="print the revision of DATABASE_URL and the code's head")
    rev = sub.add_parser("revision", help="create a new revision file (development)")
    rev.add_argument("-m", "--message", required=True)
    rev.add_argument("--autogenerate", action="store_true")
    args = parser.parse_args(argv)
    engine = make_engine(database_url_from_env())
    where = safe_url(database_url_from_env())
    try:
        if args.cmd == "upgrade":
            out = upgrade(engine)
            note = " (pre-Alembic log checked and stamped)" if out["adopted_pre_alembic"] else ""
            print(f"db-migrate {where}: {out['from'] or 'none'} -> {out['to']}{note}")
        elif args.cmd == "current":
            print(f"{where}: revision {current(engine) or 'none'}; head {head()}")
        else:
            command.revision(config(engine), message=args.message, autogenerate=args.autogenerate)
    except SchemaError as exc:
        print(f"db-migrate: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
