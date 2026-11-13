"""Alembic environment (ADR-038). Run migrations through ``src.db.migrate``, which passes the
engine in ``config.attributes``; there is no alembic.ini and no URL in any file."""

from alembic import context

from src.db.models import Base

engine = context.config.attributes.get("engine")
if engine is None:
    raise RuntimeError("run migrations with `python -m src.db.migrate` (ADR-038)")
if context.is_offline_mode():
    raise RuntimeError("offline (SQL script) migrations are not used in this project")

with engine.connect() as connection:
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,  # for --autogenerate only; revisions never import models
        render_as_batch=connection.dialect.name == "sqlite",  # SQLite ALTER limits
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()
