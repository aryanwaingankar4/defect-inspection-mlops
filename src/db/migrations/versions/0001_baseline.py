"""Baseline: the prediction log as of ADR-029/030/031 (predictions incl. reason_audit, feedback).

Written out by hand and frozen: it never imports ``src.db.models`` (ADR-038). Pre-Alembic local
logs with exactly this schema are stamped with this revision by ``src.db.migrate``.

Revision ID: 0001
Revises:
Create Date: 2026-11-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("request_id", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("image_sha256", sa.String(length=64), nullable=False),
        sa.Column("prediction", sa.String(length=64), nullable=False),
        sa.Column("calibrated_confidence", sa.Float(), nullable=False),
        sa.Column("energy", sa.Float(), nullable=False),
        sa.Column("routed_to_review", sa.Boolean(), nullable=False),
        sa.Column("reason_low_confidence", sa.Boolean(), nullable=False),
        sa.Column("reason_ood", sa.Boolean(), nullable=False),
        sa.Column("reason_audit", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("model_precision", sa.String(length=8), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("inference_ms", sa.Float(), nullable=False),
        sa.Column("image_bytes", sa.Integer(), nullable=False),
        sa.Column("image_width", sa.Integer(), nullable=False),
        sa.Column("image_height", sa.Integer(), nullable=False),
        sa.Column("image_format", sa.String(length=16), nullable=False),
        sa.Column("brightness", sa.Float(), nullable=False),
        sa.Column("contrast", sa.Float(), nullable=False),
        sa.Column("sharpness", sa.Float(), nullable=False),
        sa.Column("logits", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("request_id"),
    )
    for column in ("created_at", "image_sha256", "routed_to_review", "model_version"):
        op.create_index(f"ix_predictions_{column}", "predictions", [column], unique=False)

    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("request_id", sa.String(length=32), nullable=False),
        sa.Column("label", sa.String(length=64), nullable=False),
        sa.Column("reviewer_kind", sa.String(length=16), nullable=False),
        sa.Column("reviewer_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("simulated_error_rate", sa.Float(), nullable=True),
        sa.Column("simulated_flipped", sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(["request_id"], ["predictions.request_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("request_id"),
    )
    op.create_index("ix_feedback_created_at", "feedback", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_feedback_created_at", table_name="feedback")
    op.drop_table("feedback")
    for column in ("created_at", "image_sha256", "routed_to_review", "model_version"):
        op.drop_index(f"ix_predictions_{column}", table_name="predictions")
    op.drop_table("predictions")
