"""Table definitions (TASK-019, ADR-029). Never a column for image bytes, pixels or file names."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """One row per successful ``/predict`` (ADR-029)."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)  # UTC
    image_sha256: Mapped[str] = mapped_column(String(64), index=True)

    prediction: Mapped[str] = mapped_column(String(64))
    calibrated_confidence: Mapped[float] = mapped_column(Float)
    energy: Mapped[float] = mapped_column(Float)
    routed_to_review: Mapped[bool] = mapped_column(Boolean, index=True)
    reason_low_confidence: Mapped[bool] = mapped_column(Boolean)
    reason_ood: Mapped[bool] = mapped_column(Boolean)

    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(32), index=True)
    model_precision: Mapped[str] = mapped_column(String(8))

    latency_ms: Mapped[float] = mapped_column(Float)
    inference_ms: Mapped[float] = mapped_column(Float)

    image_bytes: Mapped[int] = mapped_column(Integer)  # the size, not the content
    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    image_format: Mapped[str] = mapped_column(String(16))

    brightness: Mapped[float] = mapped_column(Float)
    contrast: Mapped[float] = mapped_column(Float)
    sharpness: Mapped[float] = mapped_column(Float)
    logits: Mapped[list[float]] = mapped_column(JSON)  # raw logits, class order of the model
