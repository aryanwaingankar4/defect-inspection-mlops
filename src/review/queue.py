"""Review queue and label submission (TASK-024, ADR-031), through the one database layer."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.db.store import PredictionLog

UNKNOWN = "unknown"  # not one of the classes / not steel; never used for retraining (ADR-031)
REASONS = ("low_confidence", "ood", "audit")
KINDS = ("human", "simulated")


class ReviewError(ValueError):
    pass


def class_names(params: Mapping[str, Any]) -> list[str]:
    """The NEU classes in the model's order (sorted, as training uses them)."""
    with Path(params["data"]["split_file"]).open(encoding="utf-8", newline="") as f:
        return sorted({row["class"] for row in csv.DictReader(f)})


def labels(classes: list[str]) -> list[str]:
    return [*classes, UNKNOWN]


def reasons_of(row: Mapping[str, Any]) -> list[str]:
    flags = {
        "low_confidence": row["reason_low_confidence"],
        "ood": row["reason_ood"],
        "audit": row["reason_audit"],
    }
    return [r for r in REASONS if flags[r]]


def submit(
    db: PredictionLog,
    request_id: str,
    label: str,
    classes: list[str],
    *,
    reviewer_kind: str,
    reviewer_id: str,
    simulated_error_rate: float | None = None,
    simulated_flipped: bool | None = None,
) -> dict[str, Any]:
    """Store one final label for a routed prediction that has none yet."""
    if label not in labels(classes):
        raise ReviewError(f"label must be one of {labels(classes)}")
    if reviewer_kind not in KINDS:
        raise ReviewError(f"reviewer_kind must be one of {KINDS}")
    if not reviewer_id.strip():
        raise ReviewError("reviewer_id is required")
    pred = db.prediction(request_id)
    if pred is None:
        raise ReviewError(f"no prediction {request_id}")
    if not pred["routed_to_review"]:
        raise ReviewError(f"prediction {request_id} was not routed to review")
    if db.feedback_for(request_id) is not None:
        raise ReviewError(f"prediction {request_id} already has a label")
    row = {
        "request_id": request_id,
        "label": label,
        "reviewer_kind": reviewer_kind,
        "reviewer_id": reviewer_id.strip()[:64],
        "created_at": datetime.now(UTC),
        "simulated_error_rate": simulated_error_rate,
        "simulated_flipped": simulated_flipped,
    }
    db.add_feedback(row)
    return row
