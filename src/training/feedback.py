"""Feedback rows for retraining (TASK-028, ADR-034). No torch: also used by the retrain flow.

The feedback CSV comes from ``make feedback-export`` (ADR-032): ``path,label,reviewer_kind,
request_id`` with paths relative to ``data/raw``. Before training uses it, this module checks
that the file is exactly the expected export (MD5) and that every row is a ``production_stream``
image of the split file with a known label, once. Any violation stops training: retraining must
never see ``val`` or ``test_frozen`` images, whatever wrote the label. The reviewer that produced
the labels is SIMULATED in this project.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.data.split import read_split_csv

FEEDBACK_SPLIT = "production_stream"
REQUIRED_COLUMNS = ("path", "label")


class FeedbackError(ValueError):
    """The feedback file must not be used for training."""


@dataclass(frozen=True)
class Feedback:
    rows: list[tuple[str, str]]  # (path relative to raw_dir, class)
    md5: str

    def summary(self) -> dict[str, object]:
        return {
            "md5": self.md5,
            "rows": len(self.rows),
            "per_class": dict(sorted(Counter(c for _, c in self.rows).items())),
            "split": FEEDBACK_SPLIT,
        }


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_feedback(
    csv_path: Path, expected_md5: str, split_csv: Path, classes: Sequence[str]
) -> Feedback:
    """The checked feedback rows; raises ``FeedbackError`` on any problem."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FeedbackError(f"no feedback file at {csv_path}")
    md5 = file_md5(csv_path)
    if md5 != expected_md5:
        raise FeedbackError(f"feedback MD5 {md5} is not the expected {expected_md5}")
    split_of = {p: s for p, _, s in read_split_csv(Path(split_csv))}
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise FeedbackError(f"feedback file lacks columns {missing}")
        rows = [(r["path"], r["label"]) for r in reader]
    seen: set[str] = set()
    for path, label in rows:
        split = split_of.get(path)
        if split != FEEDBACK_SPLIT:
            raise FeedbackError(f"{path} is in split {split!r}, not {FEEDBACK_SPLIT} (ADR-034)")
        if label not in classes:
            raise FeedbackError(f"{path} has label {label!r}, not one of {list(classes)}")
        if path in seen:
            raise FeedbackError(f"{path} appears twice in the feedback file")
        seen.add(path)
    return Feedback(rows=rows, md5=md5)
