"""Feedback export (TASK-026, ADR-032): production_stream labels -> a path/label CSV for retraining.

THE STREAM AND THE AUTOMATED REVIEWER ARE SIMULATED; human labels from the UI are real labels on
simulated traffic. The export is the last guard before training: it re-checks every label against
the archive and keeps only images whose hash is in ``production_stream``. Excluded and counted:
``test_frozen``, ``val`` and other splits, hashes not in the archive (or without a unique class),
label ``unknown``, and images with conflicting labels. ``simulated_flipped`` is never read.

No image bytes are copied: the CSV holds the path relative to ``data/raw``. ``make feedback-export``
then runs ``dvc add data/feedback``, so each export is one DVC version.

    make feedback-export
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import load_params
from src.db.store import PredictionLog, SchemaError, database_url_from_env
from src.review.archive import PRODUCTION_SPLIT, TEST_SPLIT, ImageArchive
from src.review.queue import KINDS, UNKNOWN, class_names

CSV_NAME = "feedback.csv"
COLUMNS = ("path", "label", "reviewer_kind", "request_id")
EXCLUSIONS = (
    "test_frozen",
    "val",
    "other_split",
    "not_in_archive",
    "unknown_label",
    "invalid_label",
    "conflicting_labels",
    "repeated_image",
)


class ExportError(RuntimeError):
    pass


def exclusion(entry: Any, label: str, classes: list[str]) -> str | None:
    """Why a label row is not exported, or ``None`` if it may be (before the per-image check)."""
    if entry is None:
        return "not_in_archive"
    if entry.split == TEST_SPLIT:
        return "test_frozen"
    if entry.split == "val":
        return "val"
    if entry.split != PRODUCTION_SPLIT:
        return "other_split"
    if label == UNKNOWN:
        return "unknown_label"
    if label not in classes:
        return "invalid_label"
    return None


def select_rows(
    labelled: list[Mapping[str, Any]],
    archive: ImageArchive,
    raw_dir: Path | str,
    classes: list[str],
) -> tuple[list[dict[str, str]], Counter[str]]:
    """The rows to export (one per image, sorted by path) and the exclusion counts.

    ``labelled`` is oldest first; an image labelled more than once is kept once (latest request)
    if all its labels agree, and dropped as ``conflicting_labels`` if they do not."""
    excluded: Counter[str] = Counter({k: 0 for k in EXCLUSIONS})
    by_image: dict[str, list[Mapping[str, Any]]] = {}
    for row in labelled:
        entry = archive.lookup(row["image_sha256"])
        reason = exclusion(entry, row["label"], classes)
        if reason is not None:
            excluded[reason] += 1
            continue
        by_image.setdefault(row["image_sha256"], []).append(row)
    rows = []
    for sha, items in by_image.items():
        if len({r["label"] for r in items}) > 1:
            excluded["conflicting_labels"] += len(items)
            continue
        excluded["repeated_image"] += len(items) - 1
        latest = items[-1]
        rows.append(
            {
                "path": archive.lookup(sha).path.relative_to(Path(raw_dir)).as_posix(),
                "label": latest["label"],
                "reviewer_kind": latest["reviewer_kind"],
                "request_id": latest["request_id"],
            }
        )
    rows.sort(key=lambda r: r["path"])
    return rows, excluded


def write_csv(rows: list[dict[str, str]], path: Path) -> str:
    """Write the CSV (LF line ends) and return its MD5, the hash DVC records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_report(
    rows: list[dict[str, str]],
    excluded: Counter[str],
    read: int,
    classes: list[str],
    csv_path: str,
    csv_md5: str,
    split_file: str,
) -> dict[str, Any]:
    if len(rows) + sum(excluded.values()) != read:
        raise ExportError("export counts do not add up to the label rows read")
    per_class = Counter(r["label"] for r in rows)
    per_kind = Counter(r["reviewer_kind"] for r in rows)
    return {
        "note": "SIMULATED stream and reviewer (ADR-031/032); human labels are real labels on "
        "simulated traffic",
        "split": PRODUCTION_SPLIT,
        "split_file": split_file,
        "label_rows_read": read,
        "rows": len(rows),
        "per_class": {c: per_class[c] for c in classes},
        "per_reviewer_kind": {k: per_kind[k] for k in KINDS},
        "excluded": {k: excluded[k] for k in EXCLUSIONS},
        "csv": csv_path,
        "csv_md5": csv_md5,
        "exported_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def export(
    db: PredictionLog,
    archive: ImageArchive,
    classes: list[str],
    raw_dir: Path | str,
    out_dir: Path | str,
    report_path: Path | str,
    split_file: str,
) -> dict[str, Any]:
    """Write ``<out_dir>/feedback.csv`` and the JSON report; returns the report."""
    if len(archive) == 0:
        raise ExportError(f"no archive images under {raw_dir}: run `make pull` first")
    try:
        db.require_head()  # never migrates (ADR-038)
    except SchemaError as exc:
        raise ExportError(f"{exc} (wrong DATABASE_URL?)") from exc
    labelled = db.labelled_images()
    rows, excluded = select_rows(labelled, archive, raw_dir, classes)
    csv_path = Path(out_dir) / CSV_NAME
    md5 = write_csv(rows, csv_path)
    report = build_report(
        rows, excluded, len(labelled), classes, csv_path.as_posix(), md5, split_file
    )
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export production_stream feedback (ADR-032)")
    parser.add_argument("--config", default="configs/params.yaml")
    parser.add_argument("--out-dir", help="override review.export.out_dir")
    parser.add_argument("--report", help="override review.export.report")
    args = parser.parse_args(argv)
    params = load_params(args.config)
    cfg = params["review"]["export"]
    raw_dir, split_file = params["data"]["raw_dir"], params["data"]["split_file"]
    archive = ImageArchive.from_split_file(split_file, raw_dir)
    db = PredictionLog(database_url_from_env())
    try:
        report = export(
            db,
            archive,
            class_names(params),
            raw_dir,
            args.out_dir or cfg["out_dir"],
            args.report or cfg["report"],
            split_file,
        )
    except ExportError as exc:
        raise SystemExit(f"feedback-export: {exc}") from exc
    finally:
        db.dispose()
    print("feedback-export (SIMULATED loop):", json.dumps(report))


if __name__ == "__main__":
    main()
