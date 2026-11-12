"""Simulated reviewer (TASK-025, ADR-031). THE REVIEWER IS SIMULATED: there are no real inspectors.

For each queued prediction (oldest first, up to a limit) it takes the true NEU label of the archive
file with that hash and, with probability ``error_rate``, replaces it with a class drawn uniformly
from the other classes. The draw is seeded by ``seed`` and the request id, so a rerun gives the
same labels in any order. It labels only images whose hash is in ``production_stream`` (ADR-032):
``test_frozen`` images (that would leak test images into retraining) and images of any other split
are skipped and counted, as are images without a unique ground truth.

    make simulate-review                 # params.yaml -> review.simulated
    make simulate-review ARGS="--limit 20 --error-rate 0.05"
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import load_params
from src.db.store import PredictionLog, database_url_from_env
from src.review.archive import PRODUCTION_SPLIT, TEST_SPLIT, ImageArchive
from src.review.queue import class_names, submit


@dataclass(frozen=True)
class SimConfig:
    error_rate: float
    seed: int
    reviewer_id: str
    limit: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.error_rate <= 1.0:
            raise ValueError(f"error_rate must be in [0, 1], got {self.error_rate}")
        if self.limit < 1:
            raise ValueError("limit must be at least 1")

    @classmethod
    def from_params(cls, params: Mapping[str, Any], **overrides: Any) -> SimConfig:
        s = params["review"]["simulated"]
        values = {
            "error_rate": float(s["error_rate"]),
            "seed": int(s["seed"]),
            "reviewer_id": str(s["reviewer_id"]),
            "limit": int(s["limit"]),
        }
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


def simulated_label(
    true_label: str, classes: list[str], error_rate: float, seed: int, request_id: str
) -> tuple[str, bool]:
    """The simulated reviewer's answer and whether it is wrong on purpose."""
    rng = random.Random(f"{seed}:{request_id}")  # str seeds are hashed deterministically
    if rng.random() < error_rate:
        return rng.choice([c for c in classes if c != true_label]), True
    return true_label, False


def run(
    db: PredictionLog, archive: ImageArchive, classes: list[str], cfg: SimConfig
) -> dict[str, Any]:
    """Label up to ``cfg.limit`` queued predictions; returns counts (no labels, no pixels)."""
    counts: Counter[str] = Counter()
    queue_before = db.review_queue_size()
    for item in db.review_queue(limit=None):
        if counts["labelled"] >= cfg.limit:
            break
        sha = item["image_sha256"]
        entry = archive.lookup(sha)
        if archive.in_split(sha, TEST_SPLIT):
            counts["skipped_test_frozen"] += 1
            continue
        if entry is None or entry.label not in classes:
            counts["skipped_no_ground_truth"] += 1
            continue
        if entry.split != PRODUCTION_SPLIT:
            counts["skipped_not_production_stream"] += 1
            continue
        label, flipped = simulated_label(
            entry.label, classes, cfg.error_rate, cfg.seed, item["request_id"]
        )
        submit(
            db,
            item["request_id"],
            label,
            classes,
            reviewer_kind="simulated",
            reviewer_id=cfg.reviewer_id,
            simulated_error_rate=cfg.error_rate,
            simulated_flipped=flipped,
        )
        counts["labelled"] += 1
        counts["flipped"] += flipped
    return {
        "reviewer": "simulated (ADR-031)",
        "error_rate": cfg.error_rate,
        "seed": cfg.seed,
        "queue_before": queue_before,
        "labelled": counts["labelled"],
        "flipped": counts["flipped"],
        "skipped_test_frozen": counts["skipped_test_frozen"],
        "skipped_no_ground_truth": counts["skipped_no_ground_truth"],
        "skipped_not_production_stream": counts["skipped_not_production_stream"],
        "queue_left": db.review_queue_size(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Simulated reviewer (ADR-031)")
    parser.add_argument("--config", default="configs/params.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--error-rate", type=float)
    parser.add_argument("--out", type=Path, help="also write the summary to this JSON file")
    args = parser.parse_args(argv)
    params = load_params(args.config)
    cfg = SimConfig.from_params(params, limit=args.limit, error_rate=args.error_rate)
    archive = ImageArchive.from_split_file(params["data"]["split_file"], params["data"]["raw_dir"])
    db = PredictionLog(database_url_from_env())
    db.create_tables()
    try:
        summary = run(db, archive, class_names(params), cfg)
    finally:
        db.dispose()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes((json.dumps(summary, indent=2) + "\n").encode())
    print("SIMULATED reviewer (no real inspector):", json.dumps(summary))


if __name__ == "__main__":
    main()
