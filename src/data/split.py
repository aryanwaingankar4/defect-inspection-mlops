"""Deterministic, stratified split of the NEU images into train_v1 / val / test_frozen /
production_stream.

Per class: sort paths, shuffle with the seed from params.yaml, then cut the configured number
of images for each split in config order. The result is written as a committed CSV
(`path,class,split`) with paths relative to the raw-data folder, in POSIX form.

Usage:
    python -m src.data.split --config configs/params.yaml
"""

from __future__ import annotations

import argparse
import csv
import random
from collections.abc import Mapping
from pathlib import Path

from src.config import load_params
from src.data.download import IMAGE_SUFFIXES

CSV_FIELDS = ("path", "class", "split")
Row = tuple[str, str, str]


class SplitError(ValueError):
    """Raised when the split cannot be made as configured."""


def list_images(raw_dir: Path) -> dict[str, list[str]]:
    """Map class (parent folder name) to sorted POSIX paths relative to ``raw_dir``."""
    by_class: dict[str, list[str]] = {}
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            rel = path.relative_to(raw_dir).as_posix()
            by_class.setdefault(path.parent.name, []).append(rel)
    return {name: sorted(paths) for name, paths in sorted(by_class.items())}


def make_split(
    images_by_class: Mapping[str, list[str]], per_class: Mapping[str, int], seed: int
) -> list[Row]:
    """Return ``(path, class, split)`` rows: same counts per class, deterministic for a seed."""
    if any(n <= 0 for n in per_class.values()):
        raise SplitError(f"Split sizes must be positive: {dict(per_class)}")
    total = sum(per_class.values())
    rng = random.Random(seed)
    rows: list[Row] = []
    for name in sorted(images_by_class):
        paths = sorted(images_by_class[name])
        if len(paths) != total:
            raise SplitError(f"Class {name!r} has {len(paths)} images; split sizes sum to {total}")
        rng.shuffle(paths)
        start = 0
        for split_name, size in per_class.items():
            rows.extend((p, name, split_name) for p in paths[start : start + size])
            start += size
    order = {split_name: i for i, split_name in enumerate(per_class)}
    return sorted(rows, key=lambda r: (order[r[2]], r[1], r[0]))


def write_split_csv(rows: list[Row], path: Path) -> None:
    """Write rows as CSV with LF line endings, so the file is byte-identical on every OS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(CSV_FIELDS)
        writer.writerows(rows)


def read_split_csv(path: Path) -> list[Row]:
    """Read a split CSV back into ``(path, class, split)`` rows."""
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise SplitError(f"{path} must have columns {CSV_FIELDS}")
        return [(r["path"], r["class"], r["split"]) for r in reader]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Make the deterministic NEU split.")
    parser.add_argument("--config", required=True, type=Path, help="Path to params.yaml")
    args = parser.parse_args(argv)

    params = load_params(args.config)
    raw_dir = Path(params["data"]["raw_dir"])
    out = Path(params["data"]["split_file"])
    rows = make_split(
        list_images(raw_dir), params["split"]["per_class"], int(params["split"]["seed"])
    )
    write_split_csv(rows, out)
    print(f"[split] wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
