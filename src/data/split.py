"""Deterministic, stratified split of the NEU images into train_v1 / val / test_frozen /
production_stream.

v1 (``data.split_file_v1``, kept for history): per class, sort paths, shuffle with the seed
from params.yaml, then cut the configured number of images for each split in config order.

v2 (``data.split_file``, ADR-021): group-aware. NEU crops often overlap, so near-copies of
one surface landed in different splits in v1. Confirmed near-copies (pixel correlation >=
``near_duplicates_cnn.confirm_pixel_corr`` in the pairs file from
``src.data.near_duplicates_cnn pairs``) are joined into groups, and each group stays in ONE
split. Per class, units (groups and single images) are shuffled with the seed; each unit goes
to the first split (config order) with room for it. Groups larger than
``split.groups.max_size_outside_last`` always go to the last split (production_stream), so a
chain of overlapping crops never fills a small split. Counts per class stay exact.

Both are written as committed CSVs (`path,class,split`) with paths relative to the raw-data
folder, in POSIX form.

Usage:
    python -m src.data.split --config configs/params.yaml
"""

from __future__ import annotations

import argparse
import csv
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.config import load_params
from src.data.download import IMAGE_SUFFIXES
from src.data.groups import confirmed, connected_groups, read_overlaps

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


def _units(paths: list[str], groups: Sequence[Sequence[str]]) -> list[list[str]]:
    """Groups (each sorted) and single images of one class, in a fixed order."""
    in_group = {p for g in groups for p in g}
    units = [sorted(g) for g in groups] + [[p] for p in paths if p not in in_group]
    return sorted(units, key=lambda u: u[0])


def make_group_split(
    images_by_class: Mapping[str, list[str]],
    per_class: Mapping[str, int],
    seed: int,
    groups: Sequence[Sequence[str]],
    max_size_outside_last: int,
) -> list[Row]:
    """Like ``make_split``, but every group of near-copies ends up in one split (ADR-021)."""
    if any(n <= 0 for n in per_class.values()):
        raise SplitError(f"Split sizes must be positive: {dict(per_class)}")
    total = sum(per_class.values())
    class_of = {p: name for name, paths in images_by_class.items() for p in paths}
    by_class: dict[str, list[list[str]]] = {}
    for g in groups:
        unknown = [p for p in g if p not in class_of]
        if unknown:
            raise SplitError(f"group has images that are not in the data: {unknown}")
        classes = sorted({class_of[p] for p in g})
        if len(classes) > 1:
            raise SplitError(f"group spans classes {classes}: {sorted(g)}")
        by_class.setdefault(classes[0], []).append(list(g))

    rng = random.Random(seed)
    split_names = list(per_class)
    last = split_names[-1]
    rows: list[Row] = []
    for name in sorted(images_by_class):
        paths = sorted(images_by_class[name])
        if len(paths) != total:
            raise SplitError(f"Class {name!r} has {len(paths)} images; split sizes sum to {total}")
        units = _units(paths, by_class.get(name, []))
        rng.shuffle(units)
        room = dict(per_class)
        for unit in units:
            if len(unit) > max_size_outside_last:
                target = last
            else:
                target = next((s for s in split_names if room[s] >= len(unit)), None)
            if target is None or room[target] < len(unit):
                raise SplitError(f"Class {name!r}: no split has room for a group of {len(unit)}")
            room[target] -= len(unit)
            rows.extend((p, name, target) for p in unit)
    order = {split_name: i for i, split_name in enumerate(per_class)}
    return sorted(rows, key=lambda r: (order[r[2]], r[1], r[0]))


def near_copy_groups(params: Mapping) -> list[list[str]]:
    """Groups of confirmed near-copies from the committed pairs file (ADR-021)."""
    c = params["near_duplicates_cnn"]
    sure = confirmed(read_overlaps(Path(c["pairs_file"])), float(c["confirm_pixel_corr"]))
    return connected_groups((o.path_a, o.path_b) for o in sure)


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
    s = params["split"]
    images = list_images(raw_dir)
    v1 = make_split(images, s["per_class"], int(s["seed"]))
    write_split_csv(v1, Path(params["data"]["split_file_v1"]))
    groups = near_copy_groups(params)
    v2 = make_group_split(
        images, s["per_class"], int(s["seed"]), groups, int(s["groups"]["max_size_outside_last"])
    )
    write_split_csv(v2, Path(params["data"]["split_file"]))
    moved = sum(a != b for a, b in zip(sorted(v1), sorted(v2), strict=True))
    print(f"[split] wrote {len(v1)} rows to {params['data']['split_file_v1']} (v1)")
    print(
        f"[split] wrote {len(v2)} rows to {params['data']['split_file']} (v2, group-aware: "
        f"{len(groups)} groups, {sum(map(len, groups))} images; {moved} images differ from v1)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
