"""Near-copy pairs file and grouping (TASK-008b, ADR-021).

``src.data.near_duplicates_cnn pairs`` writes the pairs file (no pixels: paths, classes and
scores); ``src.data.split`` reads it and keeps every group of overlapping images in one split.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import astuple, dataclass
from pathlib import Path

OVERLAP_FIELDS = ("pixel_corr", "cosine", "path_a", "class_a", "path_b", "class_b")


@dataclass(frozen=True)
class Overlap:
    """Two images whose fine texture matches after some shift (scores rounded)."""

    pixel_corr: float
    cosine: float
    path_a: str
    class_a: str
    path_b: str
    class_b: str


def write_overlaps(overlaps: Sequence[Overlap], path: Path) -> None:
    """CSV with LF line endings, highest pixel correlation first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(OVERLAP_FIELDS)
        writer.writerows(astuple(o) for o in overlaps)


def read_overlaps(path: Path) -> list[Overlap]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != OVERLAP_FIELDS:
            raise ValueError(f"{path} must have columns {OVERLAP_FIELDS}")
        return [
            Overlap(
                float(r["pixel_corr"]),
                float(r["cosine"]),
                r["path_a"],
                r["class_a"],
                r["path_b"],
                r["class_b"],
            )
            for r in reader
        ]


def confirmed(overlaps: Iterable[Overlap], min_pixel_corr: float) -> list[Overlap]:
    return [o for o in overlaps if o.pixel_corr >= min_pixel_corr]


def connected_groups(edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Groups of items joined directly or through a chain of edges (union-find).

    Each group is sorted; groups are sorted by their first item. Single items are not returned.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[str]] = {}
    for x in parent:
        groups.setdefault(find(x), []).append(x)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
