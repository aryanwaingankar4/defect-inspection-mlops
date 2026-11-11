"""The operator's image archive, looked up by SHA-256 (ADR-031).

The prediction log stores only the image hash (ADR-029). The review side finds the image in the
local archive (`data/raw/`, indexed from the split file) by the hash of its file bytes, which is
exactly what the API hashed. Nothing is copied or stored. A hash shared by files of different
classes has no unique ground truth and is treated as unknown.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

TEST_SPLIT = "test_frozen"
PRODUCTION_SPLIT = "production_stream"  # the only split the loop may label or export (ADR-032)


@dataclass(frozen=True)
class ArchiveEntry:
    path: Path
    label: str
    split: str


class ImageArchive:
    def __init__(self, entries: dict[str, ArchiveEntry], ambiguous: set[str]) -> None:
        self.entries = entries
        self.ambiguous = ambiguous

    @classmethod
    def from_split_file(cls, split_file: Path | str, raw_dir: Path | str) -> ImageArchive:
        """Index every file of the split CSV that exists under ``raw_dir`` (missing files are
        skipped, so the archive is empty where the data is not present, e.g. in CI)."""
        raw = Path(raw_dir)
        found: dict[str, list[ArchiveEntry]] = {}
        with Path(split_file).open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                path = raw / row["path"]
                if not path.is_file():
                    continue
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                found.setdefault(sha, []).append(ArchiveEntry(path, row["class"], row["split"]))
        entries, ambiguous = {}, set()
        for sha, items in found.items():
            if len({e.label for e in items}) > 1:
                ambiguous.add(sha)
            else:
                # byte-identical copies of one class: keep the first; any split membership counts
                splits = {e.split for e in items}
                first = items[0]
                split = TEST_SPLIT if TEST_SPLIT in splits else first.split
                entries[sha] = ArchiveEntry(first.path, first.label, split)
        return cls(entries, ambiguous)

    def lookup(self, sha256: str) -> ArchiveEntry | None:
        return self.entries.get(sha256)

    def in_split(self, sha256: str, split: str) -> bool:
        entry = self.entries.get(sha256)
        return entry is not None and entry.split == split

    def __len__(self) -> int:
        return len(self.entries)
