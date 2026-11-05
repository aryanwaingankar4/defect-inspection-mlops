"""Download the NEU surface defect images from Kaggle into a raw-data folder.

The Kaggle copy is the authors' NEU-DET release (.jpg, same 1,800 images as NEU-CLS).
See ADR-015 and docs/DATA_CARD.md. The dataset has no formal licence: never redistribute it.

Idempotent: if the folder already holds the expected images, the download is skipped and
only the count check runs.

Usage:
    python -m src.data.download --dataset kaustubhdikshit/neu-surface-defect-database/1 \
        --out data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

NEU_CLASSES: tuple[str, ...] = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)
IMAGES_PER_CLASS = 300
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".bmp", ".png"})
MANIFEST_NAME = "download_manifest.json"

Downloader = Callable[[str, Path], Path]


class DatasetCheckError(RuntimeError):
    """Raised when the raw folder does not hold the expected images."""


def count_images(root: Path) -> Counter[str]:
    """Count image files under ``root`` by class, where the class is the parent folder name."""
    counts: Counter[str] = Counter()
    if not root.is_dir():
        return counts
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            counts[path.parent.name] += 1
    return counts


def check_counts(
    counts: Counter[str],
    classes: tuple[str, ...] = NEU_CLASSES,
    per_class: int = IMAGES_PER_CLASS,
) -> None:
    """Raise ``DatasetCheckError`` unless every class has exactly ``per_class`` images."""
    expected = {name: per_class for name in classes}
    if dict(counts) != expected:
        raise DatasetCheckError(f"Expected {expected}, found {dict(sorted(counts.items()))}")


def is_complete(
    root: Path, classes: tuple[str, ...] = NEU_CLASSES, per_class: int = IMAGES_PER_CLASS
) -> bool:
    """Return True if ``root`` already holds the expected images."""
    try:
        check_counts(count_images(root), classes, per_class)
    except DatasetCheckError:
        return False
    return True


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(zip_path: Path, out_dir: Path) -> None:
    """Extract a zip, refusing any member that would land outside ``out_dir`` (zip slip)."""
    out_root = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (out_root / member).resolve()
            if not target.is_relative_to(out_root):
                raise DatasetCheckError(f"Unsafe path in zip: {member}")
        zf.extractall(out_root)


def kaggle_download(dataset: str, dest_dir: Path) -> Path:
    """Download a Kaggle dataset zip into ``dest_dir`` and return the zip path.

    Credentials come from ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi  # lazy: import needs credentials

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(dest_dir), force=True, quiet=False)
    slug = dataset.split("/")[1]
    return dest_dir / f"{slug}.zip"


def ensure_dataset(
    dataset: str,
    out_dir: Path,
    downloader: Downloader = kaggle_download,
    classes: tuple[str, ...] = NEU_CLASSES,
    per_class: int = IMAGES_PER_CLASS,
) -> Counter[str]:
    """Make sure ``out_dir`` holds the dataset; download only if it is missing or incomplete."""
    if is_complete(out_dir, classes, per_class):
        print(f"[download] {out_dir} already complete; skipping download.")
        return count_images(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_download"
    tmp_dir.mkdir(exist_ok=True)
    try:
        zip_path = downloader(dataset, tmp_dir)
        checksum = sha256_of(zip_path)
        safe_extract(zip_path, out_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    counts = count_images(out_dir)
    check_counts(counts, classes, per_class)
    manifest = {
        "dataset": dataset,
        "zip_sha256": checksum,
        "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "counts": dict(sorted(counts.items())),
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[download] zip SHA-256: {checksum}")
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="Kaggle ref: owner/slug[/version]")
    parser.add_argument("--out", required=True, type=Path, help="Raw data folder, e.g. data/raw")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        counts = ensure_dataset(args.dataset, args.out)
    except DatasetCheckError as err:
        print(f"[download] FAILED: {err}", file=sys.stderr)
        return 1
    total = sum(counts.values())
    print(f"[download] OK: {total} images, {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
