import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from src.data import download as dl

CLASSES = ("crazing", "scratches")
PER_CLASS = 2
REPO_ROOT = Path(__file__).resolve().parents[1]


def make_images(root: Path, classes=CLASSES, per_class=PER_CLASS) -> None:
    for name in classes:
        folder = root / "NEU-DET" / "train" / "images" / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            (folder / f"{name}_{i}.jpg").write_bytes(b"x")


def test_count_images_groups_by_parent_folder(tmp_path: Path) -> None:
    make_images(tmp_path)
    (tmp_path / "notes.txt").write_text("not an image")
    assert dl.count_images(tmp_path) == Counter({"crazing": 2, "scratches": 2})


def test_check_counts_rejects_missing_images(tmp_path: Path) -> None:
    make_images(tmp_path)
    next((tmp_path / "NEU-DET").rglob("*.jpg")).unlink()
    with pytest.raises(dl.DatasetCheckError):
        dl.check_counts(dl.count_images(tmp_path), CLASSES, PER_CLASS)


def test_check_counts_rejects_unexpected_class(tmp_path: Path) -> None:
    make_images(tmp_path, classes=(*CLASSES, "extra"))
    with pytest.raises(dl.DatasetCheckError):
        dl.check_counts(dl.count_images(tmp_path), CLASSES, PER_CLASS)


def test_ensure_dataset_skips_download_when_complete(tmp_path: Path) -> None:
    make_images(tmp_path)

    def must_not_download(dataset: str, dest: Path) -> Path:
        raise AssertionError("download should be skipped")

    counts = dl.ensure_dataset("owner/slug", tmp_path, must_not_download, CLASSES, PER_CLASS)
    assert sum(counts.values()) == len(CLASSES) * PER_CLASS


def test_ensure_dataset_downloads_extracts_and_writes_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_images(source)
    out = tmp_path / "raw"

    def fake_download(dataset: str, dest: Path) -> Path:
        zip_path = dest / "slug.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in source.rglob("*.jpg"):
                zf.write(f, f.relative_to(source).as_posix())
        return zip_path

    dl.ensure_dataset("owner/slug/1", out, fake_download, CLASSES, PER_CLASS)
    assert dl.is_complete(out, CLASSES, PER_CLASS)
    assert (out / dl.MANIFEST_NAME).is_file()
    assert not (out / "_download").exists()


def test_safe_extract_rejects_zip_slip(tmp_path: Path) -> None:
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../escape.jpg", b"x")
    with pytest.raises(dl.DatasetCheckError):
        dl.safe_extract(zip_path, tmp_path / "out")
    assert not (tmp_path / "escape.jpg").exists()


def test_no_images_tracked_in_git() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    images = [p for p in tracked if Path(p).suffix.lower() in dl.IMAGE_SUFFIXES]
    assert images == [], f"Image files must not be committed (DVC tracks data): {images}"


@pytest.mark.needs_data
def test_real_raw_data_is_complete() -> None:
    dl.check_counts(dl.count_images(REPO_ROOT / "data" / "raw"))
