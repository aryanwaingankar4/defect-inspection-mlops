from collections import Counter
from pathlib import Path

import pytest

from src.config import load_params
from src.data.download import IMAGES_PER_CLASS, NEU_CLASSES
from src.data.split import (
    SplitError,
    list_images,
    make_split,
    read_split_csv,
    write_split_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
PER_CLASS = {"train_v1": 2, "val": 3, "test_frozen": 4, "production_stream": 5}
N = sum(PER_CLASS.values())


def fake_images(classes=("crazing", "scratches"), n=N) -> dict[str, list[str]]:
    return {c: [f"NEU-DET/images/{c}/{c}_{i}.jpg" for i in range(n)] for c in classes}


def assert_valid_split(rows, per_class, classes) -> None:
    paths = [r[0] for r in rows]
    assert len(paths) == len(set(paths)), "an image appears in more than one split"
    counts = Counter((r[1], r[2]) for r in rows)
    expected = {(c, s): n for c in classes for s, n in per_class.items()}
    assert dict(counts) == expected


def test_no_overlap_and_counts_per_class() -> None:
    images = fake_images()
    rows = make_split(images, PER_CLASS, seed=42)
    assert_valid_split(rows, PER_CLASS, images.keys())
    assert len(rows) == sum(len(v) for v in images.values())


def test_same_seed_same_split_regardless_of_input_order() -> None:
    images = fake_images()
    shuffled = {c: list(reversed(p)) for c, p in reversed(list(images.items()))}
    assert make_split(images, PER_CLASS, 42) == make_split(shuffled, PER_CLASS, 42)


def test_different_seed_different_split() -> None:
    images = fake_images()
    assert make_split(images, PER_CLASS, 42) != make_split(images, PER_CLASS, 43)


def test_sizes_must_match_class_count() -> None:
    with pytest.raises(SplitError):
        make_split(fake_images(n=N + 1), PER_CLASS, 42)


def test_sizes_must_be_positive() -> None:
    with pytest.raises(SplitError):
        make_split(fake_images(), {**PER_CLASS, "val": 0}, 42)


def test_list_images_relative_posix_sorted(tmp_path: Path) -> None:
    for name in ("b.jpg", "a.jpg", "notes.txt"):
        folder = tmp_path / "NEU-DET" / "train" / "images" / "crazing"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"x")
    assert list_images(tmp_path) == {
        "crazing": ["NEU-DET/train/images/crazing/a.jpg", "NEU-DET/train/images/crazing/b.jpg"]
    }


def test_csv_round_trip(tmp_path: Path) -> None:
    rows = make_split(fake_images(), PER_CLASS, 42)
    out = tmp_path / "split.csv"
    write_split_csv(rows, out)
    assert read_split_csv(out) == rows
    assert b"\r\n" not in out.read_bytes()


def test_committed_split_file_is_valid() -> None:
    per_class = PARAMS["split"]["per_class"]
    assert sum(per_class.values()) == IMAGES_PER_CLASS
    rows = read_split_csv(REPO_ROOT / PARAMS["data"]["split_file"])
    assert len(rows) == len(NEU_CLASSES) * IMAGES_PER_CLASS
    assert_valid_split(rows, per_class, NEU_CLASSES)


@pytest.mark.needs_data
def test_committed_split_reproduces_from_raw_data(tmp_path: Path) -> None:
    raw_dir = REPO_ROOT / PARAMS["data"]["raw_dir"]
    committed = REPO_ROOT / PARAMS["data"]["split_file"]
    rows = make_split(
        list_images(raw_dir), PARAMS["split"]["per_class"], int(PARAMS["split"]["seed"])
    )
    regenerated = tmp_path / "split.csv"
    write_split_csv(rows, regenerated)
    assert regenerated.read_bytes() == committed.read_bytes()
    assert all((raw_dir / r[0]).is_file() for r in rows)
