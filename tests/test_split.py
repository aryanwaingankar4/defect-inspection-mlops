from collections import Counter
from pathlib import Path

import pytest

from src.config import load_params
from src.data.download import IMAGES_PER_CLASS, NEU_CLASSES
from src.data.groups import confirmed, read_overlaps
from src.data.split import (
    SplitError,
    list_images,
    make_group_split,
    make_split,
    near_copy_groups,
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


# --- group-aware split (v2, ADR-021)

MAX_OUTSIDE = 3


def test_group_split_without_groups_equals_plain_split() -> None:
    images = fake_images()
    assert make_group_split(images, PER_CLASS, 42, [], MAX_OUTSIDE) == make_split(
        images, PER_CLASS, 42
    )


@pytest.mark.parametrize("seed", range(10))
def test_groups_stay_in_one_split_with_exact_counts(seed: int) -> None:
    images = fake_images()
    c, s = images["crazing"], images["scratches"]
    groups = [[c[0], c[1]], [c[2], c[3], c[4]], [s[5], s[6]]]
    rows = make_group_split(images, PER_CLASS, seed, groups, MAX_OUTSIDE)
    assert_valid_split(rows, PER_CLASS, images.keys())
    split_of = {r[0]: r[2] for r in rows}
    for g in groups:
        assert len({split_of[p] for p in g}) == 1, g


def test_big_groups_go_to_the_last_split() -> None:
    images = fake_images()
    big = images["crazing"][:4]  # larger than MAX_OUTSIDE
    for seed in range(10):
        rows = make_group_split(images, PER_CLASS, seed, [big], MAX_OUTSIDE)
        assert {r[2] for r in rows if r[0] in big} == {"production_stream"}


def test_group_split_is_deterministic() -> None:
    images = fake_images()
    groups = [images["crazing"][:2]]
    assert make_group_split(images, PER_CLASS, 7, groups, 3) == make_group_split(
        images, PER_CLASS, 7, [list(reversed(groups[0]))], 3
    )


def test_group_across_classes_is_refused() -> None:
    images = fake_images()
    with pytest.raises(SplitError, match="spans classes"):
        make_group_split(images, PER_CLASS, 42, [[images["crazing"][0], images["scratches"][0]]], 3)


def test_group_too_big_for_any_split_is_refused() -> None:
    images = fake_images()
    with pytest.raises(SplitError, match="no split has room"):
        make_group_split(images, PER_CLASS, 42, [images["crazing"][:6]], 3)


@pytest.mark.parametrize("key", ["split_file", "split_file_v1"])
def test_committed_split_files_are_valid(key: str) -> None:
    per_class = PARAMS["split"]["per_class"]
    assert sum(per_class.values()) == IMAGES_PER_CLASS
    rows = read_split_csv(REPO_ROOT / PARAMS["data"][key])
    assert len(rows) == len(NEU_CLASSES) * IMAGES_PER_CLASS
    assert_valid_split(rows, per_class, NEU_CLASSES)


def test_committed_v2_keeps_every_confirmed_near_copy_in_one_split() -> None:
    c = PARAMS["near_duplicates_cnn"]
    pairs = confirmed(read_overlaps(REPO_ROOT / c["pairs_file"]), c["confirm_pixel_corr"])
    assert pairs, "expected near-copies in the pairs file (TASK-008b)"
    split_of = {r[0]: r[2] for r in read_split_csv(REPO_ROOT / PARAMS["data"]["split_file"])}
    crossing = [(o.path_a, o.path_b) for o in pairs if split_of[o.path_a] != split_of[o.path_b]]
    assert crossing == []


def test_v2_keeps_v1_counts_and_images() -> None:
    v1 = read_split_csv(REPO_ROOT / PARAMS["data"]["split_file_v1"])
    v2 = read_split_csv(REPO_ROOT / PARAMS["data"]["split_file"])
    assert sorted(r[:2] for r in v1) == sorted(r[:2] for r in v2)


@pytest.mark.needs_data
def test_committed_splits_reproduce_from_raw_data(tmp_path: Path, monkeypatch) -> None:
    raw_dir = REPO_ROOT / PARAMS["data"]["raw_dir"]
    s = PARAMS["split"]
    images = list_images(raw_dir)
    monkeypatch.chdir(REPO_ROOT)  # the pairs file path in params is relative to the repo
    v2 = make_group_split(
        images,
        s["per_class"],
        int(s["seed"]),
        near_copy_groups(PARAMS),
        int(s["groups"]["max_size_outside_last"]),
    )
    for rows, key in (
        (make_split(images, s["per_class"], int(s["seed"])), "split_file_v1"),
        (v2, "split_file"),
    ):
        regenerated = tmp_path / f"{key}.csv"
        write_split_csv(rows, regenerated)
        assert regenerated.read_bytes() == (REPO_ROOT / PARAMS["data"][key]).read_bytes(), key
        assert all((raw_dir / r[0]).is_file() for r in rows)
