import csv
import json
import random
from pathlib import Path

import pytest
from PIL import Image

from src.data import near_duplicates as nd
from src.data.split import write_split_csv

HASH_SIZE = 8


def noise_image(seed: int, size: int = 64) -> Image.Image:
    rng = random.Random(seed)
    return Image.frombytes("L", (size, size), bytes(rng.randrange(256) for _ in range(size**2)))


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, list[tuple[str, str, str]]]:
    """Synthetic images: an exact cross-split copy, a slightly brightened copy, a same-split
    copy (must be ignored), and unrelated images."""
    raw = tmp_path / "raw"
    (raw / "c1").mkdir(parents=True)
    (raw / "c2").mkdir()
    base = noise_image(1)
    base.save(raw / "c1" / "base.png")
    base.save(raw / "c1" / "exact_copy.png")
    base.point(lambda v: min(255, v + 6)).save(raw / "c2" / "bright_copy.png")
    base.save(raw / "c1" / "same_split_copy.png")
    noise_image(2).save(raw / "c2" / "other_a.png")
    noise_image(3).save(raw / "c1" / "other_b.png")
    rows = [
        ("c1/base.png", "c1", "train_v1"),
        ("c1/exact_copy.png", "c1", "test_frozen"),
        ("c2/bright_copy.png", "c2", "val"),
        ("c1/same_split_copy.png", "c1", "train_v1"),
        ("c2/other_a.png", "c2", "production_stream"),
        ("c1/other_b.png", "c1", "production_stream"),
    ]
    return raw, rows


def hashes_for(raw: Path, rows) -> list[int]:
    return [nd.phash_int(raw / r[0], HASH_SIZE) for r in rows]


def test_hamming() -> None:
    assert nd.hamming(0b1010, 0b1010) == 0
    assert nd.hamming(0b1010, 0b0101) == 4
    assert nd.hamming(0, (1 << 64) - 1) == 64


def test_phash_is_64_bit_and_stable_across_channels(tmp_path: Path) -> None:
    img = noise_image(7)
    img.save(tmp_path / "gray.png")
    img.convert("RGB").save(tmp_path / "rgb.png")
    h_gray = nd.phash_int(tmp_path / "gray.png", HASH_SIZE)
    assert h_gray < 2**64
    assert h_gray == nd.phash_int(tmp_path / "rgb.png", HASH_SIZE)


def test_only_cross_split_pairs_are_compared(dataset) -> None:
    raw, rows = dataset
    pairs = list(nd.cross_split_pairs(rows, hashes_for(raw, rows)))
    assert all(p.a[2] != p.b[2] for p in pairs)
    same_split = {("c1/base.png", "c1/same_split_copy.png")}
    assert not same_split & {(p.a[0], p.b[0]) for p in pairs}


def test_within_split_copies_reported_separately(dataset) -> None:
    raw, rows = dataset
    pairs = nd.within_split_close_pairs(rows, hashes_for(raw, rows), max_distance=2)
    assert [(p.distance, p.a[0], p.b[0]) for p in pairs] == [
        (0, "c1/base.png", "c1/same_split_copy.png")
    ]


def test_finds_exact_and_near_copies_but_not_unrelated(dataset) -> None:
    raw, rows = dataset
    within, top, _ = nd.find_pairs(rows, hashes_for(raw, rows), max_distance=4, top_k=3)
    found = {frozenset((p.a[0], p.b[0])): p.distance for p in within}
    assert found[frozenset(("c1/base.png", "c1/exact_copy.png"))] == 0
    assert found[frozenset(("c1/base.png", "c2/bright_copy.png"))] <= 4
    assert not any("other_" in a + b for a, b in map(tuple, found))
    assert [p.distance for p in top] == sorted(p.distance for p in top)
    assert top[0].distance == 0


def test_summary_counts_are_cumulative(dataset) -> None:
    raw, rows = dataset
    within, _, histogram = nd.find_pairs(rows, hashes_for(raw, rows), 8, 3)
    summary = nd.summarise(histogram, within, [0, 2, 4, 6, 8], 2)
    counts = [summary["cumulative_counts_le_distance"][str(d)]["pairs"] for d in (0, 2, 4, 6, 8)]
    assert counts == sorted(counts)
    assert counts[0] >= 2  # base~exact_copy and exact_copy~same_split_copy (different splits)
    c0 = summary["cumulative_counts_le_distance"]["0"]
    assert c0["same_class"] + c0["different_class"] == c0["pairs"]
    assert summary["cross_split_pairs_compared"] == sum(histogram.values())


def test_outputs_written(dataset, tmp_path: Path) -> None:
    raw, rows = dataset
    within, top, histogram = nd.find_pairs(rows, hashes_for(raw, rows), 8, 2)
    out_csv, out_png = tmp_path / "out" / "pairs.csv", tmp_path / "out" / "grid.png"
    nd.write_pairs_csv(within, out_csv)
    nd.draw_grid(top, raw, out_png, tile=32)
    with out_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == nd.PAIR_FIELDS
        written = list(reader)
    assert len(written) == len(within)
    by_path = {r[0]: r for r in rows}
    for rec in written:  # every column holds the right value (not swapped)
        for side in ("a", "b"):
            path, cls, split = by_path[rec[f"path_{side}"]]
            assert (rec[f"class_{side}"], rec[f"split_{side}"]) == (cls, split)
    with Image.open(out_png) as grid:
        assert grid.height >= 2 * 32
    json.dumps(nd.summarise(histogram, within, [0, 2], 2))  # serialisable


def test_main_end_to_end(dataset, tmp_path: Path) -> None:
    raw, rows = dataset
    split_csv = tmp_path / "split.csv"
    write_split_csv(rows, split_csv)
    out = tmp_path / "results"
    params = tmp_path / "params.yaml"
    params.write_text(
        f"data:\n  raw_dir: {raw.as_posix()}\n  split_file: {split_csv.as_posix()}\n"
        "near_duplicates:\n  hash_size: 8\n  max_distance: 8\n"
        "  report_distances: [0, 2, 4, 6, 8]\n  alert_distance: 2\n  grid_top_k: 3\n"
        f"  out_csv: {(out / 'nd.csv').as_posix()}\n"
        f"  out_summary: {(out / 'nd.json').as_posix()}\n"
        f"  out_grid: {(out / 'private' / 'grid.png').as_posix()}\n",
        encoding="utf-8",
    )
    assert nd.main(["--config", str(params)]) == 0
    assert (out / "nd.csv").is_file() and (out / "nd.json").is_file()
    assert (out / "private" / "grid.png").is_file()
