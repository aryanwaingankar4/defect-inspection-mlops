"""Shift-tolerant near-copy check (TASK-008b). A fake embedder: no weights, no download."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from src.config import load_params
from src.data import near_duplicates_cnn as nc
from src.data.groups import Overlap, connected_groups, read_overlaps

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")


def noise(seed: int, size: int = 80) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (size, size), dtype=np.uint8)


def fake_embed(images) -> np.ndarray:
    """Shift-tolerant enough for tests: grey-level histogram (ignores pixel positions)."""
    hist = [np.bincount(np.asarray(im).ravel() // 16, minlength=16) for im in images]
    return np.stack(hist).astype(np.float32)


def small_cfg(tmp_path: Path, **over) -> dict:
    return {
        **PARAMS["near_duplicates_cnn"],
        "highpass_px": 3,
        "max_shift": 4,
        "candidate_cosine": -1.0,
        "confirm_pixel_corr": 0.9,
        "keep_pixel_corr": -1.0,
        "calib_same_class_pairs": 20,
        "calib_pixel_pairs": 4,
        "calib_shift_images_per_class": 2,
        "calib_shifts": [2, 4],
        "watch_pairs": [["c1_base.png", "c1_shifted.png"]],
        "pairs_file": str(tmp_path / "out" / "pairs.csv"),
        "pairs_summary": str(tmp_path / "out" / "pairs_summary.json"),
        "out_csv": str(tmp_path / "out" / "report.csv"),
        "out_summary": str(tmp_path / "out" / "report.json"),
        "out_grid": str(tmp_path / "out" / "private" / "grid.png"),
        **over,
    }


def cfg(tmp_path: Path, **over) -> nc.CnnDupConfig:
    return nc.CnnDupConfig.from_params({"near_duplicates_cnn": small_cfg(tmp_path, **over)})


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, list[tuple[str, str]]]:
    """c1_base / c1_shifted are crops of one surface moved (3, 2) px; the rest are unrelated."""
    raw = tmp_path / "raw"
    raw.mkdir()
    surface = noise(1)
    files = {
        "c1_base.png": surface[0:64, 0:64],
        "c1_shifted.png": surface[3:67, 2:66],
        "c1_other.png": noise(2)[:64, :64],
        "c2_a.png": noise(3)[:64, :64] // 2,
        "c2_b.png": noise(4)[:64, :64] // 2 + 100,
        "c2_c.png": noise(5)[:64, :64] // 3,
    }
    for name, arr in files.items():
        Image.fromarray(arr.astype(np.uint8)).save(raw / name)
    return raw, [(name, name.split("_")[0]) for name in files]


# --- pixel check


def brute_force_correlation(a: np.ndarray, b: np.ndarray, max_shift: int) -> float:
    """Reference: loop over every shift (slow but obvious)."""
    h, w = a.shape
    best = -1.0
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            pa = a[max(0, dy) : h + min(0, dy), max(0, dx) : w + min(0, dx)]
            pb = b[max(0, -dy) : h + min(0, -dy), max(0, -dx) : w + min(0, -dx)]
            pa, pb = pa - pa.mean(), pb - pb.mean()
            denom = np.sqrt(np.sum(pa * pa) * np.sum(pb * pb))
            if denom > 0:
                best = max(best, float(np.sum(pa * pb) / denom))
    return best


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fft_correlation_matches_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(30, 26)), rng.normal(size=(30, 26))
    b[5:, 3:] += 0.8 * a[:-5, :-3]  # partly correlated at one shift
    assert nc.best_shift_correlation(a, b, 6) == pytest.approx(brute_force_correlation(a, b, 6))


def test_best_shift_correlation_finds_shifted_crops() -> None:
    surface = noise(1).astype(np.float64)
    a, b = surface[0:64, 0:64], surface[3:67, 5:69]  # b is a moved 3 down, 5 right
    assert nc.best_shift_correlation(a, b, max_shift=6) == pytest.approx(1.0)
    assert nc.best_shift_correlation(a, b, max_shift=2) < 0.5  # shift out of range
    assert nc.best_shift_correlation(a, noise(9)[:64, :64], max_shift=6) < 0.5
    flat = np.full((64, 64), 7.0)
    assert nc.best_shift_correlation(flat, flat, max_shift=2) == 0.0  # no NaN on flat images


def test_highpass_removes_lighting_gradients() -> None:
    plane = np.add.outer(np.arange(40.0), 2 * np.arange(40.0))  # a pure brightness ramp
    out = nc.highpass(plane, 9)
    assert out.shape == (32, 32)
    assert np.abs(out).max() == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match="odd"):
        nc.highpass(plane, 4)


def test_shifted_crops_are_equal_size_and_offset() -> None:
    a, b = nc.shifted_crops(Image.fromarray(noise(1)), shift=4, max_shift=8)
    assert a.size == b.size == (72, 72)
    assert np.array_equal(np.asarray(a)[4:, 4:], np.asarray(b)[:-4, :-4])


def test_same_class_sampling_is_seeded() -> None:
    classes = ["a", "a", "a", "b", "b"]
    a1, b1 = nc.sample_same_class_pairs(classes, 50, np.random.default_rng(3))
    a2, b2 = nc.sample_same_class_pairs(classes, 50, np.random.default_rng(3))
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert all(classes[a] == classes[b] and a != b for a, b in zip(a1, b1, strict=True))


def test_connected_groups_follow_chains() -> None:
    edges = [("c", "d"), ("a", "b"), ("b", "c"), ("x", "y")]
    assert connected_groups(edges) == [["a", "b", "c", "d"], ["x", "y"]]


# --- step 1: pairs


def test_find_overlaps_confirms_only_the_shifted_copy(dataset, tmp_path: Path) -> None:
    raw, images = dataset
    summary, overlaps = nc.find_overlaps(images, raw, cfg(tmp_path), fake_embed)
    assert summary["pairs_compared"] == 15  # 6 images, all pairs (split-independent)
    assert summary["candidates"] == len(overlaps) == 15  # every pair passes cosine >= -1
    sure = [(o.path_a, o.path_b) for o in overlaps if o.pixel_corr >= 0.9]
    assert sure == [("c1_base.png", "c1_shifted.png")]  # a high cosine alone is not enough
    assert overlaps[0].pixel_corr == pytest.approx(1.0)  # best first
    assert summary["groups"] == {
        "count": 1,
        "images": 2,
        "size_histogram": {"2": 1},
        "images_by_class": {"c1": 2},
        "largest": {"size": 2},
    }
    (watch,) = summary["watch_pairs"]
    assert watch["confirmed"] is True and watch["cosine_rank_among_all_pairs"] >= 1
    assert set(summary["calibration"]["cosine_shifted_crops"]["by_shift_px"]) == {"2", "4"}


def test_candidates_need_the_cosine(dataset, tmp_path: Path) -> None:
    raw, images = dataset
    c = cfg(tmp_path, candidate_cosine=1.01)
    summary, overlaps = nc.find_overlaps(images, raw, c, fake_embed)
    assert summary["candidates"] == 0 and overlaps == []
    assert summary["watch_pairs"][0]["confirmed"] is False


def test_too_many_candidates_is_refused(dataset, tmp_path: Path) -> None:
    raw, images = dataset
    with pytest.raises(ValueError, match="max_candidates"):
        nc.find_overlaps(images, raw, cfg(tmp_path, max_candidates=2), fake_embed)


# --- step 2: report per split


def overlap(pix: float, a: str, b: str) -> Overlap:
    return Overlap(pix, 0.9, a, "c", b, "c")


def test_cross_split_report_counts_only_cross_split(tmp_path: Path) -> None:
    rows = [("a", "c", "train_v1"), ("b", "c", "val"), ("x", "c", "val"), ("y", "c", "val")]
    overlaps = [overlap(0.95, "a", "b"), overlap(0.99, "x", "y"), overlap(0.2, "a", "x")]
    c = cfg(tmp_path, confirm_pixel_corr=0.3, watch_pairs=[["a", "b"]])
    summary, table = nc.cross_split_report(overlaps, rows, c)
    assert [(t["path_a"], t["path_b"], t["confirmed"]) for t in table] == [
        ("a", "b", 1),
        ("a", "x", 0),
    ]
    assert summary["confirmed_cross_split"] == 1
    assert summary["confirmed_cross_split_by_split_pair"] == {"train_v1 / val": 1}
    assert summary["watch_pairs"][0]["same_split"] is False
    with pytest.raises(ValueError, match="not in the split"):
        nc.cross_split_report([overlap(0.5, "a", "zzz")], rows, c)


def write_params(tmp_path: Path, raw: Path, rows, **over) -> Path:
    split_csv = tmp_path / "split.csv"
    with split_csv.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows([("path", "class", "split"), *rows])
    params = {
        **PARAMS,
        "data": {"raw_dir": str(raw), "split_file": str(split_csv)},
        "near_duplicates_cnn": small_cfg(tmp_path, **over),
    }
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(params), encoding="utf-8")
    return path


def test_pairs_then_report_end_to_end(dataset, tmp_path: Path, monkeypatch) -> None:
    raw, images = dataset
    monkeypatch.setattr(nc, "timm_embedder", lambda *a: fake_embed)
    monkeypatch.setattr(nc, "list_images", lambda _: {"all": [p for p, _ in images]})
    leaky = [(p, c, "test_frozen" if p == "c1_shifted.png" else "train_v1") for p, c in images]
    config = write_params(tmp_path, raw, leaky)
    c = cfg(tmp_path)

    assert nc.main(["pairs", "--config", str(config)]) == 0
    assert read_overlaps(c.pairs_file)[0].path_a == "c1_base.png"
    assert b"\r\n" not in c.pairs_summary.read_bytes()
    assert "pairs_file" not in json.loads(c.pairs_summary.read_bytes())["params"]

    assert nc.main(["report", "--config", str(config)]) == 0
    assert nc.main(["report", "--config", str(config), "--require-clean"]) == 1
    report = json.loads(c.out_summary.read_bytes())
    assert report["confirmed_cross_split"] == 1
    with Image.open(c.out_grid) as grid:
        assert grid.height > 0

    clean = [(p, cl, "train_v1") for p, cl, _ in leaky]
    config = write_params(tmp_path, raw, clean)
    assert nc.main(["report", "--config", str(config), "--require-clean"]) == 0


def test_report_refuses_a_different_image_set(dataset, tmp_path: Path, monkeypatch) -> None:
    raw, images = dataset
    monkeypatch.setattr(nc, "timm_embedder", lambda *a: fake_embed)
    monkeypatch.setattr(nc, "list_images", lambda _: {"all": [p for p, _ in images]})
    rows = [(p, c, "train_v1") for p, c in images]
    assert nc.main(["pairs", "--config", str(write_params(tmp_path, raw, rows))]) == 0
    with pytest.raises(ValueError, match="different image set"):
        nc.main(["report", "--config", str(write_params(tmp_path, raw, rows[:-1]))])


def test_params_keep_pixels_private() -> None:
    c = PARAMS["near_duplicates_cnn"]
    assert c["out_grid"].startswith("results/private/")
    for key in ("out_csv", "out_summary", "pairs_summary", "pairs_file"):
        assert not c[key].startswith("results/private/")
        assert Path(c[key]).suffix in {".csv", ".json"}


@pytest.mark.needs_data
@pytest.mark.slow
def test_real_exact_duplicate_scores_one() -> None:
    """Positive control: patches_101 and patches_105 are byte-identical (TASK-004)."""
    from src.serving.preprocess import PreprocessConfig

    raw = REPO_ROOT / PARAMS["data"]["raw_dir"] / "NEU-DET" / "train" / "images" / "patches"
    embed = nc.timm_embedder("efficientnet_b0", 2, PreprocessConfig.from_params(PARAMS))
    images = [nc.load_image(raw / n) for n in ("patches_101.jpg", "patches_105.jpg")]
    e = nc.l2_normalise(embed(images))
    assert float(e[0] @ e[1]) == pytest.approx(1.0, abs=1e-5)
