"""Shared helpers for training/publishing tests: a tiny synthetic NEU-like dataset."""

import copy
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import load_params
from src.data.split import write_split_csv

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS_FILE = REPO_ROOT / "configs" / "params.yaml"
PARAMS = load_params(PARAMS_FILE)
CLASSES = ["alpha", "beta", "gamma"]
PER_SPLIT = {"train_v1": 3, "val": 2, "test_frozen": 1}


def make_synthetic_data(root: Path) -> tuple[Path, Path]:
    """Tiny NEU-like tree + split CSV. Each class has its own brightness so it is learnable."""
    raw = root / "raw"
    rows = []
    rng = np.random.default_rng(0)
    for ci, cls in enumerate(CLASSES):
        folder = raw / "NEU-DET" / "train" / "images" / cls
        folder.mkdir(parents=True)
        i = 0
        for split, n in PER_SPLIT.items():
            for _ in range(n):
                base = 40 + 80 * ci
                arr = np.clip(base + rng.normal(0, 10, (40, 40)), 0, 255).astype(np.uint8)
                rel = f"NEU-DET/train/images/{cls}/{cls}_{i}.jpg"
                Image.fromarray(arr).convert("RGB").save(raw / rel, quality=95)
                rows.append((rel, cls, split))
                i += 1
    csv_path = root / "split.csv"
    write_split_csv(rows, csv_path)
    return raw, csv_path


def tiny_params(raw: Path, csv_path: Path, out: Path, **train_overrides) -> dict:
    """Real params with tiny, CPU-friendly training settings and outputs under ``out``."""
    p = copy.deepcopy(PARAMS)
    p["data"] = {"raw_dir": raw.as_posix(), "split_file": csv_path.as_posix()}
    p["preprocess"]["image_size"] = 32
    tiny = {
        "pretrained": False,
        "batch_size": 4,
        "max_epochs": 1,
        "patience": 1,
        "num_workers": 0,
        "out_checkpoint": (out / "best.pt").as_posix(),
        "out_dir": (out / "results").as_posix(),
    }
    p["train"].update({**tiny, **train_overrides})
    return p
