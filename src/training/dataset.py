"""Dataset over the committed split CSV, with a hard guard on ``test_frozen``.

Per image: load_image -> augment (training only) -> to_model_input -> torch tensor.
Only the promotion gate and final evaluation may read ``test_frozen``; they must pass
``allow_test=True`` explicitly. Training never does.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.split import read_split_csv
from src.serving.preprocess import PreprocessConfig, load_image, to_model_input
from src.training.augment import AugmentConfig, augment

PROTECTED_SPLIT = "test_frozen"


class TestSetAccessError(RuntimeError):
    """Raised when code asks for ``test_frozen`` without ``allow_test=True``."""

    __test__ = False  # not a pytest test class despite the name


def class_names(split_csv: Path, splits: Sequence[str]) -> list[str]:
    """Sorted class names present in ``splits``; the index order used by the model."""
    wanted = set(splits)
    return sorted({cls for _, cls, split in read_split_csv(split_csv) if split in wanted})


class SplitDataset(Dataset):
    """Images of the given splits from the split CSV, as ``(float32 CHW tensor, label)``."""

    def __init__(
        self,
        split_csv: Path,
        raw_dir: Path,
        splits: Sequence[str],
        classes: Sequence[str],
        preprocess: PreprocessConfig,
        augment_cfg: AugmentConfig | None = None,
        allow_test: bool = False,
        images_per_class: int | None = None,
    ) -> None:
        if PROTECTED_SPLIT in splits and not allow_test:
            raise TestSetAccessError(
                f"'{PROTECTED_SPLIT}' is reserved for the promotion gate and final evaluation; "
                "pass allow_test=True only from those code paths"
            )
        self.raw_dir = Path(raw_dir)
        self.preprocess = preprocess
        self.augment_cfg = augment_cfg
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        wanted = set(splits)
        rows = [(p, c) for p, c, s in read_split_csv(Path(split_csv)) if s in wanted]
        unknown = sorted({c for _, c in rows} - set(self.class_to_idx))
        if unknown:
            raise ValueError(f"classes {unknown} are not in the model's class list")
        if images_per_class is not None:
            rows = _first_n_per_class(rows, images_per_class)
        if not rows:
            raise ValueError(f"no images for splits {sorted(wanted)} in {split_csv}")
        self.paths = [p for p, _ in rows]
        self.labels = [self.class_to_idx[c] for _, c in rows]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = load_image(self.raw_dir / self.paths[index])
        if self.augment_cfg is not None:
            # torch's RNG is seeded per DataLoader worker, so this is repeatable for a fixed seed.
            seed = int(torch.randint(0, 2**63 - 1, (1,)).item())
            image = augment(image, self.augment_cfg, np.random.default_rng(seed))
        return torch.from_numpy(to_model_input(image, self.preprocess)), self.labels[index]

    def class_counts(self) -> Counter[int]:
        return Counter(self.labels)


def _first_n_per_class(rows: list[tuple[str, str]], n: int) -> list[tuple[str, str]]:
    taken: Counter[str] = Counter()
    kept = []
    for path, cls in rows:
        if taken[cls] < n:
            kept.append((path, cls))
            taken[cls] += 1
    return kept
