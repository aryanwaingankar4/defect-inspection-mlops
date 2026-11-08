"""The one preprocessing pipeline for training, evaluation and serving (ADR-016).

Two steps, so training-only augmentation (in ``src/training/``) can sit between them:

    serving / evaluation:  load_image(source) -> to_model_input(image, cfg)
    training:              load_image(source) -> augment(image) -> to_model_input(image, cfg)

Pillow + NumPy only: the serving image has no PyTorch. Training turns the returned
array into a tensor itself. Never resize, normalise or change channels anywhere else.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError

ImageSource = str | Path | bytes | BinaryIO

INTERPOLATIONS: dict[str, Image.Resampling] = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}
CHANNELS = 3  # ImageNet-pretrained input; grayscale is replicated (ADR-016)


class PreprocessError(ValueError):
    """The input cannot be turned into a model input (e.g. it does not decode)."""


@dataclass(frozen=True)
class PreprocessConfig:
    """Settings from the ``preprocess`` section of ``configs/params.yaml``."""

    image_size: int
    interpolation: str
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(
                f"interpolation must be one of {sorted(INTERPOLATIONS)}, got {self.interpolation!r}"
            )
        if len(self.mean) != CHANNELS or len(self.std) != CHANNELS:
            raise ValueError(f"mean and std need {CHANNELS} values each")
        if any(s <= 0 for s in self.std):
            raise ValueError("std values must be positive")

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> PreprocessConfig:
        """Build from the full params dict (reads ``params['preprocess']``)."""
        p = params["preprocess"]
        return cls(
            image_size=int(p["image_size"]),
            interpolation=str(p["interpolation"]),
            mean=tuple(float(m) for m in p["mean"]),
            std=tuple(float(s) for s in p["std"]),
        )


def load_image(source: ImageSource) -> Image.Image:
    """Open a file path, raw bytes or a binary file object as a grayscale (``L``) image.

    Raises ``PreprocessError`` if the data is not a decodable image.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    try:
        with Image.open(source) as img:
            return img.convert("L")  # convert() decodes fully, so the file can close
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise PreprocessError(f"cannot decode image: {exc}") from exc


def to_model_input(image: Image.Image, cfg: PreprocessConfig) -> np.ndarray:
    """Resize, scale to [0, 1], replicate to 3 channels and normalise.

    Returns a C-contiguous ``float32`` array of shape ``(3, image_size, image_size)``.
    """
    if image.mode != "L":
        raise PreprocessError(f"expected a grayscale 'L' image from load_image, got {image.mode!r}")
    size = (cfg.image_size, cfg.image_size)
    resized = image.resize(size, resample=INTERPOLATIONS[cfg.interpolation])
    gray = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    chw = np.repeat(gray[np.newaxis, :, :], CHANNELS, axis=0)
    mean = np.asarray(cfg.mean, dtype=np.float32).reshape(CHANNELS, 1, 1)
    std = np.asarray(cfg.std, dtype=np.float32).reshape(CHANNELS, 1, 1)
    return np.ascontiguousarray((chw - mean) / std, dtype=np.float32)
