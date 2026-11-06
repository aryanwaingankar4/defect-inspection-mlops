"""Training-only augmentation on the grayscale ``L`` image (between the two preprocess steps).

    load_image -> augment -> to_model_input        (src/serving/preprocess.py)

Never used for validation, evaluation or serving. Randomness comes only from the
``numpy.random.Generator`` passed in, so a seeded generator gives a repeatable result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageStat


@dataclass(frozen=True)
class AugmentConfig:
    """Settings from ``params.yaml`` -> ``train.augment``."""

    hflip_p: float
    vflip_p: float
    rotation_deg: float
    brightness: float
    contrast: float

    def __post_init__(self) -> None:
        for name in ("hflip_p", "vflip_p"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("rotation_deg", "brightness", "contrast"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.brightness >= 1 or self.contrast >= 1:
            raise ValueError("brightness and contrast must be < 1 (factor stays positive)")

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> AugmentConfig:
        a = params["train"]["augment"]
        return cls(**{k: float(a[k]) for k in cls.__dataclass_fields__})


def augment(image: Image.Image, cfg: AugmentConfig, rng: np.random.Generator) -> Image.Image:
    """Random h/v flip, rotation (mean-grey corners), brightness, contrast. Keeps mode + size."""
    if image.mode != "L":
        raise ValueError(f"augment expects a grayscale 'L' image, got {image.mode!r}")
    # Draw every random number up front, in a fixed order, so results only depend on the RNG.
    hflip = rng.random() < cfg.hflip_p
    vflip = rng.random() < cfg.vflip_p
    angle = rng.uniform(-cfg.rotation_deg, cfg.rotation_deg)
    brightness = rng.uniform(1 - cfg.brightness, 1 + cfg.brightness)
    contrast = rng.uniform(1 - cfg.contrast, 1 + cfg.contrast)

    out = image
    if hflip:
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if vflip:
        out = out.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if angle != 0:
        fill = round(ImageStat.Stat(out).mean[0])  # black corners would be a learnable artefact
        out = out.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=fill)
    out = ImageEnhance.Brightness(out).enhance(brightness)
    return ImageEnhance.Contrast(out).enhance(contrast)
