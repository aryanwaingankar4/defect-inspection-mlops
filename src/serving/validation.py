"""Input checks for uploaded images (TASK-018, ADR-028). Limits from ``params.yaml`` -> ``serving``.

Order: size -> header (format, dimensions, before any pixel is decoded) -> decode with the shared
``preprocess.load_image`` -> near-blank check. Each failure is an ``InputRejected`` with an HTTP
status and a short code for the JSON error body and the rejection metric.
"""

from __future__ import annotations

import io
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.serving.preprocess import PreprocessError, load_image


class InputRejected(ValueError):
    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class ValidationConfig:
    max_upload_bytes: int
    allowed_formats: tuple[str, ...]
    min_side: int
    max_pixels: int
    near_blank_max_std: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> ValidationConfig:
        s = params["serving"]
        return cls(
            max_upload_bytes=int(s["max_upload_bytes"]),
            allowed_formats=tuple(str(f).upper() for f in s["allowed_formats"]),
            min_side=int(s["min_side"]),
            max_pixels=int(s["max_pixels"]),
            near_blank_max_std=float(s["near_blank_max_std"]),
        )


def check_size(n_bytes: int, cfg: ValidationConfig) -> None:
    if n_bytes == 0:
        raise InputRejected(422, "empty_file", "the uploaded file is empty")
    if n_bytes > cfg.max_upload_bytes:
        raise InputRejected(
            413, "too_large", f"the file is larger than {cfg.max_upload_bytes} bytes"
        )


def check_header(data: bytes, cfg: ValidationConfig) -> None:
    """Format and dimensions from the header only (no pixel decoding)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as img:
                fmt, (w, h) = (img.format or "").upper(), img.size
    except Image.DecompressionBombError as exc:
        raise InputRejected(422, "too_many_pixels", "the image has too many pixels") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InputRejected(422, "undecodable", "the file is not a decodable image") from exc
    if fmt not in cfg.allowed_formats:
        raise InputRejected(
            415,
            "unsupported_format",
            f"image format {fmt or 'unknown'} is not allowed; use {', '.join(cfg.allowed_formats)}",
        )
    if w * h > cfg.max_pixels:
        raise InputRejected(
            422, "too_many_pixels", f"the image has more than {cfg.max_pixels} pixels"
        )
    if min(w, h) < cfg.min_side:
        raise InputRejected(422, "too_small", f"each side must be at least {cfg.min_side} pixels")


def is_near_blank(image: Image.Image, cfg: ValidationConfig) -> bool:
    return float(np.asarray(image, dtype=np.float64).std()) < cfg.near_blank_max_std


def decode_checked(data: bytes, cfg: ValidationConfig) -> Image.Image:
    """Run every check and return the grayscale image from the shared ``load_image``."""
    check_size(len(data), cfg)
    check_header(data, cfg)
    try:
        image = load_image(data)
    except PreprocessError as exc:
        raise InputRejected(422, "undecodable", "the file is not a decodable image") from exc
    if is_near_blank(image, cfg):
        raise InputRejected(422, "near_blank", "the image is nearly blank (almost no contrast)")
    return image
