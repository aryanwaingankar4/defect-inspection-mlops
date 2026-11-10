"""Synthetic non-steel images for OOD evaluation (TASK-017, ADR-027).

Generated in code from a seed: no download and no NEU pixels. Each image is a 200x200
greyscale (``L``) image like NEU, returned as PNG bytes so it goes through the same
``load_image`` -> ``to_model_input`` path as a real request. Families:

  gaussian_noise, uniform_noise   pixel noise
  gradient                        smooth linear ramp at a random angle
  stripes                         sinusoidal stripes, random frequency and angle
  checkerboard                    random cell size
  shapes                          random rectangles and ellipses on a grey background
  blobs                           blurred noise, contrast stretched
  near_flat                       almost uniform grey (the API rejects these anyway)
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

Generator = Callable[[np.random.Generator, int], np.ndarray]


def _gaussian_noise(rng: np.random.Generator, s: int) -> np.ndarray:
    return rng.normal(rng.uniform(60, 190), rng.uniform(15, 60), (s, s))


def _uniform_noise(rng: np.random.Generator, s: int) -> np.ndarray:
    return rng.uniform(0, 255, (s, s))


def _grid(s: int, angle: float) -> np.ndarray:
    yy, xx = np.mgrid[0:s, 0:s] / s
    return xx * np.cos(angle) + yy * np.sin(angle)


def _gradient(rng: np.random.Generator, s: int) -> np.ndarray:
    ramp = _grid(s, rng.uniform(0, np.pi))
    lo, hi = sorted(rng.uniform(0, 255, 2))
    return lo + (ramp - ramp.min()) / np.ptp(ramp) * (hi - lo)


def _stripes(rng: np.random.Generator, s: int) -> np.ndarray:
    phase = 2 * np.pi * rng.uniform(2, 40) * _grid(s, rng.uniform(0, np.pi))
    return 128 + rng.uniform(30, 120) * np.sin(phase)


def _checkerboard(rng: np.random.Generator, s: int) -> np.ndarray:
    cell = int(rng.integers(4, 51))
    yy, xx = np.mgrid[0:s, 0:s] // cell
    lo, hi = sorted(rng.uniform(0, 255, 2))
    return np.where((yy + xx) % 2 == 0, lo, hi)


def _shapes(rng: np.random.Generator, s: int) -> np.ndarray:
    img = Image.new("L", (s, s), int(rng.integers(0, 256)))
    draw = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(5, 16))):
        x0, y0 = rng.integers(0, s, 2)
        x1, y1 = x0 + rng.integers(10, s // 2), y0 + rng.integers(10, s // 2)
        shape = draw.rectangle if rng.random() < 0.5 else draw.ellipse
        shape([int(x0), int(y0), int(x1), int(y1)], fill=int(rng.integers(0, 256)))
    return np.asarray(img, dtype=np.float64)


def _blobs(rng: np.random.Generator, s: int) -> np.ndarray:
    noise = Image.fromarray(rng.uniform(0, 255, (s, s)).astype(np.uint8))
    blurred = np.asarray(noise.filter(ImageFilter.GaussianBlur(rng.uniform(3, 10))), np.float64)
    return (blurred - blurred.min()) / max(np.ptp(blurred), 1e-9) * 255


def _near_flat(rng: np.random.Generator, s: int) -> np.ndarray:
    return rng.uniform(20, 235) + rng.normal(0, 1.0, (s, s))


FAMILIES: dict[str, Generator] = {
    "gaussian_noise": _gaussian_noise,
    "uniform_noise": _uniform_noise,
    "gradient": _gradient,
    "stripes": _stripes,
    "checkerboard": _checkerboard,
    "shapes": _shapes,
    "blobs": _blobs,
    "near_flat": _near_flat,
}


def to_png(pixels: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def generate(
    families: Sequence[str], n_per_family: int, seed: int, size: int = 200
) -> list[tuple[str, bytes]]:
    """``(family, png_bytes)`` pairs, deterministic for a given seed (one stream per family)."""
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown synthetic families: {unknown}")
    out = []
    for k, name in enumerate(families):
        rng = np.random.default_rng([seed, k])
        out.extend((name, to_png(FAMILIES[name](rng, size))) for _ in range(n_per_family))
    return out
