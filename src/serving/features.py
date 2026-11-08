"""Per-image features for the prediction log and drift checks (TASK-019, ADR-029). NumPy only.

Computed on the decoded greyscale image at its original size (grey levels 0-255), so the
drift generators (TASK-033) and Evidently reports (TASK-034) use exactly the same numbers:

- ``brightness``: mean grey level;
- ``contrast``: standard deviation of the grey levels;
- ``sharpness``: variance of the 3x3 Laplacian (4-neighbour), over interior pixels.

Sharpness depends on resolution: compare it only between images of the same size.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def laplacian(gray: np.ndarray) -> np.ndarray:
    """4-neighbour Laplacian of a 2-D array, interior pixels only (shape minus 2 per side)."""
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2 or min(g.shape) < 3:
        raise ValueError("need a 2-D image of at least 3x3 pixels")
    return g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]


def image_features(image: Image.Image) -> dict[str, float]:
    """Brightness, contrast and sharpness of a greyscale (``L``) image."""
    if image.mode != "L":
        raise ValueError(f"expected a greyscale 'L' image, got {image.mode!r}")
    g = np.asarray(image, dtype=np.float64)
    return {
        "brightness": float(g.mean()),
        "contrast": float(g.std()),
        "sharpness": float(laplacian(g).var()),
    }
