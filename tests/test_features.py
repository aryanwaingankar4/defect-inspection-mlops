"""TASK-019: per-image drift features (ADR-029), hand-checked values."""

import numpy as np
import pytest
from PIL import Image

from src.serving.features import image_features, laplacian


def gray(a) -> Image.Image:
    return Image.fromarray(np.asarray(a, dtype=np.uint8), mode=None)


def test_flat_image():
    f = image_features(gray(np.full((10, 10), 80)))
    assert f == {"brightness": 80.0, "contrast": 0.0, "sharpness": 0.0}


def test_linear_ramp_has_zero_laplacian():
    ramp = np.tile(np.arange(0, 200, 10), (20, 1))  # second derivative is zero
    f = image_features(gray(ramp))
    assert f["sharpness"] == 0.0 and f["contrast"] > 0


def test_checkerboard_hand_values():
    board = (np.indices((8, 8)).sum(axis=0) % 2) * 100  # 0 / 100
    lap = laplacian(board)
    # each interior pixel: 4 neighbours of the other colour -> +400 on 0, -400 on 100
    assert set(np.unique(lap)) == {-400.0, 400.0}
    f = image_features(gray(board))
    assert f["brightness"] == 50.0 and f["contrast"] == 50.0
    assert f["sharpness"] == pytest.approx(400.0**2)


def test_blur_lowers_sharpness():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (64, 64))
    blurred = (sharp + np.roll(sharp, 1, 0) + np.roll(sharp, 1, 1) + np.roll(sharp, 1, (0, 1))) / 4
    assert image_features(gray(blurred))["sharpness"] < image_features(gray(sharp))["sharpness"]


def test_needs_a_greyscale_image():
    with pytest.raises(ValueError, match="greyscale"):
        image_features(Image.new("RGB", (8, 8)))
    with pytest.raises(ValueError):
        laplacian(np.zeros((2, 5)))
