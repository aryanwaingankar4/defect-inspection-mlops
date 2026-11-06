"""Shared preprocessing (TASK-007, ADR-016). Synthetic images only, so this runs in CI."""

import ast
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.config import load_params
from src.serving.preprocess import (
    PreprocessConfig,
    PreprocessError,
    load_image,
    to_model_input,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
CFG = PreprocessConfig.from_params(PARAMS)
MODULE = REPO_ROOT / "src" / "serving" / "preprocess.py"


def synthetic_gray(size: tuple[int, int] = (200, 200), seed: int = 0) -> np.ndarray:
    """Textured uint8 grayscale pattern (H, W), like a steel surface crop."""
    rng = np.random.default_rng(seed)
    h, w = size[1], size[0]
    ramp = np.linspace(0, 255, w, dtype=np.float64)[np.newaxis, :].repeat(h, axis=0)
    noise = rng.normal(0, 25, size=(h, w))
    return np.clip(ramp * 0.6 + noise + 50, 0, 255).astype(np.uint8)


def encode(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def serve_path(source) -> np.ndarray:
    return to_model_input(load_image(source), CFG)


def unnormalise(x: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    mean = np.asarray(cfg.mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(cfg.std, dtype=np.float32).reshape(3, 1, 1)
    return x * std + mean


def test_parity_file_path_vs_upload_bytes(tmp_path: Path) -> None:
    """Training reads files from disk, serving gets upload bytes: outputs must be identical."""
    jpg = tmp_path / "synthetic.jpg"
    Image.fromarray(synthetic_gray()).convert("RGB").save(jpg, format="JPEG", quality=90)

    from_path = serve_path(jpg)
    from_str = serve_path(str(jpg))
    from_bytes = serve_path(jpg.read_bytes())
    with jpg.open("rb") as f:
        from_fileobj = serve_path(f)

    assert from_path.tobytes() == from_bytes.tobytes()
    assert from_path.tobytes() == from_str.tobytes()
    assert from_path.tobytes() == from_fileobj.tobytes()


@pytest.mark.parametrize("value", [0, 1, 128, 254, 255])
def test_flat_image_gives_exact_normalised_values(value: int) -> None:
    img = Image.new("L", (200, 200), color=value)
    out = to_model_input(img, CFG)
    for c in range(3):
        expected = (value / 255.0 - CFG.mean[c]) / CFG.std[c]
        np.testing.assert_allclose(out[c], expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
@pytest.mark.parametrize("size", [(200, 200), (64, 48), (500, 300)])
def test_shape_dtype_and_equal_channels(mode: str, size: tuple[int, int]) -> None:
    img = Image.fromarray(synthetic_gray(size)).convert(mode)
    out = serve_path(encode(img, "PNG"))
    assert out.shape == (3, CFG.image_size, CFG.image_size)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    pixels = unnormalise(out, CFG)
    np.testing.assert_allclose(pixels[0], pixels[1], atol=1e-6)
    np.testing.assert_allclose(pixels[0], pixels[2], atol=1e-6)
    assert pixels.min() >= -1e-6 and pixels.max() <= 1 + 1e-6


def test_gray_rgb_and_l_give_identical_output() -> None:
    """A 3-channel grayscale file (like the Kaggle NEU JPEGs) equals its 'L' version."""
    gray = Image.fromarray(synthetic_gray())
    out_l = serve_path(encode(gray, "PNG"))
    out_rgb = serve_path(encode(gray.convert("RGB"), "PNG"))
    assert out_l.tobytes() == out_rgb.tobytes()


def test_colour_is_removed_before_the_model() -> None:
    rgb = Image.new("RGB", (200, 200), color=(200, 30, 90))
    out = serve_path(encode(rgb, "PNG"))
    pixels = unnormalise(out, CFG)
    np.testing.assert_allclose(pixels[0], pixels[1], atol=1e-6)


def test_load_image_returns_l() -> None:
    img = load_image(encode(Image.new("RGB", (10, 10)), "JPEG"))
    assert img.mode == "L"


@pytest.mark.parametrize("data", [b"", b"not an image", b"\xff\xd8\xff\xe0truncated"])
def test_undecodable_input_raises(data: bytes) -> None:
    with pytest.raises(PreprocessError):
        load_image(data)


def test_to_model_input_rejects_non_l_images() -> None:
    with pytest.raises(PreprocessError):
        to_model_input(Image.new("RGB", (10, 10)), CFG)


def test_config_comes_from_params() -> None:
    p = PARAMS["preprocess"]
    assert CFG.image_size == p["image_size"]
    assert CFG.interpolation == p["interpolation"]
    assert CFG.mean == tuple(p["mean"])
    assert CFG.std == tuple(p["std"])


def test_image_size_from_config_is_used() -> None:
    cfg = PreprocessConfig(image_size=32, interpolation="bilinear", mean=CFG.mean, std=CFG.std)
    assert to_model_input(Image.new("L", (200, 200)), cfg).shape == (3, 32, 32)


@pytest.mark.parametrize(
    "overrides",
    [
        {"image_size": 0},
        {"interpolation": "area"},
        {"mean": (0.5, 0.5)},
        {"std": (0.2, 0.0, 0.2)},
    ],
)
def test_bad_config_rejected(overrides: dict) -> None:
    fields = {"image_size": 224, "interpolation": "bilinear", "mean": CFG.mean, "std": CFG.std}
    with pytest.raises(ValueError):
        PreprocessConfig(**(fields | overrides))


def test_preprocess_imports_no_torch() -> None:
    """The serving image has no PyTorch: only stdlib, Pillow and NumPy are allowed."""
    allowed = set(sys.stdlib_module_names) | {"PIL", "numpy"}
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= allowed, f"unexpected imports: {sorted(imported - allowed)}"
