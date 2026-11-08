"""TASK-018: input checks (ADR-028). The near-blank limit is checked on NEU train_v1/val images."""

import csv
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.config import load_params
from src.data import synthetic_ood as so
from src.serving import validation as va
from src.serving.preprocess import load_image
from tests import serving_helpers as sh

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
CFG = va.ValidationConfig.from_params(PARAMS)


def rejected(data: bytes, cfg=CFG) -> va.InputRejected:
    with pytest.raises(va.InputRejected) as info:
        va.decode_checked(data, cfg)
    return info.value


def test_good_image_decodes_to_grayscale():
    img = va.decode_checked(sh.image_bytes(100.0), CFG)
    assert img.mode == "L" and img.size == (200, 200)


def test_rgb_image_is_converted_by_the_shared_loader():
    buf = io.BytesIO()
    rgb = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(buf, format="PNG")
    assert va.decode_checked(buf.getvalue(), CFG).mode == "L"


def test_format_is_decided_by_the_bytes_not_the_name():
    jpeg = sh.image_bytes(100.0, fmt="JPEG")
    assert va.decode_checked(jpeg, CFG).mode == "L"
    e = rejected(sh.image_bytes(100.0, fmt="WEBP"))
    assert (e.status, e.code) == (415, "unsupported_format")


def test_pillow_decompression_bomb_guard_is_a_422(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    e = rejected(sh.image_bytes(100.0))
    assert (e.status, e.code) == (422, "too_many_pixels")


def test_config_limits_are_sane():
    assert CFG.max_upload_bytes > 0 and CFG.min_side > 0
    assert CFG.max_pixels >= 200 * 200  # NEU size
    assert set(CFG.allowed_formats) <= {"JPEG", "PNG", "BMP"}


def test_synthetic_near_flat_images_are_rejected():
    for _, png in so.generate(["near_flat"], 10, seed=23):
        assert va.is_near_blank(load_image(png), CFG)


def test_other_synthetic_families_mostly_pass():
    """Non-steel images are the OOD score's job (ADR-027), not the near-blank check's."""
    for fam in ("gaussian_noise", "uniform_noise", "stripes", "blobs"):
        for _, png in so.generate([fam], 10, seed=23):
            assert not va.is_near_blank(load_image(png), CFG), fam


@pytest.mark.needs_data
def test_near_blank_limit_rejects_no_neu_train_or_val_image():
    split_file = REPO_ROOT / PARAMS["data"]["split_file"]
    raw = REPO_ROOT / PARAMS["data"]["raw_dir"]
    with split_file.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in ("train_v1", "val")]
    if not (raw / rows[0]["path"]).is_file():
        pytest.skip("NEU images not on disk")
    stds = []
    for r in rows:
        img = load_image(raw / r["path"])
        stds.append(float(np.asarray(img, dtype=np.float64).std()))
        assert not va.is_near_blank(img, CFG), r["path"]
    # well below, not just below: at least twice the limit
    assert min(stds) >= 2 * CFG.near_blank_max_std
