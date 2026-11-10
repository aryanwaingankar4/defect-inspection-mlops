"""Golden images (TASK-020, ADR-017): a synthetic image made by a formula, and ``make golden``.

The synthetic image uses integer arithmetic only (no random generator, no dataset pixels), so
the same 200x200 greyscale image comes out everywhere; a test regenerates it and compares it
with the committed PNG pixel by pixel.

``make golden`` writes ``tests/golden/synthetic_golden.png`` and ``tests/golden/expected.json``:
the preprocessed-tensor summary of the synthetic image (checked in CI), and the ``@champion``
bundle's outputs on the synthetic image and two fixed ``val`` images (checked with
``needs_model`` / ``needs_data``). It refuses to write unless the two NEU results agree with the
PyTorch-based ``val_predictions.csv``. Never reads ``test_frozen``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.config import load_params
from src.serving.bundle import Bundle, load_bundle
from src.serving.preprocess import PreprocessConfig, load_image, to_model_input
from src.serving.validation import ValidationConfig, decode_checked

SIZE = 200  # NEU image side
SYNTHETIC_NAME = "synthetic_golden.png"
EXPECTED_NAME = "expected.json"
CSV_TOLERANCE = 6e-5  # calibrate's val_predictions.csv rounds confidences to 4 decimals
# fixed (channel, row, column) points of the preprocessed tensor recorded in expected.json
SAMPLE_POINTS = [(c, y, x) for c in range(3) for y, x in [(0, 0), (37, 151), (112, 112), (223, 5)]]


class GoldenError(RuntimeError):
    pass


def synthetic_pixels(size: int = SIZE) -> np.ndarray:
    """Stripes, a scratch-like line, a dark disc and integer pseudo-noise; uint8 [size, size]."""
    y, x = np.mgrid[0:size, 0:size].astype(np.int64)
    img = np.full((size, size), 110, dtype=np.int64)
    img += ((x + 2 * y) // 12 % 2) * 40  # diagonal stripes
    img += np.where(np.abs(x - y - 20) < 3, 60, 0)  # a thin bright line
    img -= np.where((x - 140) ** 2 + (y - 60) ** 2 < 900, 50, 0)  # a dark disc
    img += (x * 73 + y * 151 + x * y * 7) % 31 - 15  # deterministic texture
    return np.clip(img, 0, 255).astype(np.uint8)


def synthetic_png() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(synthetic_pixels()).save(buf, format="PNG")
    return buf.getvalue()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_summary(data: bytes, pre: PreprocessConfig) -> dict[str, Any]:
    """Shape, per-channel mean/std and fixed sample values of the shared preprocessing output."""
    x = to_model_input(load_image(data), pre)
    return {
        "shape": list(x.shape),
        "channel_mean": [float(v) for v in x.mean(axis=(1, 2))],
        "channel_std": [float(v) for v in x.std(axis=(1, 2))],
        "samples": {f"{c},{y},{x_}": float(x[c, y, x_]) for c, y, x_ in SAMPLE_POINTS},
    }


def model_output(bundle: Bundle, data: bytes, vcfg: ValidationConfig) -> dict[str, Any]:
    """The serving path (checks, shared preprocessing, ONNX, shared scoring) on one image."""
    from src.serving.app import predict_one  # the exact function /predict uses

    result, _, logits = predict_one(bundle, decode_checked(data, vcfg).image)
    return {**result, "logits": logits}


def val_images(params: Mapping[str, Any]) -> list[dict[str, str]]:
    """The first ``val`` image in the split file and the ``val`` image with the lowest calibrated
    confidence in ``val_predictions.csv`` (ADR-017 note). Never ``test_frozen``."""
    split = str(params["train"]["val_split"])
    if split == "test_frozen":
        raise GoldenError("golden images never come from test_frozen")
    with Path(params["data"]["split_file"]).open(encoding="utf-8", newline="") as f:
        first = next(r["path"] for r in csv.DictReader(f) if r["split"] == split)
    with csv_path(params).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lowest = min(rows, key=lambda r: float(r["calibrated_confidence"]))["path"]
    return [{"id": "val_first", "path": first}, {"id": "val_lowest_confidence", "path": lowest}]


def csv_path(params: Mapping[str, Any]) -> Path:
    return Path(params["compare_val"]["models"][0]["results_dir"]) / "val_predictions.csv"


def check_against_csv(params: Mapping[str, Any], path: str, out: Mapping[str, Any]) -> None:
    with csv_path(params).open(encoding="utf-8", newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["path"] == path)
    if out["prediction"] != row["pred"]:
        raise GoldenError(
            f"{path}: serving says {out['prediction']}, val_predictions.csv says {row['pred']}"
        )
    if abs(out["calibrated_confidence"] - float(row["calibrated_confidence"])) > CSV_TOLERANCE:
        raise GoldenError(f"{path}: calibrated confidence differs from val_predictions.csv")


def build_expected(params: Mapping[str, Any], bundle_dir: Path, png: bytes) -> dict[str, Any]:
    pre = PreprocessConfig.from_params(params)
    vcfg = ValidationConfig.from_params(params)
    bundle = load_bundle(bundle_dir, params["preprocess"])
    m = bundle.manifest
    raw = Path(params["data"]["raw_dir"])
    images = [{"id": "synthetic", "source": SYNTHETIC_NAME, "data": png}]
    images += [
        {"id": v["id"], "source": v["path"], "data": (raw / v["path"]).read_bytes()}
        for v in val_images(params)
    ]
    entries = []
    for img in images:
        out = model_output(bundle, img["data"], vcfg)
        if img["id"] != "synthetic":
            check_against_csv(params, img["source"], out)
        entries.append(
            {
                "id": img["id"],
                "source": img["source"],
                "image_sha256": sha256_bytes(img["data"]),
                **out,
            }
        )
    parity = params["export"]["parity"]
    return {
        "decision_record": "ADR-017",
        "synthetic_image": {
            "file": SYNTHETIC_NAME,
            "pixels_sha256": sha256_bytes(synthetic_pixels().tobytes()),
            "tensor": tensor_summary(png, pre),
        },
        "tolerance": {"atol": float(parity["atol"]), "rtol": float(parity["rtol"])},
        "champion": {
            "registered_model": m["registered_model"],
            "version": m["version"],
            "precision": m["precision"],
            "onnx_sha256": m["onnx_sha256"],
            "temperature": m["temperature"],
            "review_threshold": m["review_threshold"],
            "ood_energy_threshold": m["ood_energy_threshold"],
            "images": entries,
        },
    }


def write(params: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    png = synthetic_png()
    expected = build_expected(params, Path(params["serving"]["bundle_dir"]), png)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SYNTHETIC_NAME).write_bytes(png)
    (out_dir / EXPECTED_NAME).write_bytes((json.dumps(expected, indent=2) + "\n").encode())
    return expected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/params.yaml")
    parser.add_argument("--out-dir", default="tests/golden")
    args = parser.parse_args(argv)
    e = write(load_params(args.config), Path(args.out_dir))
    c = e["champion"]
    print(f"golden -> {args.out_dir}/ ({SYNTHETIC_NAME}, {EXPECTED_NAME}; ADR-017)")
    print(
        f"  model {c['registered_model']} v{c['version']} ({c['precision']}), "
        f"onnx {c['onnx_sha256'][:12]}..."
    )
    for img in c["images"]:
        print(
            f"  {img['id']:22} {img['prediction']:15} cal {img['calibrated_confidence']:.4f} "
            f"energy {img['energy']:8.3f} reasons {img['reasons']}"
        )
    print("  NEU results agree with val_predictions.csv")


if __name__ == "__main__":
    main()
