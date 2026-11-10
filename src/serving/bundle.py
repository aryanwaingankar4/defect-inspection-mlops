"""The serving bundle (ADR-028): one ONNX file plus ``manifest.json``, checked before serving.

``make serving-bundle`` (``src/tracking/serving_bundle.py``) builds it from the ``@champion``
version at deploy time; the API only reads it, so it needs no registry access or credentials.
``load_bundle`` refuses a bundle whose file hash, class order, preprocessing or version does not
match. Imports no torch and no mlflow.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.serving.onnx_model import OnnxClassifier
from src.serving.preprocess import PreprocessConfig
from src.serving.scoring import Thresholds

MANIFEST_NAME = "manifest.json"
BUNDLE_FORMAT = 1
PRECISIONS = ("fp32", "int8")
MANIFEST_KEYS = (
    "bundle_format",
    "decision_record",
    "registered_model",
    "version",
    "alias",
    "run_id",
    "precision",
    "onnx_file",
    "onnx_sha256",
    "temperature",
    "review_threshold",
    "ood_energy_threshold",
    "classes",
    "preprocess",
    "built_at",
)


class BundleError(RuntimeError):
    """The bundle is missing, incomplete or does not match itself: the API must not start."""


@dataclass(frozen=True)
class Bundle:
    """A verified bundle, ready to serve."""

    manifest: dict[str, Any]
    model: OnnxClassifier
    classes: list[str]
    thresholds: Thresholds
    preprocess: PreprocessConfig

    @property
    def model_info(self) -> dict[str, str]:
        m = self.manifest
        return {
            "name": m["registered_model"],
            "version": m["version"],
            "alias": m["alias"],
            "precision": m["precision"],
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(bundle_dir: Path) -> dict[str, Any]:
    path = Path(bundle_dir) / MANIFEST_NAME
    if not path.is_file():
        raise BundleError(f"no {MANIFEST_NAME} in {bundle_dir}: run `make serving-bundle` first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in MANIFEST_KEYS if k not in manifest]
    if missing:
        raise BundleError(f"{MANIFEST_NAME} lacks {missing}")
    if manifest["bundle_format"] != BUNDLE_FORMAT:
        raise BundleError(f"unsupported bundle_format {manifest['bundle_format']!r}")
    if manifest["precision"] not in PRECISIONS:
        raise BundleError(f"precision must be one of {PRECISIONS}, got {manifest['precision']!r}")
    return manifest


def thresholds_from(manifest: Mapping[str, Any]) -> Thresholds:
    t = float(manifest["temperature"])
    review = float(manifest["review_threshold"])
    ood = float(manifest["ood_energy_threshold"])
    if not (math.isfinite(t) and t > 0):
        raise BundleError(f"temperature must be positive and finite, got {t}")
    if not 0.0 < review < 1.0:
        raise BundleError(f"review_threshold must be in (0, 1), got {review}")
    if not math.isfinite(ood):
        raise BundleError(f"ood_energy_threshold must be finite, got {ood}")
    return Thresholds(temperature=t, review_threshold=review, ood_energy_threshold=ood)


def check_metadata(model: OnnxClassifier, manifest: Mapping[str, Any]) -> None:
    """The ONNX file's own metadata must agree with the manifest (written at export, ADR-025)."""
    meta = model.metadata
    if model.classes != list(manifest["classes"]):
        raise BundleError("class order in the ONNX metadata differs from the manifest")
    if json.loads(meta.get("preprocess", "null")) != manifest["preprocess"]:
        raise BundleError("preprocessing in the ONNX metadata differs from the manifest")
    if meta.get("registry_version") != str(manifest["version"]):
        raise BundleError("registry version in the ONNX metadata differs from the manifest")
    if meta.get("registered_model") != manifest["registered_model"]:
        raise BundleError("registered model in the ONNX metadata differs from the manifest")
    if meta.get("precision", "fp32") != manifest["precision"]:
        raise BundleError("precision in the ONNX metadata differs from the manifest")


def load_bundle(bundle_dir: Path | str, preprocess_params: Mapping[str, Any]) -> Bundle:
    """Read and verify the bundle; ``preprocess_params`` is ``params['preprocess']``."""
    bundle_dir = Path(bundle_dir)
    manifest = read_manifest(bundle_dir)
    onnx_path = bundle_dir / Path(manifest["onnx_file"]).name
    if not onnx_path.is_file():
        raise BundleError(f"missing {onnx_path}")
    if sha256_file(onnx_path) != manifest["onnx_sha256"]:
        raise BundleError(f"{onnx_path.name} does not match onnx_sha256 in the manifest")
    if dict(preprocess_params) != manifest["preprocess"]:
        raise BundleError("params.yaml preprocess differs from the bundle's (ADR-016)")
    model = OnnxClassifier(onnx_path)
    check_metadata(model, manifest)
    return Bundle(
        manifest=dict(manifest),
        model=model,
        classes=model.classes,
        thresholds=thresholds_from(manifest),
        preprocess=PreprocessConfig.from_params({"preprocess": manifest["preprocess"]}),
    )
