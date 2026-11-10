"""A tiny hand-built ONNX classifier and bundle for serving tests (no torch, no NEU data).

The graph is ``GlobalAveragePool -> Flatten -> Gemm``: the logits depend only on the mean
brightness, so tests can make confident images (dark or bright) and unsure ones (mid grey).
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
SLOPES = [8.0, 4.0, 0.0, -4.0, -8.0, 0.0]  # logit_k = SLOPES[k] * normalised mean of channel 0
TEMPERATURE = 2.0
REVIEW = 0.8
OOD = -5.0


def tiny_onnx(
    path: Path,
    preprocess: Mapping[str, Any],
    *,
    classes: list[str] = CLASSES,
    version: str = "7",
    registered_model: str = "steel-defect-classifier",
    precision: str | None = None,
) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    size = int(preprocess["image_size"])
    w = np.zeros((3, len(classes)), dtype=np.float32)
    w[0] = SLOPES[: len(classes)]
    nodes = [
        helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "W", "B"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, size, size])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(classes)])],
        [
            numpy_helper.from_array(w, "W"),
            numpy_helper.from_array(np.zeros(len(classes), np.float32), "B"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    meta = {
        "registered_model": registered_model,
        "registry_version": version,
        "classes": json.dumps(classes),
        "preprocess": json.dumps(dict(preprocess)),
    }
    if precision:
        meta["precision"] = precision
    helper.set_model_props(model, meta)
    onnx.checker.check_model(model, full_check=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_bundle(bundle_dir: Path, preprocess: Mapping[str, Any], **manifest_changes: Any) -> Path:
    """A valid bundle (tiny model + manifest); ``manifest_changes`` override manifest fields."""
    onnx_path = tiny_onnx(bundle_dir / "steel-defect-classifier-v7.fp32.onnx", preprocess)
    manifest = {
        "bundle_format": 1,
        "decision_record": "ADR-028",
        "registered_model": "steel-defect-classifier",
        "version": "7",
        "alias": "champion",
        "run_id": "r" * 32,
        "precision": "fp32",
        "onnx_file": onnx_path.name,
        "onnx_sha256": sha(onnx_path),
        "onnx_artifact": f"onnx/{onnx_path.name}",
        "checkpoint_sha256": "c" * 64,
        "temperature": TEMPERATURE,
        "review_threshold": REVIEW,
        "ood_energy_threshold": OOD,
        "classes": CLASSES,
        "preprocess": json.loads(json.dumps(dict(preprocess))),
        "built_at": "2026-11-08T00:00:00+00:00",
        **manifest_changes,
    }
    (bundle_dir / "manifest.json").write_bytes(json.dumps(manifest, indent=2).encode())
    return bundle_dir


def image_bytes(
    mean: float, *, std: float = 20.0, size: int = 200, fmt: str = "PNG", seed: int = 0
) -> bytes:
    """A noisy grey image (mean and std in 0-255 grey levels), encoded as ``fmt``."""
    rng = np.random.default_rng(seed)
    pixels = rng.normal(mean, std, (size, size)).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format=fmt)
    return buf.getvalue()


def without_audit(params: Mapping[str, Any]) -> dict:
    """Params with the random audit off, for tests that check exact routing (ADR-030)."""
    p = copy.deepcopy(dict(params))
    p["serving"]["audit_rate"] = 0.0
    return p
