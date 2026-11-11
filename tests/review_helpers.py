"""Helpers for review tests (TASK-024/025): a tiny fake archive (generated PNGs, no NEU pixels) with
a split CSV, and prediction rows for the log."""

from __future__ import annotations

import csv
import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def png(seed: int) -> bytes:
    pixels = np.random.default_rng(seed).integers(0, 255, (40, 40), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    return buf.getvalue()


def make_archive(
    root: Path, items: list[tuple[str, str, int]]
) -> tuple[Path, Path, dict[str, str]]:
    """``items``: (class, split, seed). Returns (split CSV, raw dir, {name: sha256})."""
    raw = root / "raw"
    split_file = root / "split.csv"
    shas = {}
    rows = []
    for i, (label, split, seed) in enumerate(items):
        rel = f"{label}/{label}_{i}.png"
        path = raw / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        data = png(seed)
        path.write_bytes(data)
        shas[f"{label}_{i}"] = hashlib.sha256(data).hexdigest()
        rows.append({"path": rel, "class": label, "split": split})
    with split_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return split_file, raw, shas


def prediction(request_id: str, sha: str = "f" * 64, *, reasons: tuple[str, ...] = (), **changes):
    """A prediction-log row; routed if any reason is given."""
    row = {
        "request_id": request_id,
        "created_at": datetime(2026, 11, 11, 12, 0, tzinfo=UTC),
        "image_sha256": sha,
        "prediction": "scratches",
        "calibrated_confidence": 0.6,
        "energy": -7.5,
        "routed_to_review": bool(reasons),
        "reason_low_confidence": "low_confidence" in reasons,
        "reason_ood": "ood" in reasons,
        "reason_audit": "audit" in reasons,
        "model_name": "steel-defect-classifier",
        "model_version": "2",
        "model_precision": "fp32",
        "latency_ms": 40.0,
        "inference_ms": 30.0,
        "image_bytes": 1234,
        "image_width": 200,
        "image_height": 200,
        "image_format": "JPEG",
        "brightness": 120.0,
        "contrast": 20.0,
        "sharpness": 300.0,
        "logits": [0.0] * 6,
    }
    row.update(changes)
    return row
