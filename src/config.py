"""Load `configs/params.yaml`, the single source of seeds, sizes and thresholds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_params(path: Path | str) -> dict[str, Any]:
    """Read a YAML params file and return it as a dict."""
    with Path(path).open(encoding="utf-8") as f:
        params = yaml.safe_load(f)
    if not isinstance(params, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return params
