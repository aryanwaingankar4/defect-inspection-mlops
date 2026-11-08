"""ONNX Runtime classifier for serving (TASK-015, ADR-025). Imports no torch.

The graph takes ``input`` [batch, 3, S, S] float32 (from ``preprocess.to_model_input``) and
returns ``logits`` [batch, n_classes]: RAW logits. Callers apply ``softmax(logits / T)`` and
the review threshold from the ``@champion`` version tags (ADR-024, ADR-025).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort

INPUT_NAME = "input"
OUTPUT_NAME = "logits"


class OnnxClassifier:
    """One CPU inference session plus the metadata written at export time."""

    def __init__(self, path: Path | str, providers: Sequence[str] = ("CPUExecutionProvider",)):
        self.session = ort.InferenceSession(str(path), providers=list(providers))
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        inputs = [i.name for i in self.session.get_inputs()]
        outputs = [o.name for o in self.session.get_outputs()]
        if inputs != [INPUT_NAME] or outputs != [OUTPUT_NAME]:
            raise ValueError(
                f"expected {INPUT_NAME!r} -> {OUTPUT_NAME!r}, got {inputs} -> {outputs}"
            )

    @property
    def classes(self) -> list[str]:
        return list(json.loads(self.metadata["classes"]))

    def logits(self, batch: np.ndarray) -> np.ndarray:
        """Raw logits for a preprocessed batch [n, 3, S, S]."""
        x = np.ascontiguousarray(batch, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError("batch must have shape [n, 3, S, S]")
        return self.session.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
