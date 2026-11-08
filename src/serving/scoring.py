"""Calibrated confidence, energy OOD score and the review routing decision. NumPy only.

One implementation for offline analysis and serving (ADR-023, ADR-025, ADR-027):
  - calibrated probabilities = softmax(logits / T);
  - energy = -T * logsumexp(logits / T) (higher = less familiar input);
  - routed to review if calibrated confidence < review_threshold (reason ``low_confidence``)
    OR energy > ood_energy_threshold (reason ``ood``).
``T`` and both thresholds come from the ``@champion`` version tags.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REASON_LOW_CONFIDENCE = "low_confidence"
REASON_OOD = "ood"


def _scaled(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not temperature > 0:
        raise ValueError("temperature must be > 0")
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("logits must have shape [n, n_classes]")
    return z / temperature


def calibrated_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(logits / T), numerically stable."""
    z = _scaled(logits, temperature)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def energy(logits: np.ndarray, temperature: float) -> np.ndarray:
    """-T * logsumexp(logits / T) per image (Liu et al. 2020)."""
    z = _scaled(logits, temperature)
    m = z.max(axis=1)
    return -temperature * (m + np.log(np.exp(z - m[:, None]).sum(axis=1)))


@dataclass(frozen=True)
class Thresholds:
    temperature: float
    review_threshold: float
    ood_energy_threshold: float


@dataclass(frozen=True)
class Scores:
    pred: np.ndarray  # class index
    confidence: np.ndarray  # top calibrated probability
    energy: np.ndarray
    low_confidence: np.ndarray  # bool
    ood: np.ndarray  # bool

    @property
    def routed(self) -> np.ndarray:
        return self.low_confidence | self.ood

    def reasons(self, i: int) -> list[str]:
        out = []
        if self.low_confidence[i]:
            out.append(REASON_LOW_CONFIDENCE)
        if self.ood[i]:
            out.append(REASON_OOD)
        return out


def score(logits: np.ndarray, th: Thresholds) -> Scores:
    probs = calibrated_probs(logits, th.temperature)
    conf = probs.max(axis=1)
    e = energy(logits, th.temperature)
    return Scores(
        pred=probs.argmax(axis=1),
        confidence=conf,
        energy=e,
        low_confidence=conf < th.review_threshold,
        ood=e > th.ood_energy_threshold,
    )
