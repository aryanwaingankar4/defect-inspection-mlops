"""Classification metrics in NumPy (no scikit-learn). Shared by training and evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Counts with rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    both = np.concatenate([y_true, y_pred])
    if both.size and (both.min() < 0 or both.max() >= n_classes):
        raise ValueError(f"labels must be in [0, {n_classes})")
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """F1 per class. A class with no true and no predicted samples gets F1 = 0."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(axis=0) + cm.sum(axis=1)  # 2*TP + FP + FN
    return np.divide(2 * tp, denom, out=np.zeros(n_classes), where=denom > 0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of per-class F1."""
    return float(per_class_f1(y_true, y_pred, n_classes).mean())


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    return float((y_true == np.asarray(y_pred)).mean()) if y_true.size else 0.0


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts.

    ``a_only`` = items only model A got right, ``b_only`` = items only model B got right.
    Under the null hypothesis each discordant item is a fair coin flip, so the p-value is the
    two-sided binomial test of ``min(a_only, b_only)`` in ``a_only + b_only`` trials at 0.5.
    No discordant items gives p = 1.
    """
    if a_only < 0 or b_only < 0:
        raise ValueError("counts must be non-negative")
    n, k = a_only + b_only, min(a_only, b_only)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def per_class_precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall and F1 per class. Undefined values (no predicted / no true samples)
    are 0, as in ``per_class_f1``."""
    return _prf_from_cm(confusion_matrix(y_true, y_pred, n_classes))


def _prf_from_cm(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    pred_n, true_n = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(tp, pred_n, out=np.zeros(k), where=pred_n > 0)
    recall = np.divide(tp, true_n, out=np.zeros(k), where=true_n > 0)
    f1 = np.divide(2 * tp, pred_n + true_n, out=np.zeros(k), where=(pred_n + true_n) > 0)
    return precision, recall, f1


def grouped_bootstrap_indices(
    labels: np.ndarray, unit_ids: np.ndarray, n_resamples: int, seed: int
) -> list[np.ndarray]:
    """Item indices of ``n_resamples`` cluster bootstrap samples, stratified by class.

    Items with the same ``unit_ids`` value (e.g. a group of overlapping crops) are drawn
    together. Within each class, as many units as the class has are drawn with replacement,
    so every class keeps its number of units. Every unit must hold one class only.
    """
    labels, unit_ids = np.asarray(labels), np.asarray(unit_ids)
    if labels.shape != unit_ids.shape:
        raise ValueError("labels and unit_ids must have the same shape")
    members: dict[Any, list[int]] = {}
    for i, u in enumerate(unit_ids.tolist()):
        members.setdefault(u, []).append(i)
    units_by_class: dict[int, list[np.ndarray]] = {}
    for idx in members.values():
        classes = set(labels[idx].tolist())
        if len(classes) != 1:
            raise ValueError(f"a unit spans several classes: {sorted(classes)}")
        units_by_class.setdefault(classes.pop(), []).append(np.array(idx, dtype=np.int64))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples):
        parts = []
        for cls in sorted(units_by_class):
            units = units_by_class[cls]
            parts.extend(units[j] for j in rng.integers(0, len(units), size=len(units)))
        samples.append(np.concatenate(parts))
    return samples


def bootstrap_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, samples: Sequence[np.ndarray]
) -> dict[str, np.ndarray]:
    """Accuracy, macro-F1 and per-class precision/recall/F1 for each bootstrap sample.

    Returns arrays of shape (n_samples,) for scalars and (n_samples, n_classes) per class.
    """
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    out: dict[str, list] = {k: [] for k in ("accuracy", "macro_f1", "precision", "recall", "f1")}
    for idx in samples:
        t, p = y_true[idx], y_pred[idx]
        cm = np.bincount(t * n_classes + p, minlength=n_classes**2).reshape(n_classes, n_classes)
        precision, recall, f1 = _prf_from_cm(cm)
        out["accuracy"].append(np.trace(cm) / cm.sum())
        out["macro_f1"].append(f1.mean())
        out["precision"].append(precision)
        out["recall"].append(recall)
        out["f1"].append(f1)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def percentile_interval(values: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided percentile interval along axis 0 (e.g. level 0.95 -> 2.5th and 97.5th)."""
    tail = (1.0 - level) / 2.0 * 100.0
    return np.percentile(values, tail, axis=0), np.percentile(values, 100.0 - tail, axis=0)
