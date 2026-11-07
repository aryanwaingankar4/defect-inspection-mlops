"""Classification metrics in NumPy (no scikit-learn). Shared by training and evaluation."""

from __future__ import annotations

import math

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
