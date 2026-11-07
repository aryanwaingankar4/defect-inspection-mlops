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


# ---------------------------------------------------------------- calibration (TASK-012)

CALIBRATION_SCORES = ("ece", "nll", "brier")  # all: lower is better


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax of ``logits / temperature`` (numerically stable)."""
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class (probabilities clipped at 1e-12)."""
    p = np.asarray(probs)[np.arange(len(y_true)), np.asarray(y_true, dtype=np.int64)]
    return float(-np.log(np.clip(p, 1e-12, 1.0)).mean())


def brier(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Multi-class Brier score: mean over images of the squared error to the one-hot label."""
    probs = np.asarray(probs, dtype=np.float64)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), np.asarray(y_true, dtype=np.int64)] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def reliability_bins(probs: np.ndarray, y_true: np.ndarray, n_bins: int) -> dict[str, np.ndarray]:
    """Top-label confidence in ``n_bins`` equal-width bins over (0, 1].

    Returns per bin: ``count``, mean ``confidence`` and ``accuracy`` (NaN for empty bins).
    """
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(y_true)).astype(np.float64)
    # Bin i holds (i/n, (i+1)/n]; confidence 0 cannot happen for a top label.
    idx = np.clip(np.ceil(conf * n_bins).astype(np.int64) - 1, 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.bincount(idx, weights=conf, minlength=n_bins) / count
        acc = np.bincount(idx, weights=correct, minlength=n_bins) / count
    return {"count": count, "confidence": mean_conf, "accuracy": acc}


def ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int) -> float:
    """Expected calibration error: count-weighted mean |accuracy - confidence| over bins."""
    b = reliability_bins(probs, y_true, n_bins)
    used = b["count"] > 0
    gaps = np.abs(b["accuracy"][used] - b["confidence"][used])
    return float((gaps * b["count"][used]).sum() / b["count"].sum())


def calibration_scores(probs: np.ndarray, y_true: np.ndarray, n_bins: int) -> dict[str, float]:
    return {
        "ece": ece(probs, y_true, n_bins),
        "nll": nll(probs, y_true),
        "brier": brier(probs, y_true),
    }


def _nll_at(logits: np.ndarray, y: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
    """Mean NLL of ``softmax(logits / T)`` for each T (vectorised over T)."""
    z = logits[None, :, :] / temperatures[:, None, None]
    zmax = z.max(axis=2, keepdims=True)
    log_norm = np.log(np.exp(z - zmax).sum(axis=2)) + zmax[..., 0]
    true_z = z[:, np.arange(len(y)), y]
    return (log_norm - true_z).mean(axis=1)


def fit_temperature(
    logits: np.ndarray, y_true: np.ndarray, t_min: float, t_max: float, grid_size: int
) -> tuple[float, bool]:
    """T in [t_min, t_max] minimising the mean NLL; also whether it sits on a bound.

    A log-spaced grid finds the best region; golden-section search on log T refines it
    between the neighbouring grid points. NumPy only.
    """
    if not 0 < t_min < 1 < t_max:
        raise ValueError("need 0 < t_min < 1 < t_max")
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    log_grid = np.linspace(np.log(t_min), np.log(t_max), grid_size)
    losses = _nll_at(logits, y, np.exp(log_grid))
    i = int(np.argmin(losses))
    if i in (0, grid_size - 1):
        return float(np.exp(log_grid[i])), True
    lo, hi = log_grid[i - 1], log_grid[i + 1]
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        fa, fb = _nll_at(logits, y, np.exp(np.array([a, b])))
        if fa < fb:
            hi = b
        else:
            lo = a
    return float(np.exp((lo + hi) / 2.0)), False


def stratified_group_folds(
    labels: np.ndarray, unit_ids: np.ndarray, k: int, seed: int
) -> np.ndarray:
    """Fold number (0..k-1) per item. Units (e.g. near-copy groups) stay in one fold; within
    each class the units are shuffled and dealt to the folds in turn."""
    labels, unit_ids = np.asarray(labels), np.asarray(unit_ids)
    rng = np.random.default_rng(seed)
    folds = np.full(len(labels), -1, dtype=np.int64)
    for cls in np.unique(labels):
        units = np.unique(unit_ids[labels == cls])
        for j, u in enumerate(rng.permutation(units)):
            folds[unit_ids == u] = j % k
    if (folds < 0).any():
        raise ValueError("every item needs a unit")
    return folds


def compare_calibration(
    y_true: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    samples: Sequence[np.ndarray],
    n_bins: int,
    level: float,
) -> dict[str, Any]:
    """Scores before and after calibration and the paired change (after - before), each with
    a percentile bootstrap interval over the same resamples, plus a plain verdict per score."""
    y_true = np.asarray(y_true, dtype=np.int64)
    point_b, point_a = (
        calibration_scores(before, y_true, n_bins),
        calibration_scores(after, y_true, n_bins),
    )
    boot = {s: {"before": [], "after": []} for s in CALIBRATION_SCORES}
    for idx in samples:
        sb = calibration_scores(before[idx], y_true[idx], n_bins)
        sa = calibration_scores(after[idx], y_true[idx], n_bins)
        for s in CALIBRATION_SCORES:
            boot[s]["before"].append(sb[s])
            boot[s]["after"].append(sa[s])
    out: dict[str, Any] = {"before": {}, "after": {}, "change": {}, "verdict": {}}
    for s in CALIBRATION_SCORES:
        b, a = np.asarray(boot[s]["before"]), np.asarray(boot[s]["after"])
        for key, point, values in (
            ("before", point_b[s], b),
            ("after", point_a[s], a),
            ("change", point_a[s] - point_b[s], a - b),
        ):
            lo, hi = percentile_interval(values, level)
            out[key][s] = {"value": float(point), "ci_low": float(lo), "ci_high": float(hi)}
        ch = out["change"][s]
        if ch["ci_high"] < 0:
            out["verdict"][s] = "lower after calibration (interval of the change is below 0)"
        elif ch["ci_low"] > 0:
            out["verdict"][s] = "higher after calibration (interval of the change is above 0)"
        else:
            out["verdict"][s] = "no clear change: the interval of the change includes 0"
    return out
