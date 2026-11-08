"""Evaluate the trained models on ``test_frozen`` (TASK-011, ADR-022). CPU inference only.

    python -m src.training.evaluate --config configs/params.yaml [--log-mlflow]

Reports only: the model choice was fixed on ``val`` before this runs (ADR-022) and test
results never change it. There is no head-to-head test of the two models here; the
champion/challenger test belongs to the promotion gate (ADR-005).

Before any test image is read, it checks that:
  - the split is clean (committed near-copy report: 0 confirmed near-copies across splits);
  - the ADR-022 comparison exists for this split, is not significant, and lists the models in
    the same order as ``compare_val.models`` with the pre-declared main model first;
  - each checkpoint is the one its results and the ADR-022 report were made with;
  - each model has a calibration fitted on ``val`` of this split with this checkpoint;
  - the main model has a review threshold set on ``val`` with that calibration (ADR-023).

Per model, in ``evaluate.out_dir/<name>/``: ``test_metrics.json`` (accuracy, macro-F1,
per-class precision/recall/F1, confusion matrix, each metric with a bootstrap interval),
``test_predictions.csv`` (file names only) and two confusion-matrix plots. The interval is a
cluster bootstrap stratified by class: confirmed near-copies inside ``test_frozen`` (ADR-021)
are resampled together, because overlapping crops are not independent test cases.

Calibration (TASK-012, ADR-006): the temperature T fitted on ``val`` by ``calibrate.py`` is
read from ``<results_dir>/calibration.json`` and only APPLIED here (never fitted on test). The
report adds ECE / NLL / Brier before and after T with bootstrap intervals and two reliability
diagrams; the predictions CSV gets ``calibrated_confidence``. Predicted classes do not change.

Review threshold (TASK-013, ADR-023), main model only: the threshold set on ``val`` by
``threshold.py`` is read from ``<results_dir>/threshold.json`` and only APPLIED here. The report
adds a ``review`` block (routed share, risk among accepted images with its Clopper-Pearson
bound and bootstrap interval, test errors caught by review); the predictions CSV gets
``routed_to_review``. Test results never change the threshold.

``--log-mlflow`` adds the test metrics, tags and allowlisted files to each model's existing
MLflow run (from ``mlflow_run.json``); without it nothing talks to MLflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import load_params
from src.data.groups import confirmed, connected_groups, read_overlaps
from src.pipelines.kaggle_job import LaunchError, check_split_is_clean
from src.serving.preprocess import PreprocessConfig
from src.training.compare_val import ModelSpec, load_checkpoint, specs_from_params
from src.training.dataset import PROTECTED_SPLIT, SplitDataset
from src.training.metrics import (
    bootstrap_metrics,
    clopper_pearson_upper,
    compare_calibration,
    confusion_matrix,
    grouped_bootstrap_indices,
    per_class_precision_recall_f1,
    percentile_interval,
    reliability_bins,
    softmax,
    threshold_bootstrap,
    top_confidence,
)
from src.training.train import git_commit, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
ROLES = ("main", "lighter alternative")  # ADR-022: first model = main, second = alternative
PLOT_DIR = "plots"  # MLflow artifact folder for allowlisted plots (ADR-018)
CALIBRATION_FILE = "calibration.json"  # in each model's results_dir, by calibrate.py


class EvalError(RuntimeError):
    """Evaluation must not run (unsettled model choice, unclean split, wrong checkpoint)."""


@dataclass(frozen=True)
class EvalConfig:
    batch_size: int
    n_resamples: int
    seed: int
    level: float
    out_dir: Path
    pairs_file: Path
    min_pixel_corr: float
    n_bins: int

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> EvalConfig:
        e, b = params["evaluate"], params["evaluate"]["bootstrap"]
        nd = params["near_duplicates_cnn"]
        return cls(
            batch_size=int(e["batch_size"]),
            n_resamples=int(b["n_resamples"]),
            seed=int(b["seed"]),
            level=float(b["level"]),
            out_dir=Path(e["out_dir"]),
            pairs_file=Path(nd["pairs_file"]),
            min_pixel_corr=float(nd["confirm_pixel_corr"]),
            n_bins=int(params["calibrate"]["n_bins"]),
        )


# ---------------------------------------------------------------- checks (before test is read)


def check_model_choice(params: Mapping[str, Any], root: Path = REPO_ROOT) -> list[ModelSpec]:
    """The ADR-022 choice must be settled on val for THIS split; returns the specs, main first."""
    specs = specs_from_params(params["compare_val"])
    if specs[0].name != params["train"]["model"]:
        raise EvalError("compare_val.models must list the pre-declared main model first")
    path = root / params["compare_val"]["out"]
    if not path.is_file():
        raise EvalError(f"no ADR-022 comparison {path.name}: run `make compare-val` first")
    report = json.loads(path.read_text(encoding="utf-8"))
    split_csv = root / params["data"]["split_file"]
    if report.get("split") != "val" or report.get("split_file") != split_csv.name:
        raise EvalError("the ADR-022 comparison is not for val of the current split")
    if report.get("split_sha256") != sha256_file(split_csv):
        raise EvalError("the split file changed after the ADR-022 comparison")
    if report.get("significant") is not False:
        raise EvalError("ADR-022: the val difference is significant; the owner must decide first")
    if [m["name"] for m in report["models"]] != [s.name for s in specs]:
        raise EvalError("the ADR-022 comparison lists other models or another order")
    for spec, m in zip(specs, report["models"], strict=True):
        recorded = json.loads((root / spec.results_dir / "val_metrics.json").read_text("utf-8"))
        actual = sha256_file(root / spec.checkpoint)
        if not (actual == m["checkpoint_sha256"] == recorded["checkpoint_sha256"]):
            raise EvalError(f"{spec.checkpoint} is not the checkpoint compared under ADR-022")
    return specs


def load_temperature(
    params: Mapping[str, Any], spec: ModelSpec, root: Path = REPO_ROOT
) -> tuple[float, Path]:
    """T of ``spec``, fitted on val of the current split with this checkpoint (TASK-012)."""
    path = root / spec.results_dir / CALIBRATION_FILE
    if not path.is_file():
        raise EvalError(f"no calibration for {spec.name}: run `make calibrate` first")
    cal = json.loads(path.read_text(encoding="utf-8"))
    fitted, split_csv = cal.get("fitted_on", {}), root / params["data"]["split_file"]
    if fitted.get("split") != "val" or fitted.get("split_file") != split_csv.name:
        raise EvalError(f"the calibration of {spec.name} was not fitted on val of this split")
    if fitted.get("split_sha256") != sha256_file(split_csv):
        raise EvalError(f"the split file changed after {spec.name} was calibrated")
    if cal.get("checkpoint_sha256") != sha256_file(root / spec.checkpoint):
        raise EvalError(f"the calibration of {spec.name} is for another checkpoint")
    if cal.get("at_bound") is not False:
        raise EvalError(f"the temperature of {spec.name} sits on its range bound")
    return float(cal["temperature"]), path


def load_threshold(
    params: Mapping[str, Any], spec: ModelSpec, calibration: Path, root: Path = REPO_ROOT
) -> tuple[float, Path]:
    """Review threshold of the main model, set on val of this split with this checkpoint and
    this calibration under the current rule (ADR-023)."""
    rule = params["threshold"]
    path = root / spec.results_dir / rule["out_file"]
    if not path.is_file():
        raise EvalError(f"no review threshold for {spec.name}: run `make threshold` first")
    thr = json.loads(path.read_text(encoding="utf-8"))
    fitted, split_csv = thr.get("fitted_on", {}), root / params["data"]["split_file"]
    if fitted.get("split") != "val" or fitted.get("split_file") != split_csv.name:
        raise EvalError(f"the threshold of {spec.name} was not set on val of this split")
    if fitted.get("split_sha256") != sha256_file(split_csv):
        raise EvalError(f"the split file changed after the threshold of {spec.name} was set")
    if thr.get("checkpoint_sha256") != sha256_file(root / spec.checkpoint):
        raise EvalError(f"the threshold of {spec.name} is for another checkpoint")
    if thr.get("calibration_sha256") != sha256_file(calibration):
        raise EvalError(f"the threshold of {spec.name} was set with another calibration")
    stored = thr.get("rule", {})
    for key in ("review_budget", "max_risk_ucb", "ucb_level"):
        if stored.get(key) != rule[key]:
            raise EvalError(f"threshold.{key} changed since the threshold was set")
    if thr.get("passed") is not True:
        raise EvalError(f"the threshold of {spec.name} did not pass its risk check")
    return float(thr["threshold"]), path


def check_split(params: Mapping[str, Any], root: Path = REPO_ROOT) -> None:
    try:
        check_split_is_clean(params, root)
    except LaunchError as exc:
        raise EvalError(str(exc)) from exc


# ---------------------------------------------------------------- data + inference


def unit_ids(paths: Sequence[str], groups: Sequence[Sequence[str]]) -> np.ndarray:
    """One id per image: images in the same near-copy group share an id, others get their own."""
    in_set = set(paths)
    group_of = {}
    for g, members in enumerate(groups):
        for p in members:
            if p in in_set:
                group_of[p] = g
    next_id = len(groups)
    ids = []
    for p in paths:
        if p in group_of:
            ids.append(group_of[p])
        else:
            ids.append(next_id)
            next_id += 1
    return np.array(ids, dtype=np.int64)


def near_copy_groups_within(paths: Sequence[str], cfg: EvalConfig) -> list[list[str]]:
    """Confirmed near-copy groups restricted to ``paths`` (edges inside the set only)."""
    in_set = set(paths)
    edges = [
        (o.path_a, o.path_b)
        for o in confirmed(read_overlaps(cfg.pairs_file), cfg.min_pixel_corr)
        if o.path_a in in_set and o.path_b in in_set
    ]
    return connected_groups(edges)


@torch.inference_mode()
def predict_outputs(
    model: nn.Module, loader: DataLoader
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True labels, logits and softmax probabilities (T = 1), in loader order (CPU)."""
    model.eval()
    labels, logits, probs = [], [], []
    for x, y in loader:
        out = model(x)
        logits.append(out.numpy())
        probs.append(torch.softmax(out, dim=1).numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(logits), np.concatenate(probs)


# ---------------------------------------------------------------- metrics


def _with_ci(value: float, low: float, high: float) -> dict[str, float]:
    return {"value": float(value), "ci_low": float(low), "ci_high": float(high)}


def metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
    samples: Sequence[np.ndarray],
    level: float,
) -> dict[str, Any]:
    """Point metrics with percentile bootstrap intervals over the given resamples."""
    k = len(classes)
    cm = confusion_matrix(y_true, y_pred, k)
    precision, recall, f1 = per_class_precision_recall_f1(y_true, y_pred, k)
    boot = bootstrap_metrics(y_true, y_pred, k, samples)
    lo, hi = {}, {}
    for name, values in boot.items():
        lo[name], hi[name] = percentile_interval(values, level)
    per_class = {
        c: {
            "support": int(cm[i].sum()),
            "precision": _with_ci(precision[i], lo["precision"][i], hi["precision"][i]),
            "recall": _with_ci(recall[i], lo["recall"][i], hi["recall"][i]),
            "f1": _with_ci(f1[i], lo["f1"][i], hi["f1"][i]),
        }
        for i, c in enumerate(classes)
    }
    return {
        "accuracy": _with_ci(np.trace(cm) / cm.sum(), lo["accuracy"], hi["accuracy"]),
        "macro_f1": _with_ci(f1.mean(), lo["macro_f1"], hi["macro_f1"]),
        "n_errors": int(cm.sum() - np.trace(cm)),
        "per_class": per_class,
        "confusion_matrix": {"rows": "true class", "columns": "predicted", "counts": cm.tolist()},
    }


def reliability_json(probs: np.ndarray, y_true: np.ndarray, n_bins: int) -> dict[str, Any]:
    """Reliability-diagram data as JSON (empty bins: null)."""
    b = reliability_bins(probs, y_true, n_bins)

    def clean(a: np.ndarray) -> list[float | None]:
        return [None if np.isnan(v) else float(v) for v in a]

    return {
        "n_bins": n_bins,
        "count": b["count"].tolist(),
        "confidence": clean(b["confidence"]),
        "accuracy": clean(b["accuracy"]),
    }


def calibration_report(
    y_true: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    samples: Sequence[np.ndarray],
    n_bins: int,
    level: float,
) -> dict[str, Any]:
    """ECE / NLL / Brier before and after T, the paired change with intervals, verdicts and
    the reliability-diagram data. Says how many errors the scores rest on."""
    if not np.array_equal(before.argmax(axis=1), after.argmax(axis=1)):
        raise EvalError("temperature scaling changed a predicted class")  # impossible for T > 0
    n_errors = int((before.argmax(axis=1) != y_true).sum())
    return {
        "n_images": int(len(y_true)),
        "n_errors": n_errors,
        **compare_calibration(y_true, before, after, samples, n_bins, level),
        "reliability_before": reliability_json(before, y_true, n_bins),
        "reliability_after": reliability_json(after, y_true, n_bins),
        "caution": (
            f"Only {n_errors} of {len(y_true)} images are misclassified, so NLL and ECE rest "
            "mostly on those few images; treat small changes as uncertain."
        ),
    }


def review_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    units: np.ndarray,
    samples: Sequence[np.ndarray],
    ucb_level: float,
    level: float,
) -> dict[str, Any]:
    """Routing at a fixed threshold: who goes to review, and risk among the auto-accepted
    images with a one-sided Clopper-Pearson bound and a grouped bootstrap interval (ADR-023)."""
    correct = y_pred == y_true
    accepted = confidence >= threshold
    n, n_acc = len(y_true), int(accepted.sum())
    k = int((~correct & accepted).sum())
    boot = threshold_bootstrap(confidence, correct, threshold, samples)
    cov_lo, cov_hi = percentile_interval(boot["coverage"], level)
    risk_lo, risk_hi = percentile_interval(boot["risk"], level)
    units_acc = np.unique(units[accepted])
    units_err = np.unique(units[accepted & ~correct])
    return {
        "threshold": float(threshold),
        "route_rule": "routed to review if calibrated confidence < threshold",
        "n_images": int(n),
        "n_routed": int(n - n_acc),
        "n_accepted": n_acc,
        "routed_share": float((n - n_acc) / n),
        "coverage": _with_ci(n_acc / n, cov_lo, cov_hi),
        "n_errors": int((~correct).sum()),
        "errors_routed": int((~correct & ~accepted).sum()),
        "errors_accepted": k,
        "risk": _with_ci(k / n_acc if n_acc else 0.0, risk_lo, risk_hi),
        "risk_ucb": {
            "value": clopper_pearson_upper(k, n_acc, ucb_level),
            "level": ucb_level,
            "method": "one-sided Clopper-Pearson on images (assumes independent images)",
        },
        "risk_ucb_by_group": {
            "value": clopper_pearson_upper(len(units_err), len(units_acc), ucb_level),
            "n_units_accepted": int(len(units_acc)),
            "n_units_with_error": int(len(units_err)),
            "note": "each near-copy group counted once; sensitivity check, not the stop check",
        },
    }


# ---------------------------------------------------------------- outputs


def plot_reliability(bins: Mapping[str, Any], ece_value: float, path: Path, title: str) -> None:
    """Accuracy vs mean confidence per bin (top) and images per bin (bottom). No images."""
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = int(bins["n_bins"])
    edges = np.linspace(0.0, 1.0, n + 1)
    centers, width = (edges[:-1] + edges[1:]) / 2, 1.0 / n
    count = np.array(bins["count"])
    acc = np.array([np.nan if v is None else v for v in bins["accuracy"]], dtype=float)
    conf = np.array([np.nan if v is None else v for v in bins["confidence"]], dtype=float)
    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(6, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    used = count > 0
    ax.bar(centers[used], acc[used], width=width, edgecolor="black", label="accuracy in bin")
    ax.plot(centers[used], conf[used], "o", color="tab:orange", label="mean confidence in bin")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.set(ylabel="accuracy", ylim=(0, 1.02), title=f"{title}\nECE = {ece_value:.4f}")
    ax.legend(loc="upper left", fontsize=8)
    ax_n.bar(centers[used], count[used], width=width, edgecolor="black", color="gray")
    ax_n.set(xlabel="confidence (top class)", ylabel="images", yscale="log", xlim=(0, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, classes: Sequence[str], path: Path, title: str) -> None:
    """Counts (or row-normalised rates if ``cm`` is float) as a labelled grid. No images."""
    os.environ["MPLBACKEND"] = "Agg"  # see train.write_curve: Jupyter may export another backend
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalized = np.issubdtype(cm.dtype, np.floating)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalized else max(1, cm.max()))
    ticks = range(len(classes))
    ax.set(xticks=ticks, yticks=ticks, xlabel="predicted", ylabel="true", title=title)
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_yticklabels(classes)
    threshold = (1 if normalized else cm.max()) / 2
    for i in ticks:
        for j in ticks:
            text = f"{cm[i, j]:.2f}" if normalized else str(cm[i, j])
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_predictions(
    path: Path,
    paths: Sequence[str],
    y_true: np.ndarray,
    probs: np.ndarray,
    calibrated: np.ndarray,
    classes: Sequence[str],
    routed: np.ndarray | None = None,
) -> None:
    """One row per image (file name only): raw and temperature-scaled top-class confidence,
    and ``routed_to_review`` (0/1) when a review threshold applies."""
    pred = probs.argmax(axis=1)
    rows = zip(paths, y_true, pred, probs.max(axis=1), calibrated.max(axis=1), strict=True)
    header = ["path", "true", "pred", "confidence", "calibrated_confidence", "correct"]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header + (["routed_to_review"] if routed is not None else []))
        for i, (p, t, y, conf, cal) in enumerate(rows):
            row = [p, classes[t], classes[y], f"{conf:.4f}", f"{cal:.4f}", int(t == y)]
            w.writerow(row + ([int(routed[i])] if routed is not None else []))


def write_outputs(
    out_dir: Path,
    report: Mapping[str, Any],
    paths: Sequence[str],
    y_true: np.ndarray,
    probs: np.ndarray,
    calibrated: np.ndarray,
    classes: Sequence[str],
    routed: np.ndarray | None = None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "metrics": out_dir / "test_metrics.json",
        "predictions": out_dir / "test_predictions.csv",
        "confusion_matrix": out_dir / "confusion_matrix.png",
        "confusion_matrix_normalized": out_dir / "confusion_matrix_normalized.png",
        "reliability_before": out_dir / "reliability_diagram_test_before.png",
        "reliability_after": out_dir / "reliability_diagram_test_after.png",
    }
    files["metrics"].write_bytes((json.dumps(report, indent=2) + "\n").encode())
    write_predictions(files["predictions"], paths, y_true, probs, calibrated, classes, routed)
    cal = report["calibration"]
    for when in ("before", "after"):
        label = "raw softmax" if when == "before" else f"T = {cal['temperature']:.3f}"
        plot_reliability(
            cal[f"reliability_{when}"],
            cal[when]["ece"]["value"],
            files[f"reliability_{when}"],
            f"{report['model']} on {PROTECTED_SPLIT}, {label}",
        )
    cm = np.array(report["confusion_matrix"]["counts"], dtype=np.int64)
    n = report["n_images"]
    title = f"{report['model']} on {PROTECTED_SPLIT} (n={n})"
    plot_confusion(cm, classes, files["confusion_matrix"], title)
    rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plot_confusion(rates, classes, files["confusion_matrix_normalized"], f"{title}, row-normalised")
    return files


def mlflow_payload(report: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    """Flat ``test_*`` metrics and ``test_eval_*`` tags for the model's MLflow run."""
    metrics: dict[str, float] = {}
    for name in ("accuracy", "macro_f1"):
        m = report[name]
        metrics[f"test_{name}"] = m["value"]
        metrics[f"test_{name}_ci_low"] = m["ci_low"]
        metrics[f"test_{name}_ci_high"] = m["ci_high"]
    for cls, m in report["per_class"].items():
        for name in ("precision", "recall", "f1"):
            metrics[f"test_{name}.{cls}"] = m[name]["value"]
    cal = report["calibration"]
    metrics["test_temperature_applied"] = cal["temperature"]
    for score in ("ece", "nll", "brier"):
        for when in ("before", "after", "change"):
            metrics[f"test_{score}_{when}"] = cal[when][score]["value"]
    review = report.get("review")
    if review is not None:
        metrics["test_review_threshold"] = review["threshold"]
        metrics["test_review_routed_share"] = review["routed_share"]
        for name in ("coverage", "risk"):
            metrics[f"test_review_{name}"] = review[name]["value"]
            metrics[f"test_review_{name}_ci_low"] = review[name]["ci_low"]
            metrics[f"test_review_{name}_ci_high"] = review[name]["ci_high"]
        metrics["test_review_risk_ucb"] = review["risk_ucb"]["value"]
        metrics["test_review_errors_routed"] = float(review["errors_routed"])
        metrics["test_review_errors_accepted"] = float(review["errors_accepted"])
    prov, boot = report["provenance"], report["bootstrap"]
    tags = {
        "test_eval_split_file": report["split_file"],
        "test_eval_role": report["role"],
        "test_eval_git_commit": prov["git_commit"],
        "test_eval_bootstrap": f"{boot['method']}; n={boot['n_resamples']}",
        "test_eval_decision_record": "ADR-022",
    }
    if review is not None:
        tags["test_eval_threshold_record"] = "ADR-023"
    return metrics, tags


def log_to_mlflow(params: Mapping[str, Any], spec: ModelSpec, files: Mapping[str, Path]) -> str:
    from src.tracking import mlflow_tracking as mt  # only with --log-mlflow

    pointer = json.loads((spec.results_dir / "mlflow_run.json").read_text(encoding="utf-8"))
    report = json.loads(files["metrics"].read_text(encoding="utf-8"))
    metrics, tags = mlflow_payload(report)
    return mt.add_to_finished_run(
        mt.TrackingConfig.from_params(params, smoke=False),
        pointer["run_id"],
        metrics=metrics,
        tags=tags,
        files=[
            (files["confusion_matrix"], PLOT_DIR),
            (files["confusion_matrix_normalized"], PLOT_DIR),
            (files["reliability_before"], PLOT_DIR),
            (files["reliability_after"], PLOT_DIR),
            (files["metrics"], "eval"),
        ],
    )


# ---------------------------------------------------------------- main


def run(params: Mapping[str, Any], *, log_mlflow: bool = False) -> list[dict[str, Any]]:
    """Check, then evaluate each model (main first). Returns one summary per model."""
    check_split(params)
    specs = check_model_choice(params)
    temperatures = [load_temperature(params, spec) for spec in specs]
    threshold, thr_path = load_threshold(params, specs[0], temperatures[0][1])  # main only
    ucb_level = float(params["threshold"]["ucb_level"])
    cfg = EvalConfig.from_params(params)
    split_csv, raw_dir = Path(params["data"]["split_file"]), Path(params["data"]["raw_dir"])
    pre_cfg = PreprocessConfig.from_params(params)
    commit = git_commit(REPO_ROOT)

    summaries: list[dict[str, Any]] = []
    samples, units, groups, ref_labels, classes = None, None, None, None, None
    for spec, role, (temperature, cal_path) in zip(specs, ROLES, temperatures, strict=True):
        model, model_classes, timm_name = load_checkpoint(spec.checkpoint)
        if classes is not None and model_classes != classes:
            raise EvalError("the models use different class orders")
        classes = model_classes
        # The only place outside the promotion gate that reads test_frozen (ADR-022 settled).
        ds = SplitDataset(split_csv, raw_dir, [PROTECTED_SPLIT], classes, pre_cfg, allow_test=True)
        y_true, logits, probs = predict_outputs(model, DataLoader(ds, batch_size=cfg.batch_size))
        calibrated = softmax(logits, temperature)
        if samples is None:
            groups = near_copy_groups_within(ds.paths, cfg)
            units = unit_ids(ds.paths, groups)
            samples = grouped_bootstrap_indices(y_true, units, cfg.n_resamples, cfg.seed)
            ref_labels = y_true
        elif not np.array_equal(y_true, ref_labels):
            raise EvalError("the models did not see the same images in the same order")
        assert units is not None and groups is not None

        recorded = json.loads((spec.results_dir / "val_metrics.json").read_text("utf-8"))
        report = {
            "model": spec.name,
            "role": role,
            "decision_record": "ADR-022",
            "timm_model": timm_name,
            "split": PROTECTED_SPLIT,
            "split_file": split_csv.name,
            "split_sha256": sha256_file(split_csv),
            "n_images": int(len(y_true)),
            "classes": list(classes),
            **metrics_report(y_true, probs.argmax(axis=1), classes, samples, cfg.level),
            "bootstrap": {
                "method": "cluster bootstrap stratified by class, percentile interval",
                "unit": "confirmed near-copy group (ADR-021) or single image",
                "n_units": int(len(np.unique(units))),
                "n_groups": len(groups),
                "images_in_groups": sum(len(g) for g in groups),
                "n_resamples": cfg.n_resamples,
                "seed": cfg.seed,
                "level": cfg.level,
                "note": (
                    "A metric with no errors in the test set has the interval [1, 1]: "
                    "the bootstrap cannot show uncertainty it never sees."
                ),
            },
            "calibration": {
                "temperature": temperature,
                "source": cal_path.relative_to(REPO_ROOT).as_posix(),
                "applied_only": "T was fitted on val (calibrate.py); nothing is fitted on test",
                **calibration_report(y_true, probs, calibrated, samples, cfg.n_bins, cfg.level),
            },
            "provenance": {
                "git_commit": commit["sha"],
                "git_dirty": commit["dirty"],
                "checkpoint_sha256": recorded["checkpoint_sha256"],
                "train_run_uid": recorded.get("run_uid"),
                "results_dir": spec.results_dir.as_posix(),
                "device": "cpu",
                "torch": torch.__version__,
                "timm": timm.__version__,
            },
        }
        routed = None
        if role == ROLES[0]:  # the review threshold exists for the main model only (ADR-023)
            conf = top_confidence(calibrated)
            routed = conf < threshold
            review = review_report(
                y_true, probs.argmax(axis=1), conf, threshold, units, samples, ucb_level, cfg.level
            )
            report["review"] = {
                "source": thr_path.relative_to(REPO_ROOT).as_posix(),
                "decision_record": "ADR-023",
                "applied_only": "threshold set on val (threshold.py); nothing is set on test",
                **review,
            }
            report["provenance"] = report.pop("provenance")  # keep provenance last
        files = write_outputs(
            cfg.out_dir / spec.name, report, ds.paths, y_true, probs, calibrated, classes, routed
        )
        summary = {"model": spec.name, "role": role, "out_dir": (cfg.out_dir / spec.name)}
        if log_mlflow:
            summary["mlflow_run"] = log_to_mlflow(params, spec, files)
        summaries.append({**summary, "report": report})
    return summaries


def _fmt(m: Mapping[str, float]) -> str:
    return f"{m['value']:.4f} [{m['ci_low']:.4f}, {m['ci_high']:.4f}]"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="add test metrics and allowlisted plots to each model's existing MLflow run",
    )
    args = parser.parse_args(argv)
    params = load_params(args.config)
    try:
        summaries = run(params, log_mlflow=args.log_mlflow)
    except EvalError as exc:
        print(f"evaluate: {exc}", file=sys.stderr)
        return 1
    for s in summaries:
        r = s["report"]
        print(
            f"{s['model']} ({s['role']}): accuracy {_fmt(r['accuracy'])}, "
            f"macro-F1 {_fmt(r['macro_f1'])}, errors {r['n_errors']}/{r['n_images']}"
        )
        cal = r["calibration"]
        print(f"  calibration (T = {cal['temperature']:.4f} from val, applied only):")
        for score in ("ece", "nll", "brier"):
            print(
                f"    {score}: before {_fmt(cal['before'][score])} -> after "
                f"{_fmt(cal['after'][score])}; {cal['verdict'][score]}"
            )
        if "review" in r:
            rv = r["review"]
            print(
                f"  review (threshold {rv['threshold']:.4f} from val, applied only): routed "
                f"{rv['n_routed']}/{rv['n_images']} ({rv['routed_share']:.1%}); errors routed "
                f"{rv['errors_routed']}/{rv['n_errors']}; risk among accepted "
                f"{_fmt(rv['risk'])}, {rv['risk_ucb']['level']:.0%} upper bound "
                f"{rv['risk_ucb']['value']:.4f}"
            )
        print(f"  wrote {s['out_dir'].as_posix()}/")
        if "mlflow_run" in s:
            print(f"  MLflow: {s['mlflow_run']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
