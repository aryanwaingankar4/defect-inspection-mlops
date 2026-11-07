"""Evaluate the trained models on ``test_frozen`` (TASK-011, ADR-022). CPU inference only.

    python -m src.training.evaluate --config configs/params.yaml [--log-mlflow]

Reports only: the model choice was fixed on ``val`` before this runs (ADR-022) and test
results never change it. There is no head-to-head test of the two models here; the
champion/challenger test belongs to the promotion gate (ADR-005).

Before any test image is read, it checks that:
  - the split is clean (committed near-copy report: 0 confirmed near-copies across splits);
  - the ADR-022 comparison exists for this split, is not significant, and lists the models in
    the same order as ``compare_val.models`` with the pre-declared main model first;
  - each checkpoint is the one its results and the ADR-022 report were made with.

Per model, in ``evaluate.out_dir/<name>/``: ``test_metrics.json`` (accuracy, macro-F1,
per-class precision/recall/F1, confusion matrix, each metric with a bootstrap interval),
``test_predictions.csv`` (file names only) and two confusion-matrix plots. The interval is a
cluster bootstrap stratified by class: confirmed near-copies inside ``test_frozen`` (ADR-021)
are resampled together, because overlapping crops are not independent test cases.

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
    confusion_matrix,
    grouped_bootstrap_indices,
    per_class_precision_recall_f1,
    percentile_interval,
)
from src.training.train import git_commit, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
ROLES = ("main", "lighter alternative")  # ADR-022: first model = main, second = alternative
PLOT_DIR = "plots"  # MLflow artifact folder for allowlisted plots (ADR-018)


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
def predict_proba(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """True labels and softmax probabilities, in loader order (CPU)."""
    model.eval()
    labels, probs = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x), dim=1).numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


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


# ---------------------------------------------------------------- outputs


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
    path: Path, paths: Sequence[str], y_true: np.ndarray, probs: np.ndarray, classes: Sequence[str]
) -> None:
    pred = probs.argmax(axis=1)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["path", "true", "pred", "confidence", "correct"])
        for p, t, y, conf in zip(paths, y_true, pred, probs.max(axis=1), strict=True):
            w.writerow([p, classes[t], classes[y], f"{conf:.4f}", int(t == y)])


def write_outputs(
    out_dir: Path,
    report: Mapping[str, Any],
    paths: Sequence[str],
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: Sequence[str],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "metrics": out_dir / "test_metrics.json",
        "predictions": out_dir / "test_predictions.csv",
        "confusion_matrix": out_dir / "confusion_matrix.png",
        "confusion_matrix_normalized": out_dir / "confusion_matrix_normalized.png",
    }
    files["metrics"].write_bytes((json.dumps(report, indent=2) + "\n").encode())
    write_predictions(files["predictions"], paths, y_true, probs, classes)
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
    prov, boot = report["provenance"], report["bootstrap"]
    tags = {
        "test_eval_split_file": report["split_file"],
        "test_eval_role": report["role"],
        "test_eval_git_commit": prov["git_commit"],
        "test_eval_bootstrap": f"{boot['method']}; n={boot['n_resamples']}",
        "test_eval_decision_record": "ADR-022",
    }
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
            (files["metrics"], "eval"),
        ],
    )


# ---------------------------------------------------------------- main


def run(params: Mapping[str, Any], *, log_mlflow: bool = False) -> list[dict[str, Any]]:
    """Check, then evaluate each model (main first). Returns one summary per model."""
    check_split(params)
    specs = check_model_choice(params)
    cfg = EvalConfig.from_params(params)
    split_csv, raw_dir = Path(params["data"]["split_file"]), Path(params["data"]["raw_dir"])
    pre_cfg = PreprocessConfig.from_params(params)
    commit = git_commit(REPO_ROOT)

    summaries: list[dict[str, Any]] = []
    samples, units, groups, ref_labels, classes = None, None, None, None, None
    for spec, role in zip(specs, ROLES, strict=True):
        model, model_classes, timm_name = load_checkpoint(spec.checkpoint)
        if classes is not None and model_classes != classes:
            raise EvalError("the models use different class orders")
        classes = model_classes
        # The only place outside the promotion gate that reads test_frozen (ADR-022 settled).
        ds = SplitDataset(split_csv, raw_dir, [PROTECTED_SPLIT], classes, pre_cfg, allow_test=True)
        y_true, probs = predict_proba(model, DataLoader(ds, batch_size=cfg.batch_size))
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
        files = write_outputs(cfg.out_dir / spec.name, report, ds.paths, y_true, probs, classes)
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
        print(f"  wrote {s['out_dir'].as_posix()}/")
        if "mlflow_run" in s:
            print(f"  MLflow: {s['mlflow_run']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
