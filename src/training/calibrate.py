"""Temperature scaling on ``val`` (TASK-012, ADR-006). CPU inference only; never ``test_frozen``.

    python -m src.training.calibrate --config configs/params.yaml [--log-mlflow]

For each model in ``compare_val.models`` (main model first), one temperature T is fitted on
the val logits by minimising the mean NLL of ``softmax(logits / T)``, within the declared
range ``[calibrate.t_min, calibrate.t_max]``. If the best T sits on a bound, nothing is
written and the run stops (exit 1): a bound fit means the range or the data needs a human look.

Reported on val, with grouped bootstrap intervals (near-copy groups inside val resampled
together) for ECE, NLL, Brier, before and after, and for the paired change:
  - in-sample: T fitted and scored on the same images (optimistic by construction);
  - cross-fitted: T fitted on k-1 group folds and applied to the held-out fold (the honest
    val estimate; the verdict uses this one);
  - an interval for T itself, by refitting T on bootstrap resamples.

Writes into each model's ``results_dir``: ``calibration.json`` (T, its range and interval,
what it was fitted on, the reports), ``val_predictions.csv`` (raw and calibrated confidence;
file names only) and ``reliability_diagram_before/after.png``. ``evaluate.py`` applies T on
``test_frozen`` (report only); TASK-015 applies it in ONNX / serving.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_params
from src.serving.preprocess import PreprocessConfig
from src.training.compare_val import ModelSpec, load_checkpoint, specs_from_params
from src.training.dataset import PROTECTED_SPLIT, SplitDataset
from src.training.evaluate import (
    CALIBRATION_FILE,
    PLOT_DIR,
    EvalConfig,
    calibration_report,
    near_copy_groups_within,
    plot_reliability,
    predict_outputs,
    unit_ids,
    write_predictions,
)
from src.training.metrics import (
    fit_temperature,
    grouped_bootstrap_indices,
    percentile_interval,
    softmax,
    stratified_group_folds,
)
from src.training.train import git_commit, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]


class CalibrationError(RuntimeError):
    """Calibration must not be written (wrong split or checkpoint, T on a range bound)."""


@dataclass(frozen=True)
class CalibConfig:
    t_min: float
    t_max: float
    grid_size: int
    n_bins: int
    cv_folds: int
    seed: int
    n_resamples: int
    t_resamples: int
    level: float
    batch_size: int

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> CalibConfig:
        c, b = params["calibrate"], params["calibrate"]["bootstrap"]
        return cls(
            t_min=float(c["t_min"]),
            t_max=float(c["t_max"]),
            grid_size=int(c["grid_size"]),
            n_bins=int(c["n_bins"]),
            cv_folds=int(c["cv_folds"]),
            seed=int(c["seed"]),
            n_resamples=int(b["n_resamples"]),
            t_resamples=int(b["t_resamples"]),
            level=float(b["level"]),
            batch_size=int(c["batch_size"]),
        )


def fit(logits: np.ndarray, y: np.ndarray, cfg: CalibConfig) -> tuple[float, bool]:
    return fit_temperature(logits, y, cfg.t_min, cfg.t_max, cfg.grid_size)


def cross_fitted_probs(
    logits: np.ndarray, y: np.ndarray, folds: np.ndarray, cfg: CalibConfig
) -> tuple[np.ndarray, list[float], list[bool]]:
    """Out-of-fold calibrated probabilities: each fold uses T fitted on the other folds."""
    out = np.zeros_like(logits, dtype=np.float64)
    temps, bounds = [], []
    for f in range(cfg.cv_folds):
        held = folds == f
        t, at_bound = fit(logits[~held], y[~held], cfg)
        out[held] = softmax(logits[held], t)
        temps.append(t)
        bounds.append(at_bound)
    return out, temps, bounds


def temperature_interval(
    logits: np.ndarray, y: np.ndarray, samples: Sequence[np.ndarray], cfg: CalibConfig
) -> dict[str, Any]:
    """Percentile interval of T refitted on the first ``t_resamples`` bootstrap resamples."""
    fits = [fit(logits[idx], y[idx], cfg) for idx in samples[: cfg.t_resamples]]
    temps = np.array([t for t, _ in fits])
    lo, hi = percentile_interval(temps, cfg.level)
    return {
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_resamples": len(fits),
        "share_on_bound": float(np.mean([b for _, b in fits])),
    }


def check_checkpoint(spec: ModelSpec, root: Path = REPO_ROOT) -> dict[str, Any]:
    recorded = json.loads((root / spec.results_dir / "val_metrics.json").read_text("utf-8"))
    if sha256_file(root / spec.checkpoint) != recorded["checkpoint_sha256"]:
        raise CalibrationError(f"{spec.checkpoint} is not the checkpoint of {spec.results_dir}")
    return recorded


def summary_text(cal: Mapping[str, Any]) -> str:
    cv, ti = cal["val"]["cross_fitted"], cal["temperature_interval"]
    verdicts = "; ".join(f"{s.upper()}: {v}" for s, v in cv["verdict"].items())
    text = (
        f"T = {cal['temperature']:.3f} (95% interval {ti['ci_low']:.3f} to {ti['ci_high']:.3f}). "
        f"Val has {cv['n_errors']} misclassified of {cv['n_images']} images, so the scores rest "
        f"on very few errors. Honest (cross-fitted) change on val: {verdicts}."
    )
    folds_on_bound = sum(cv["fold_at_bound"])
    if folds_on_bound or ti["share_on_bound"] > 0:
        text += (
            f" T is poorly determined: the fit sat on a range bound in "
            f"{ti['share_on_bound']:.0%} of bootstrap resamples and in {folds_on_bound} of "
            f"{len(cv['fold_at_bound'])} cross-fit folds (a resample or fold with no errors "
            "pushes T to the lower bound)."
        )
    return text


INHERITED = "inherited_T"  # ADR-037
FITTED = "fitted"


def choose_temperature(
    fitted_t: float, fit_at_bound: bool, inherit: Mapping[str, Any] | None
) -> tuple[float, str, bool]:
    """(T in use, ``fitted`` | ``inherited_T``, whether the T in use sits on a bound).

    A fit on a range bound inherits the champion's T when ``inherit`` is given (retrain,
    ADR-037); otherwise it stays on the bound and the caller stops (ADR-006)."""
    if fit_at_bound and inherit is not None:
        return float(inherit["temperature"]), INHERITED, False
    return float(fitted_t), FITTED, bool(fit_at_bound)


def calibrate_model(
    spec: ModelSpec,
    params: Mapping[str, Any],
    cfg: CalibConfig,
    val_split: str,
    inherit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit T for one model and build its report (nothing written here).

    ``inherit`` (retrain only, ADR-037): ``{"version", "temperature"}`` of the current champion.
    If the fit sits on a range bound, that validated T is used instead and recorded as such."""
    recorded = check_checkpoint(spec)
    model, classes, timm_name = load_checkpoint(spec.checkpoint)
    split_csv, raw_dir = Path(params["data"]["split_file"]), Path(params["data"]["raw_dir"])
    ds = SplitDataset(
        split_csv, raw_dir, [val_split], classes, PreprocessConfig.from_params(params)
    )
    y, logits, probs = predict_outputs(model, DataLoader(ds, batch_size=cfg.batch_size))

    eval_cfg = EvalConfig.from_params(params)
    groups = near_copy_groups_within(ds.paths, eval_cfg)
    units = unit_ids(ds.paths, groups)
    samples = grouped_bootstrap_indices(y, units, cfg.n_resamples, cfg.seed)
    folds = stratified_group_folds(y, units, cfg.cv_folds, cfg.seed)

    fitted_t, fit_at_bound = fit(logits, y, cfg)
    temperature, source, at_bound = choose_temperature(fitted_t, fit_at_bound, inherit)
    calibrated = softmax(logits, temperature)
    oof, fold_temps, fold_bounds = cross_fitted_probs(logits, y, folds, cfg)
    commit = git_commit(REPO_ROOT)
    cal = {
        "model": spec.name,
        "timm_model": timm_name,
        "decision_record": "ADR-006" if source == FITTED else "ADR-006, ADR-037",
        "temperature": temperature,
        "at_bound": at_bound,
        "calibration": source,
        "fit": {"temperature": fitted_t, "at_bound": fit_at_bound},
        "t_range": [cfg.t_min, cfg.t_max],
        "method": "one temperature minimising mean val NLL (log grid + golden-section search)",
        "apply": "probabilities = softmax(logits / temperature); predicted class unchanged",
        "classes": list(classes),
        "checkpoint_sha256": recorded["checkpoint_sha256"],
        "train_run_uid": recorded.get("run_uid"),
        "fitted_on": {
            "split": val_split,
            "split_file": split_csv.name,
            "split_sha256": sha256_file(split_csv),
            "n_images": int(len(y)),
            "n_errors": int((probs.argmax(axis=1) != y).sum()),
        },
        "temperature_interval": temperature_interval(logits, y, samples, cfg),
        "val": {
            "in_sample": {
                "note": "T fitted and scored on the same images: optimistic by construction",
                **calibration_report(y, probs, calibrated, samples, cfg.n_bins, cfg.level),
            },
            "cross_fitted": {
                "note": (
                    f"{cfg.cv_folds} folds stratified by class with near-copy groups kept "
                    "together; each fold scored with T fitted on the other folds"
                ),
                "fold_temperatures": fold_temps,
                "fold_at_bound": fold_bounds,
                **calibration_report(y, probs, oof, samples, cfg.n_bins, cfg.level),
            },
        },
        "bootstrap": {
            "method": "cluster bootstrap stratified by class, percentile interval",
            "unit": "confirmed near-copy group (ADR-021) or single image",
            "n_units": int(len(np.unique(units))),
            "n_groups": len(groups),
            "n_resamples": cfg.n_resamples,
            "seed": cfg.seed,
            "level": cfg.level,
        },
        "provenance": {
            "git_commit": commit["sha"],
            "git_dirty": commit["dirty"],
            "device": "cpu",
            "torch": torch.__version__,
        },
    }
    if source == INHERITED:
        cal["inherited_from"] = {
            "version": str(inherit["version"]),
            "temperature": temperature,
            "source": f"temperature tag of @champion v{inherit['version']} (its validated T)",
            "why": "the val fit sat on a range bound (e.g. no val errors): val cannot "
            "determine T for this model (ADR-037). The inherited T is NOT validated for it.",
        }
    cal["summary"] = summary_text(cal)
    return {
        "spec": spec,
        "cal": cal,
        "paths": ds.paths,
        "y": y,
        "probs": probs,
        "calibrated": calibrated,
        "classes": classes,
    }


def write_outputs(result: Mapping[str, Any]) -> dict[str, Path]:
    spec, cal = result["spec"], result["cal"]
    out = spec.results_dir
    files = {
        "calibration": out / CALIBRATION_FILE,
        "predictions": out / "val_predictions.csv",
        "reliability_before": out / "reliability_diagram_before.png",
        "reliability_after": out / "reliability_diagram_after.png",
    }
    files["calibration"].write_bytes((json.dumps(cal, indent=2) + "\n").encode())
    write_predictions(
        files["predictions"],
        result["paths"],
        result["y"],
        result["probs"],
        result["calibrated"],
        result["classes"],
    )
    ins = cal["val"]["in_sample"]
    for when in ("before", "after"):
        how = "inherited, ADR-037" if cal.get("calibration") == INHERITED else "fitted here"
        label = "raw softmax" if when == "before" else f"T = {cal['temperature']:.3f} ({how})"
        plot_reliability(
            ins[f"reliability_{when}"],
            ins[when]["ece"]["value"],
            files[f"reliability_{when}"],
            f"{spec.name} on val, {label}",
        )
    return files


def mlflow_payload(cal: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    ins, cv = cal["val"]["in_sample"], cal["val"]["cross_fitted"]
    metrics = {
        "temperature": cal["temperature"],
        "temperature_ci_low": cal["temperature_interval"]["ci_low"],
        "temperature_ci_high": cal["temperature_interval"]["ci_high"],
    }
    for s in ("ece", "nll", "brier"):
        metrics[f"val_{s}_before"] = ins["before"][s]["value"]
        metrics[f"val_{s}_after"] = ins["after"][s]["value"]
        metrics[f"val_{s}_after_cv"] = cv["after"][s]["value"]
        metrics[f"val_{s}_change_cv"] = cv["change"][s]["value"]
    tags = {
        "calibration_split_file": cal["fitted_on"]["split_file"],
        "calibration_git_commit": cal["provenance"]["git_commit"],
        "calibration_decision_record": cal["decision_record"],
        "calibration_source": cal.get("calibration", FITTED),
    }
    return metrics, tags


def log_to_mlflow(params: Mapping[str, Any], spec: ModelSpec, files: Mapping[str, Path]) -> str:
    from src.tracking import mlflow_tracking as mt  # only with --log-mlflow

    pointer = json.loads((spec.results_dir / "mlflow_run.json").read_text(encoding="utf-8"))
    cal = json.loads(files["calibration"].read_text(encoding="utf-8"))
    metrics, tags = mlflow_payload(cal)
    return mt.add_to_finished_run(
        mt.TrackingConfig.from_params(params, smoke=False),
        pointer["run_id"],
        metrics=metrics,
        tags=tags,
        files=[
            (files["reliability_before"], PLOT_DIR),
            (files["reliability_after"], PLOT_DIR),
            (files["calibration"], "calibration"),
        ],
    )


def run(params: Mapping[str, Any], *, log_mlflow: bool = False) -> list[dict[str, Any]]:
    """Fit every model first; write only if no temperature sits on a range bound."""
    val_split = str(params["train"]["val_split"])
    if val_split == PROTECTED_SPLIT:
        raise CalibrationError(f"calibration never uses '{PROTECTED_SPLIT}' (ADR-006)")
    cfg = CalibConfig.from_params(params)
    results = [
        calibrate_model(s, params, cfg, val_split) for s in specs_from_params(params["compare_val"])
    ]
    on_bound = [r["cal"] for r in results if r["cal"]["at_bound"]]
    if on_bound:
        names = ", ".join(f"{c['model']} (T = {c['temperature']:.3f})" for c in on_bound)
        raise CalibrationError(
            f"temperature on the range bound [{cfg.t_min}, {cfg.t_max}] for {names}; "
            "nothing written: check the range or the data with the project owner"
        )
    summaries = []
    for r in results:
        files = write_outputs(r)
        summary = {"model": r["spec"].name, "cal": r["cal"], "out": files["calibration"]}
        if log_mlflow:
            summary["mlflow_run"] = log_to_mlflow(params, r["spec"], files)
        summaries.append(summary)
    return summaries


def _fmt(m: Mapping[str, float]) -> str:
    return f"{m['value']:.4f} [{m['ci_low']:.4f}, {m['ci_high']:.4f}]"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="add T, val calibration metrics and allowlisted plots to each model's MLflow run",
    )
    args = parser.parse_args(argv)
    params = load_params(args.config)
    try:
        summaries = run(params, log_mlflow=args.log_mlflow)
    except CalibrationError as exc:
        print(f"calibrate: {exc}", file=sys.stderr)
        return 1
    for s in summaries:
        cal = s["cal"]
        print(f"{s['model']}: {cal['summary']}")
        for name in ("in_sample", "cross_fitted"):
            rep = cal["val"][name]
            parts = [
                f"{sc} {_fmt(rep['before'][sc])} -> {_fmt(rep['after'][sc])}"
                for sc in ("ece", "nll", "brier")
            ]
            print(f"  val {name}: " + "; ".join(parts))
        print(f"  wrote {s['out'].as_posix()} (+ val_predictions.csv, 2 reliability diagrams)")
        if "mlflow_run" in s:
            print(f"  MLflow: {s['mlflow_run']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
