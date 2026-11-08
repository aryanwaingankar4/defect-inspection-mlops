"""Review threshold on ``val`` from a fixed review budget (TASK-013, ADR-023). CPU only.

    python -m src.training.threshold --config configs/params.yaml [--log-mlflow]

Main model only (``train.model``, first in ``compare_val.models``); never ``test_frozen``.
Confidence = top-class calibrated confidence, ``softmax(logits / T)`` with T from the model's
``calibration.json`` (checked: same checkpoint, same split).

Rule (ADR-023, numbers in ``params.yaml`` -> ``threshold``):
  - route to review the ``ceil(review_budget * n)`` least confident val images; the threshold
    is the midpoint between the highest routed and the lowest accepted confidence (labels
    play no part in choosing it);
  - risk = error rate among the accepted images; its one-sided Clopper-Pearson upper bound at
    ``ucb_level`` must be <= ``max_risk_ucb``, otherwise nothing is written and the run stops
    (exit 1) for the project owner to decide.

Writes into the main model's ``results_dir``: ``threshold.json`` (threshold, rule, hashes of
split / checkpoint / calibration, routing and risk on val, the full risk-coverage curve with
grouped bootstrap bands, and what this can and cannot show) and ``risk_coverage.png``
(allowlisted). ``evaluate.py`` applies the threshold on ``test_frozen`` (report only).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
from src.training.calibrate import CalibrationError, check_checkpoint
from src.training.compare_val import ModelSpec, load_checkpoint, specs_from_params
from src.training.dataset import PROTECTED_SPLIT, SplitDataset
from src.training.evaluate import (
    PLOT_DIR,
    EvalConfig,
    EvalError,
    load_temperature,
    near_copy_groups_within,
    predict_outputs,
    review_report,
    unit_ids,
)
from src.training.metrics import (
    budget_threshold,
    clopper_pearson_upper,
    grouped_bootstrap_indices,
    percentile_interval,
    risk_coverage_curve,
    risk_curve_bootstrap,
    softmax,
    top_confidence,
)
from src.training.train import git_commit, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_NAME = "risk_coverage.png"  # allowlisted (ADR-018)
CSV_TOLERANCE = 6e-5  # calibrate's val_predictions.csv rounds confidences to 4 decimals


class ThresholdError(RuntimeError):
    """The threshold must not be written (wrong split, model or calibration; risk check)."""


@dataclass(frozen=True)
class ThresholdConfig:
    review_budget: float
    max_risk_ucb: float
    ucb_level: float
    reference_risk: float
    min_coverage: float
    coverage_step: float
    batch_size: int
    n_resamples: int
    seed: int
    level: float
    out_file: str

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> ThresholdConfig:
        t, b = params["threshold"], params["threshold"]["bootstrap"]
        return cls(
            review_budget=float(t["review_budget"]),
            max_risk_ucb=float(t["max_risk_ucb"]),
            ucb_level=float(t["ucb_level"]),
            reference_risk=float(t["reference_risk"]),
            min_coverage=float(t["min_coverage"]),
            coverage_step=float(t["coverage_step"]),
            batch_size=int(t["batch_size"]),
            n_resamples=int(b["n_resamples"]),
            seed=int(b["seed"]),
            level=float(b["level"]),
            out_file=str(t["out_file"]),
        )

    def coverage_grid(self) -> np.ndarray:
        steps = round((1.0 - self.min_coverage) / self.coverage_step)
        return np.round(np.linspace(self.min_coverage, 1.0, steps + 1), 10)


def main_spec(params: Mapping[str, Any]) -> ModelSpec:
    spec = specs_from_params(params["compare_val"])[0]
    if spec.name != params["train"]["model"]:
        raise ThresholdError("compare_val.models must list the main model (train.model) first")
    return spec


def images_needed(reference_risk: float, level: float) -> int:
    """Accepted images with zero errors needed for the upper bound to reach ``reference_risk``."""
    return math.ceil(math.log(1.0 - level) / math.log(1.0 - reference_risk))


def check_against_calibration_csv(
    spec: ModelSpec, paths: Sequence[str], pred: Sequence[str], conf: np.ndarray
) -> None:
    """Inference here must reproduce calibrate's val predictions (same images, classes, T)."""
    with (spec.results_dir / "val_predictions.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if [r["path"] for r in rows] != list(paths) or [r["pred"] for r in rows] != list(pred):
        raise ThresholdError("val predictions differ from calibrate's val_predictions.csv")
    stored = np.array([float(r["calibrated_confidence"]) for r in rows])
    if np.max(np.abs(stored - conf)) > CSV_TOLERANCE:
        raise ThresholdError("calibrated confidences differ from calibrate's val_predictions.csv")


def claims(val: Mapping[str, Any], cfg: ThresholdConfig, n_groups: int) -> dict[str, list[str]]:
    """Plain statements of what the val evidence shows and does not show."""
    ucb, ucb_g = val["risk_ucb"]["value"], val["risk_ucb_by_group"]["value"]
    best_case = clopper_pearson_upper(0, val["n_accepted"], cfg.ucb_level)
    return {
        "can_claim": [
            f"At threshold {val['threshold']:.4f}, {val['n_routed']} of {val['n_images']} val "
            f"images ({val['routed_share']:.1%}) go to review; {val['errors_routed']} of the "
            f"{val['n_errors']} val errors are among them; the other {val['errors_accepted']} "
            f"{'is' if val['errors_accepted'] == 1 else 'are'} auto-accepted.",
            f"With {cfg.ucb_level:.0%} confidence (one-sided Clopper-Pearson), the error rate "
            f"among auto-accepted images like these is at most {ucb:.1%} (observed "
            f"{val['risk']['value']:.1%}: {val['errors_accepted']} of {val['n_accepted']}).",
        ],
        "cannot_claim": [
            f"A risk of {cfg.reference_risk:.0%} or lower: with {val['n_accepted']} accepted "
            f"images even zero errors would only show {best_case:.1%}; about "
            f"{images_needed(cfg.reference_risk, cfg.ucb_level)} accepted images without an "
            "error would be needed.",
            "That this threshold is better than nearby ones: the rule fixes the review budget "
            f"and does not optimise risk, and with {val['n_errors']} val errors the bootstrap "
            "bands of the curve are wide.",
            f"That val images are independent: {n_groups} near-copy groups sit inside val; "
            f"counting each group once, the upper bound is {ucb_g:.1%}.",
            "Anything about images unlike val: new defect types, other lines or drift (OOD "
            "score, TASK-017), or the errors that actually ship (random audit sample, ADR-007).",
            "Unbiased val numbers: T, the threshold and early stopping all used val. "
            "test_frozen reports the same quantities at this threshold (evaluate.py, report "
            "only; it never changes the threshold).",
        ],
    }


def plot_risk_coverage(result: Mapping[str, Any], path: Path, max_risk_ucb: float) -> None:
    """Risk vs coverage with its bootstrap band and the chosen point. No images."""
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve, bands, val = result["curve"], result["bands"], result["val"]
    lo_cov = bands["coverage"][0]
    cov, risk = np.array(curve["coverage"]), np.array(curve["risk"])
    keep = cov >= lo_cov - 1e-9
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(
        bands["coverage"],
        bands["ci_low"],
        bands["ci_high"],
        color="tab:blue",
        alpha=0.2,
        label=f"{bands['level']:.0%} bootstrap band",
    )
    ax.plot(cov[keep], risk[keep], "-", color="tab:blue", label="val risk-coverage curve")
    c, r, u = val["coverage"]["value"], val["risk"]["value"], val["risk_ucb"]["value"]
    ax.plot([c, c], [r, u], color="black", lw=1.5)
    ax.plot(c, u, "v", color="black", label=f"{val['risk_ucb']['level']:.0%} upper bound")
    ax.plot(c, r, "o", color="tab:orange", ms=8, label=f"chosen: {val['n_routed']} routed")
    ax.axhline(max_risk_ucb, ls="--", color="tab:red", label=f"max_risk_ucb = {max_risk_ucb}")
    ax.set(
        xlim=(lo_cov, 1.0),
        ylim=(0, max(max_risk_ucb, u, float(np.max(bands["ci_high"]))) * 1.15),
        xlabel="coverage (share auto-accepted)",
        ylabel="risk (error rate among accepted)",
        title=(
            f"{result['model']} on val (n={val['n_images']}, {val['n_errors']} errors)\n"
            f"threshold {val['threshold']:.4f} on calibrated confidence"
        ),
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def build(params: Mapping[str, Any], cfg: ThresholdConfig) -> dict[str, Any]:
    """Set the threshold on val and build its evidence (nothing written here)."""
    val_split = str(params["train"]["val_split"])
    if val_split == PROTECTED_SPLIT:
        raise ThresholdError(f"the review threshold never uses '{PROTECTED_SPLIT}' (ADR-023)")
    spec = main_spec(params)
    try:
        recorded = check_checkpoint(spec)
    except CalibrationError as exc:
        raise ThresholdError(str(exc)) from exc
    temperature, cal_path = load_temperature(params, spec)
    model, classes, timm_name = load_checkpoint(spec.checkpoint)
    split_csv, raw_dir = Path(params["data"]["split_file"]), Path(params["data"]["raw_dir"])
    ds = SplitDataset(
        split_csv, raw_dir, [val_split], classes, PreprocessConfig.from_params(params)
    )
    y, logits, _ = predict_outputs(model, DataLoader(ds, batch_size=cfg.batch_size))
    pred = logits.argmax(axis=1)
    conf = top_confidence(softmax(logits, temperature))
    check_against_calibration_csv(spec, ds.paths, [classes[i] for i in pred], conf)

    groups = near_copy_groups_within(ds.paths, EvalConfig.from_params(params))
    units = unit_ids(ds.paths, groups)
    samples = grouped_bootstrap_indices(y, units, cfg.n_resamples, cfg.seed)
    threshold, _ = budget_threshold(conf, cfg.review_budget)
    val = review_report(y, pred, conf, threshold, units, samples, cfg.ucb_level, cfg.level)

    correct = pred == y
    curve = risk_coverage_curve(conf, correct)
    grid = cfg.coverage_grid()
    band_lo, band_hi = percentile_interval(
        risk_curve_bootstrap(conf, correct, grid, samples), cfg.level
    )
    commit = git_commit(REPO_ROOT)
    result: dict[str, Any] = {
        "model": spec.name,
        "timm_model": timm_name,
        "decision_record": "ADR-023",
        "threshold": threshold,
        "applies_to": "top-class calibrated confidence, max of softmax(logits / temperature)",
        "route_rule": "routed to review if calibrated confidence < threshold",
        "rule": {
            "review_budget": cfg.review_budget,
            "max_risk_ucb": cfg.max_risk_ucb,
            "ucb_level": cfg.ucb_level,
            "choice": (
                "ceil(review_budget * n) least confident val images are routed; the threshold "
                "is the midpoint between the highest routed and the lowest accepted confidence"
            ),
        },
        "passed": bool(val["risk_ucb"]["value"] <= cfg.max_risk_ucb),
        "temperature": temperature,
        "calibration_file": cal_path.relative_to(REPO_ROOT).as_posix(),
        "calibration_sha256": sha256_file(cal_path),
        "checkpoint_sha256": recorded["checkpoint_sha256"],
        "train_run_uid": recorded.get("run_uid"),
        "fitted_on": {
            "split": val_split,
            "split_file": split_csv.name,
            "split_sha256": sha256_file(split_csv),
            "n_images": int(len(y)),
            "n_errors": int((~correct).sum()),
        },
        "val": val,
        "curve": {
            "note": "every threshold on val: accept images with confidence >= threshold",
            **{k: v.tolist() for k, v in curve.items()},
        },
        "bands": {
            "note": "risk when the ceil(coverage * n) most confident images are accepted",
            "level": cfg.level,
            "coverage": grid.tolist(),
            "risk": risk_curve_bootstrap(conf, correct, grid, [np.arange(len(y))])[0].tolist(),
            "ci_low": band_lo.tolist(),
            "ci_high": band_hi.tolist(),
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
        "claims": claims(val, cfg, len(groups)),
        "provenance": {
            "git_commit": commit["sha"],
            "git_dirty": commit["dirty"],
            "device": "cpu",
            "torch": torch.__version__,
        },
    }
    return {"spec": spec, "result": result}


def write_outputs(spec: ModelSpec, result: Mapping[str, Any], cfg: ThresholdConfig) -> dict:
    files = {"threshold": spec.results_dir / cfg.out_file, "plot": spec.results_dir / PLOT_NAME}
    files["threshold"].write_bytes((json.dumps(result, indent=2) + "\n").encode())
    plot_risk_coverage(result, files["plot"], cfg.max_risk_ucb)
    return files


def mlflow_payload(result: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    val = result["val"]
    metrics = {
        "review_threshold": result["threshold"],
        "val_review_routed_share": val["routed_share"],
        "val_review_coverage": val["coverage"]["value"],
        "val_review_risk": val["risk"]["value"],
        "val_review_risk_ci_low": val["risk"]["ci_low"],
        "val_review_risk_ci_high": val["risk"]["ci_high"],
        "val_review_risk_ucb": val["risk_ucb"]["value"],
        "val_review_risk_ucb_by_group": val["risk_ucb_by_group"]["value"],
        "val_review_errors_routed": float(val["errors_routed"]),
        "val_review_errors_accepted": float(val["errors_accepted"]),
    }
    tags = {
        "threshold_decision_record": "ADR-023",
        "threshold_split_file": result["fitted_on"]["split_file"],
        "threshold_git_commit": result["provenance"]["git_commit"],
        "threshold_review_budget": str(result["rule"]["review_budget"]),
    }
    return metrics, tags


def log_to_mlflow(params: Mapping[str, Any], spec: ModelSpec, files: Mapping[str, Path]) -> str:
    from src.tracking import mlflow_tracking as mt  # only with --log-mlflow

    pointer = json.loads((spec.results_dir / "mlflow_run.json").read_text(encoding="utf-8"))
    result = json.loads(files["threshold"].read_text(encoding="utf-8"))
    metrics, tags = mlflow_payload(result)
    return mt.add_to_finished_run(
        mt.TrackingConfig.from_params(params, smoke=False),
        pointer["run_id"],
        metrics=metrics,
        tags=tags,
        files=[(files["plot"], PLOT_DIR), (files["threshold"], "threshold")],
    )


def run(params: Mapping[str, Any], *, log_mlflow: bool = False) -> dict[str, Any]:
    cfg = ThresholdConfig.from_params(params)
    built = build(params, cfg)
    spec, result = built["spec"], built["result"]
    if not result["passed"]:
        ucb = result["val"]["risk_ucb"]["value"]
        raise ThresholdError(
            f"risk upper bound {ucb:.4f} among accepted val images is above max_risk_ucb "
            f"{cfg.max_risk_ucb}; nothing written: the project owner decides (ADR-023)"
        )
    files = write_outputs(spec, result, cfg)
    summary = {"result": result, "out": files["threshold"]}
    if log_mlflow:
        summary["mlflow_run"] = log_to_mlflow(params, spec, files)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="add the threshold, val routing metrics and the plot to the main model's MLflow run",
    )
    args = parser.parse_args(argv)
    params = load_params(args.config)
    try:
        summary = run(params, log_mlflow=args.log_mlflow)
    except (ThresholdError, EvalError) as exc:
        print(f"threshold: {exc}", file=sys.stderr)
        return 1
    r = summary["result"]
    for line in r["claims"]["can_claim"]:
        print(f"  {line}")
    print("  Cannot be claimed from val:")
    for line in r["claims"]["cannot_claim"]:
        print(f"    - {line}")
    print(f"  wrote {summary['out'].as_posix()} (+ {PLOT_NAME})")
    if "mlflow_run" in summary:
        print(f"  MLflow: {summary['mlflow_run']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
