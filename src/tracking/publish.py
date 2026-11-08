"""Publish a finished training run to MLflow from the laptop (ADR-020).

    python -m src.tracking.publish --config configs/params.yaml
        [--run-dir DIR --checkpoint PT | --smoke] [--role candidate|comparison]
        [--register] [--results-dir DIR]

Training (on Kaggle, or a local smoke run) only writes files and holds no credentials. This
module reads those files and creates ONE MLflow run: params, per-epoch metrics (step = epoch),
best-epoch metrics, provenance tags, the allowlisted files (through the artifact policy) and the
model rebuilt from the checkpoint. The run keeps the real training start/end times.

Safety rules:
  - the checkpoint must match the SHA-256 recorded by training;
  - the model is registered only with ``--register``; ``--role comparison`` can never be;
  - a training run (``run_uid``) is published at most once. With ``--register``, an already
    published run gets its existing model registered instead (once).
Defaults read the Kaggle output layout (``kaggle.output_dir`` + ``kaggle.output_results`` /
``kaggle.output_checkpoint``); ``--smoke`` reads ``train.smoke.out_dir``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from src.config import load_params
from src.tracking import mlflow_tracking as mt
from src.training.train import build_model, sha256_file

ROLES = ("candidate", "comparison")
RUN_UID_TAG = "train_run_uid"
# (file in the run folder, MLflow artifact dir). Plots go to plots/ (ADR-018 allowlist).
ARTIFACTS = (
    ("val_metrics.json", None),
    ("params.yaml", "config"),
    ("training_curves.csv", None),
    ("training_curves.png", "plots"),
)
PUBLISH_RECORD = "mlflow_run.json"


class PublishError(RuntimeError):
    """The run cannot be published safely."""


@dataclass(frozen=True)
class TrainedRun:
    run_dir: Path
    checkpoint: Path
    metrics: dict[str, Any]
    params: dict[str, Any]
    curve: list[dict[str, float]]


def read_curve(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def load_run(run_dir: Path, checkpoint: Path) -> TrainedRun:
    """Read a training output folder and check it is complete and matches the checkpoint."""
    missing = [n for n, _ in ARTIFACTS if not (run_dir / n).is_file()]
    if missing or not checkpoint.is_file():
        raise PublishError(f"incomplete run in {run_dir}: missing {missing or [str(checkpoint)]}")
    metrics = json.loads((run_dir / "val_metrics.json").read_text(encoding="utf-8"))
    for key in ("run_uid", "checkpoint_sha256", "timing", "model"):
        if key not in metrics:
            raise PublishError(f"val_metrics.json has no {key!r}: written before ADR-020?")
    if sha256_file(checkpoint) != metrics["checkpoint_sha256"]:
        raise PublishError(f"{checkpoint} is not the checkpoint this run wrote (SHA-256 differs)")
    params = yaml.safe_load((run_dir / "params.yaml").read_text(encoding="utf-8"))
    return TrainedRun(
        run_dir, checkpoint, metrics, params, read_curve(run_dir / "training_curves.csv")
    )


def run_tags(metrics: Mapping[str, Any], role: str) -> dict[str, Any]:
    return {
        **metrics["provenance"],
        RUN_UID_TAG: metrics["run_uid"],
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "model": metrics["model"],
        "role": role,
        "device": metrics["device"],
        "smoke": metrics["smoke"],
        "n_train": metrics["n_train"],
        "n_val": metrics["n_val"],
        "n_params": metrics["n_params"],
        "published_from": "laptop",
    }


def best_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        "best_epoch": metrics["best_epoch"],
        "best_val_macro_f1": metrics["val_macro_f1"],
        "best_val_accuracy": metrics["val_accuracy"],
        "best_val_loss": metrics["val_loss"],
        **{f"best_val_f1.{c}": f for c, f in metrics["val_per_class_f1"].items()},
    }


def model_from_checkpoint(path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild the trained network on CPU (no pretrained download) and its metadata."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    meta = {k: v for k, v in ckpt.items() if k not in ("state_dict", "format_version")}
    meta["checkpoint_format"] = ckpt["format_version"]
    return model.eval(), meta


def _log_new_run(run: TrainedRun, cfg: mt.TrackingConfig, role: str, register: bool) -> dict:
    m = run.metrics
    timing = m["timing"]
    tracker = mt.RunTracker(
        cfg, start_time_ms=timing["started_at_ms"], end_time_ms=timing["finished_at_ms"]
    )
    with tracker:
        tracker.log_params({k: run.params[k] for k in ("data", "preprocess", "train")})
        tracker.set_tags(run_tags(m, role))
        for row in run.curve:
            values = {k: v for k, v in row.items() if k != "epoch"}
            tracker.log_metrics(values, step=int(row["epoch"]))
        tracker.log_metrics(best_metrics(m))
        for name, artifact_dir in ARTIFACTS:
            tracker.log_file(run.run_dir / name, artifact_dir)

        model, meta = model_from_checkpoint(run.checkpoint)
        size = int(meta["preprocess"]["image_size"])
        prov = m["provenance"]
        version = tracker.log_model(
            model,
            input_shape=(3, size, size),
            n_outputs=len(meta["classes"]),
            metadata=meta,
            pip_requirements=[f"torch=={prov['torch'].split('+')[0]}", f"timm=={prov['timm']}"],
            register=register,
        )
    return {"run_id": tracker.run_id, "run_url": tracker.run_url, "registered_version": version}


def publish(
    params: Mapping[str, Any],
    run_dir: Path,
    checkpoint: Path,
    *,
    role: str = "candidate",
    register: bool = False,
) -> dict[str, Any]:
    """Publish one training run. Returns what was done (written to ``mlflow_run.json``)."""
    if role not in ROLES:
        raise PublishError(f"role must be one of {ROLES}")
    if register and role == "comparison":
        raise PublishError("comparison runs are never registered")
    run = load_run(run_dir, checkpoint)
    cfg = mt.TrackingConfig.from_params(params, smoke=bool(run.metrics["smoke"]))

    existing = mt.find_finished_run(cfg, RUN_UID_TAG, run.metrics["run_uid"])
    if existing is None:
        info = _log_new_run(run, cfg, role, register)
    elif register:
        info = {"run_id": existing, "registered_version": mt.register_run_model(cfg, existing)}
    else:
        raise PublishError(
            f"already published as MLflow run {existing}; pass --register to register its model"
        )
    record = {
        "experiment": cfg.experiment,
        "role": role,
        **info,
        "registered_model": cfg.registered_model if info["registered_version"] else None,
    }
    (run_dir / PUBLISH_RECORD).write_bytes((json.dumps(record, indent=2) + "\n").encode())
    return record


def copy_results(run_dir: Path, results_dir: Path) -> list[Path]:
    """Copy the committed result files (no checkpoint) into ``results/...``."""
    results_dir.mkdir(parents=True, exist_ok=True)
    names = [n for n, _ in ARTIFACTS] + [PUBLISH_RECORD]
    return [Path(shutil.copy2(run_dir / n, results_dir / n)) for n in names]


def default_paths(params: Mapping[str, Any], smoke: bool) -> tuple[Path, Path]:
    if smoke:
        smoke_dir = Path(params["train"]["smoke"]["out_dir"])
        return smoke_dir / "results", smoke_dir / "best.pt"
    k = params["kaggle"]
    out = Path(k["output_dir"])
    return out / k["output_results"], out / k["output_checkpoint"]


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        # MLflow prints an emoji when a run ends; a cp1252 Windows console would crash on it.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, help="folder with val_metrics.json etc.")
    parser.add_argument("--checkpoint", type=Path, help="the run's best.pt")
    parser.add_argument("--smoke", action="store_true", help="publish the local smoke run")
    parser.add_argument("--role", choices=ROLES, default="candidate")
    parser.add_argument("--register", action="store_true", help="also register the model")
    parser.add_argument("--results-dir", type=Path, help="copy result files here (results/...)")
    args = parser.parse_args(argv)

    params = load_params(args.config)
    run_dir, checkpoint = default_paths(params, args.smoke)
    run_dir, checkpoint = args.run_dir or run_dir, args.checkpoint or checkpoint
    try:
        record = publish(params, run_dir, checkpoint, role=args.role, register=args.register)
    except PublishError as exc:
        print(f"not published: {exc}", file=sys.stderr)
        return 1
    print(f"MLflow run {record['run_id']} in {record['experiment']} (role {record['role']})")
    if record.get("run_url"):
        print(f"  {record['run_url']}")
    if record["registered_version"]:
        print(f"  registered {record['registered_model']} version {record['registered_version']}")
    else:
        print("  model logged, not registered")
    if args.results_dir:
        copied = copy_results(run_dir, args.results_dir)
        print(f"  copied {len(copied)} result files to {args.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
