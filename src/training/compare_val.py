"""Paired McNemar test of two trained models on ``val`` (ADR-022). CPU inference only.

    python -m src.training.compare_val --config configs/params.yaml

Both checkpoints predict the same images of ``compare_val.split``. Before testing, each
model's predictions must reproduce the val accuracy and macro-F1 recorded in its
``val_metrics.json``, and its checkpoint must match the recorded SHA-256. Writes
``compare_val.out`` (JSON: per-model metrics, the 2x2 table, the exact p-value, the
discordant images by file name, and the ADR-022 outcome). ``test_frozen`` is refused:
model choice never looks at the frozen test set.
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
from torch import nn
from torch.utils.data import DataLoader

from src.config import load_params
from src.serving.preprocess import PreprocessConfig
from src.training.dataset import PROTECTED_SPLIT, SplitDataset
from src.training.metrics import accuracy, macro_f1, mcnemar_exact
from src.training.train import build_model, predict, sha256_file


class CompareError(RuntimeError):
    """The comparison cannot be trusted (wrong split, mismatched models or results)."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    results_dir: Path


@dataclass(frozen=True)
class Paired:
    """Counts of the paired 2x2 table (A = first model, B = second model)."""

    both_right: int
    a_only: int
    b_only: int
    both_wrong: int


def paired_table(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Paired:
    if not (y_true.shape == pred_a.shape == pred_b.shape):
        raise ValueError("labels and both predictions must have the same shape")
    a, b = pred_a == y_true, pred_b == y_true
    return Paired(
        both_right=int((a & b).sum()),
        a_only=int((a & ~b).sum()),
        b_only=int((~a & b).sum()),
        both_wrong=int((~a & ~b).sum()),
    )


def adr022_outcome(p_value: float, alpha: float, names: Sequence[str]) -> str:
    """The pre-declared rule of ADR-022 (first model = the pre-declared main model)."""
    if p_value >= alpha:
        return (
            f"not significant (p >= {alpha}): {names[0]} stays the main model; "
            f"{names[1]} is reported as the lighter alternative"
        )
    return f"significant (p < {alpha}): stop, the project owner decides (no automatic switch)"


def load_checkpoint(path: Path) -> tuple[nn.Module, list[str], str]:
    """Trained network on CPU (no pretrained download), its class order and timm name."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), list(ckpt["classes"]), str(ckpt["model_name"])


def specs_from_params(cfg: Mapping[str, Any]) -> list[ModelSpec]:
    specs = [
        ModelSpec(m["name"], Path(m["checkpoint"]), Path(m["results_dir"])) for m in cfg["models"]
    ]
    if len(specs) != 2:
        raise CompareError("compare_val.models must list exactly two models")
    return specs


def _recorded(spec: ModelSpec) -> dict[str, Any]:
    return json.loads((spec.results_dir / "val_metrics.json").read_text(encoding="utf-8"))


def check_reproduces(
    spec: ModelSpec, recorded: Mapping[str, Any], acc: float, f1: float, tol: float
) -> None:
    """Refuse to test if the local predictions do not match what training recorded."""
    want_sha = recorded.get("checkpoint_sha256")
    if want_sha and sha256_file(spec.checkpoint) != want_sha:
        raise CompareError(f"{spec.checkpoint} is not the checkpoint of {spec.results_dir}")
    for key, got in (("val_accuracy", acc), ("val_macro_f1", f1)):
        if abs(float(recorded[key]) - got) > tol:
            raise CompareError(
                f"{spec.name}: {key} {got:.6f} does not reproduce the recorded "
                f"{float(recorded[key]):.6f} (tolerance {tol})"
            )


def run(params: Mapping[str, Any]) -> dict[str, Any]:
    cfg = params["compare_val"]
    split = str(cfg["split"])
    if split == PROTECTED_SPLIT:
        raise CompareError(f"model choice never uses '{PROTECTED_SPLIT}' (ADR-022)")
    specs = specs_from_params(cfg)
    split_csv, raw_dir = Path(params["data"]["split_file"]), Path(params["data"]["raw_dir"])
    pre_cfg = PreprocessConfig.from_params(params)
    device = torch.device("cpu")

    per_model, preds, y_true, paths, classes = [], [], None, None, None
    for spec in specs:
        model, model_classes, timm_name = load_checkpoint(spec.checkpoint)
        if classes is not None and model_classes != classes:
            raise CompareError("the two models use different class orders")
        classes = model_classes
        ds = SplitDataset(split_csv, raw_dir, [split], classes, pre_cfg)
        loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False)
        _, true, pred = predict(model, loader, device)
        if y_true is not None and not np.array_equal(true, y_true):
            raise CompareError("the two models did not see the same images in the same order")
        y_true, paths = true, ds.paths
        acc, f1 = accuracy(true, pred), macro_f1(true, pred, len(classes))
        recorded = _recorded(spec)
        check_reproduces(spec, recorded, acc, f1, float(cfg["reproduce_tolerance"]))
        preds.append(pred)
        per_model.append(
            {
                "name": spec.name,
                "timm_model": timm_name,
                "results_dir": spec.results_dir.as_posix(),
                "checkpoint_sha256": recorded.get("checkpoint_sha256"),
                "run_uid": recorded.get("run_uid"),
                "accuracy": acc,
                "macro_f1": f1,
                "n_correct": int((pred == true).sum()),
                "reproduces_recorded_val_metrics": True,
            }
        )

    assert y_true is not None and paths is not None and classes is not None
    table = paired_table(y_true, preds[0], preds[1])
    p_value = mcnemar_exact(table.a_only, table.b_only)
    alpha = float(cfg["alpha"])
    names = [s.name for s in specs]
    discordant = [
        {
            "path": paths[i],
            "true": classes[y_true[i]],
            f"pred_{names[0]}": classes[preds[0][i]],
            f"pred_{names[1]}": classes[preds[1][i]],
        }
        for i in np.flatnonzero((preds[0] == y_true) != (preds[1] == y_true))
    ]
    return {
        "decision_record": "ADR-022",
        "split": split,
        "split_file": split_csv.name,
        "split_sha256": sha256_file(split_csv),
        "n_images": int(len(y_true)),
        "device": "cpu",
        "models": per_model,
        "table": {
            "both_right": table.both_right,
            f"only_{names[0]}_right": table.a_only,
            f"only_{names[1]}_right": table.b_only,
            "both_wrong": table.both_wrong,
        },
        "test": "exact two-sided McNemar (binomial test on discordant images, p = 0.5)",
        "n_discordant": table.a_only + table.b_only,
        "p_value": p_value,
        "alpha": alpha,
        "significant": p_value < alpha,
        "outcome": adr022_outcome(p_value, alpha, names),
        "discordant_images": discordant,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/params.yaml"))
    args = parser.parse_args(argv)
    params = load_params(args.config)
    try:
        report = run(params)
    except CompareError as exc:
        print(f"compare_val: {exc}", file=sys.stderr)
        return 1
    out = Path(params["compare_val"]["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(report, indent=2) + "\n").encode())
    t = report["table"]
    print(f"{report['n_images']} images; table {t}")
    print(f"p = {report['p_value']:.4g} -> {report['outcome']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
