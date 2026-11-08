"""Static INT8 (QDQ) quantisation of the champion's FP32 ONNX (TASK-016, ADR-002, ADR-026).

    python -m src.training.quantize --config configs/params.yaml [--log-mlflow]

CPU only. Steps:
  1. input = the FP32 file of TASK-015, only if its SHA-256 matches ``export_report.json`` and
     the ``@champion`` version's ``onnx_sha256`` tag;
  2. ``quant_pre_process``, then ``quantize_static`` (QDQ, settings in ``params.yaml`` ->
     ``quantize``) calibrated on ``train_v1`` only, through the shared preprocessing;
  3. FP32 vs INT8 on ``val`` (never ``test_frozen``): per-class precision/recall/F1, class and
     review-routing agreement, logit changes, ECE with the champion's T, routing and the
     ADR-023 risk bound at the stored threshold, file size and a rough CPU latency;
  4. the ADR-026 rule decides ``serving_precision``: ``int8`` only if no class F1 drops by more
     than ``max_class_f1_drop``, macro-F1 by more than ``max_macro_f1_drop``, and the INT8 risk
     bound stays <= ``threshold.max_risk_ucb``; otherwise ``fp32`` (the INT8 file is kept and
     logged as evidence either way; no other setting is tried).

Writes ``<model>-v<version>.int8.onnx`` next to the FP32 file (git-ignored) and
``quantize.report_file``. ``--log-mlflow`` adds both to the champion's run under ``onnx/`` and
tags the version with ``onnx_int8_sha256``, ``onnx_int8_artifact`` and ``serving_precision``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch.utils.data import DataLoader

from src.config import load_params
from src.serving.onnx_model import INPUT_NAME, OnnxClassifier
from src.serving.preprocess import PreprocessConfig
from src.training.dataset import PROTECTED_SPLIT, SplitDataset
from src.training.export_onnx import ExportConfig, champion
from src.training.metrics import (
    clopper_pearson_upper,
    ece,
    per_class_precision_recall_f1,
    softmax,
    top_confidence,
)
from src.training.train import git_commit, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]


class QuantizeError(RuntimeError):
    """Quantisation must not run (wrong input file, calibration on a held-out split)."""


@dataclass(frozen=True)
class QuantizeConfig:
    calibration_split: str
    calibrate_method: str
    per_channel: bool
    activation_type: str
    weight_type: str
    reduce_range: bool
    batch_size: int
    max_class_f1_drop: float
    max_macro_f1_drop: float
    latency_warmup: int
    latency_runs: int
    latency_batch_sizes: tuple[int, ...]
    report_file: Path

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> QuantizeConfig:
        q, rule, lat = params["quantize"], params["quantize"]["rule"], params["quantize"]["latency"]
        return cls(
            calibration_split=str(q["calibration_split"]),
            calibrate_method=str(q["calibrate_method"]),
            per_channel=bool(q["per_channel"]),
            activation_type=str(q["activation_type"]),
            weight_type=str(q["weight_type"]),
            reduce_range=bool(q["reduce_range"]),
            batch_size=int(q["batch_size"]),
            max_class_f1_drop=float(rule["max_class_f1_drop"]),
            max_macro_f1_drop=float(rule["max_macro_f1_drop"]),
            latency_warmup=int(lat["warmup"]),
            latency_runs=int(lat["runs"]),
            latency_batch_sizes=tuple(int(b) for b in lat["batch_sizes"]),
            report_file=Path(q["report_file"]),
        )


# ---------------------------------------------------------------- calibration data


class BatchReader(CalibrationDataReader):
    """Feeds preprocessed batches to ONNX Runtime's calibrator, once."""

    def __init__(self, batches: Sequence[np.ndarray]):
        self._it: Iterator[np.ndarray] = iter(batches)

    def get_next(self) -> dict[str, np.ndarray] | None:
        batch = next(self._it, None)
        return None if batch is None else {INPUT_NAME: np.ascontiguousarray(batch, np.float32)}


def check_calibration_split(split: str, params: Mapping[str, Any]) -> None:
    """ADR-002: calibrate on training data only, never on val or test_frozen."""
    if split in (PROTECTED_SPLIT, str(params["train"]["val_split"])):
        raise QuantizeError(f"quantisation is never calibrated on '{split}' (ADR-002)")
    if split not in params["train"]["train_splits"]:
        raise QuantizeError(f"'{split}' is not a training split (train.train_splits)")


def calibration_batches(
    params: Mapping[str, Any], split: str, classes: Sequence[str], batch_size: int
) -> list[np.ndarray]:
    check_calibration_split(split, params)
    ds = SplitDataset(
        Path(params["data"]["split_file"]),
        Path(params["data"]["raw_dir"]),
        [split],
        classes,
        PreprocessConfig.from_params(params),
    )  # no augmentation: the shared preprocessing only
    return [x.numpy() for x, _ in DataLoader(ds, batch_size=batch_size)]


# ---------------------------------------------------------------- quantise


def quantize_file(
    fp32: Path, int8: Path, batches: Sequence[np.ndarray], cfg: QuantizeConfig
) -> None:
    """``quant_pre_process`` then static QDQ quantisation; metadata copied from FP32."""
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        quant_pre_process(str(fp32), str(pre))
        quantize_static(
            str(pre),
            str(int8),
            BatchReader(batches),
            quant_format=QuantFormat.QDQ,
            per_channel=cfg.per_channel,
            reduce_range=cfg.reduce_range,
            activation_type=QuantType[cfg.activation_type],
            weight_type=QuantType[cfg.weight_type],
            calibrate_method=CalibrationMethod[cfg.calibrate_method],
        )
    meta = {p.key: p.value for p in onnx.load(str(fp32)).metadata_props}
    meta.update(
        {
            "precision": "int8",
            "quantization": (
                f"static {QuantFormat.QDQ.name}; weights {cfg.weight_type}"
                f"{' per channel' if cfg.per_channel else ''}; activations "
                f"{cfg.activation_type}; {cfg.calibrate_method} calibration"
            ),
            "calibration_split": cfg.calibration_split,
            "fp32_sha256": sha256_file(fp32),
        }
    )
    graph = onnx.load(str(int8))
    del graph.metadata_props[:]
    onnx.helper.set_model_props(graph, meta)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(int8))


# ---------------------------------------------------------------- compare on val


def scores(y: np.ndarray, logits: np.ndarray, classes: Sequence[str]) -> dict[str, Any]:
    pred = logits.argmax(axis=1)
    precision, recall, f1 = per_class_precision_recall_f1(y, pred, len(classes))
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1.mean()),
        "n_errors": int((pred != y).sum()),
        "per_class": {
            c: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i, c in enumerate(classes)
        },
    }


def routing(
    y: np.ndarray, logits: np.ndarray, temperature: float, threshold: float, level: float
) -> dict[str, Any]:
    """ADR-023 quantities at the stored threshold (no bootstrap here)."""
    conf = top_confidence(softmax(logits, temperature))
    accepted = conf >= threshold
    wrong = logits.argmax(axis=1) != y
    k, n_acc = int((wrong & accepted).sum()), int(accepted.sum())
    return {
        "n_routed": int((~accepted).sum()),
        "errors_routed": int((wrong & ~accepted).sum()),
        "errors_accepted": k,
        "risk": k / n_acc if n_acc else 0.0,
        "risk_ucb": clopper_pearson_upper(k, n_acc, level),
    }


def decide(
    fp32: Mapping[str, Any],
    int8: Mapping[str, Any],
    int8_routing: Mapping[str, Any],
    cfg: QuantizeConfig,
    max_risk_ucb: float,
) -> dict[str, Any]:
    """ADR-026: INT8 serves only if every check passes; otherwise FP32 (the fallback)."""
    drops = {c: fp32["per_class"][c]["f1"] - int8["per_class"][c]["f1"] for c in fp32["per_class"]}
    checks = {
        "max_class_f1_drop": {
            "value": max(drops.values()),
            "limit": cfg.max_class_f1_drop,
            "worst_class": max(drops, key=drops.get),
        },
        "macro_f1_drop": {
            "value": fp32["macro_f1"] - int8["macro_f1"],
            "limit": cfg.max_macro_f1_drop,
        },
        "int8_risk_ucb": {"value": int8_routing["risk_ucb"], "limit": max_risk_ucb},
    }
    for c in checks.values():
        c["passed"] = bool(c["value"] <= c["limit"] + 1e-12)
    passed = all(c["passed"] for c in checks.values())
    return {
        "decision_record": "ADR-026",
        "class_f1_drop": drops,
        "checks": checks,
        "passed": passed,
        "serving_precision": "int8" if passed else "fp32",
    }


def latency_ms(session: OnnxClassifier, x: np.ndarray, batch: int, warmup: int, runs: int) -> dict:
    """Median and p90 wall time of one ``session.logits`` call (rough; one machine)."""
    b = np.ascontiguousarray(x[:batch])
    for _ in range(warmup):
        session.logits(b)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.logits(b)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "batch": int(batch),
        "median_ms": float(np.median(times)),
        "p90_ms": float(np.percentile(times, 90)),
        "median_ms_per_image": float(np.median(times) / batch),
    }


# ---------------------------------------------------------------- run


def fp32_input(params: Mapping[str, Any], champ: Mapping[str, Any]) -> tuple[Path, dict]:
    """The TASK-015 file, only if it is the one reported and tagged on the champion."""
    report = json.loads(ExportConfig.from_params(params).report_file.read_text("utf-8"))
    path = Path(report["onnx_file"])
    if report["version"] != champ["version"]:
        raise QuantizeError("the ONNX export report is for another version than @champion")
    if not path.is_file():
        raise QuantizeError(f"missing {path}: run `make export-onnx` first")
    sha = sha256_file(path)
    if not (sha == report["onnx_sha256"] == champ["tags"].get("onnx_sha256")):
        raise QuantizeError(f"{path} is not the FP32 file reported and tagged on @champion")
    return path, report


def run(params: Mapping[str, Any], *, log_mlflow: bool = False) -> dict[str, Any]:
    cfg = QuantizeConfig.from_params(params)
    val_split = str(params["train"]["val_split"])
    if val_split == PROTECTED_SPLIT:
        raise QuantizeError(f"the FP32 vs INT8 comparison never uses '{PROTECTED_SPLIT}'")
    check_calibration_split(cfg.calibration_split, params)
    champ = champion(params)
    fp32_path, export_report = fp32_input(params, champ)
    classes = list(export_report["classes"])
    int8_path = fp32_path.with_name(fp32_path.name.replace(".fp32.onnx", ".int8.onnx"))
    batches = calibration_batches(params, cfg.calibration_split, classes, cfg.batch_size)
    quantize_file(fp32_path, int8_path, batches, cfg)

    fp32, int8 = OnnxClassifier(fp32_path), OnnxClassifier(int8_path)
    if int8.classes != classes:
        raise QuantizeError("class order in the INT8 metadata differs from FP32")
    ds = SplitDataset(
        Path(params["data"]["split_file"]),
        Path(params["data"]["raw_dir"]),
        [val_split],
        classes,
        PreprocessConfig.from_params(params),
    )
    ys, lf, li, xs = [], [], [], []
    for x, y in DataLoader(ds, batch_size=cfg.batch_size):
        arr = x.numpy()
        lf.append(fp32.logits(arr))
        li.append(int8.logits(arr))
        ys.append(y.numpy())
        xs.append(arr)
    y, lf, li, x_all = (
        np.concatenate(ys),
        np.concatenate(lf),
        np.concatenate(li),
        np.concatenate(xs),
    )

    temperature = float(champ["tags"]["temperature"])
    threshold = float(champ["tags"]["review_threshold"])
    level = float(params["threshold"]["ucb_level"])
    n_bins = int(params["calibrate"]["n_bins"])
    s_fp32, s_int8 = scores(y, lf, classes), scores(y, li, classes)
    r_fp32 = routing(y, lf, temperature, threshold, level)
    r_int8 = routing(y, li, temperature, threshold, level)
    conf_f = top_confidence(softmax(lf, temperature))
    conf_i = top_confidence(softmax(li, temperature))
    decision = decide(s_fp32, s_int8, r_int8, cfg, float(params["threshold"]["max_risk_ucb"]))
    gap = np.abs(li.astype(np.float64) - lf.astype(np.float64))
    commit = git_commit(REPO_ROOT)
    report = {
        "decision_record": "ADR-026",
        "registered_model": champ["name"],
        "version": champ["version"],
        "alias": champ["alias"],
        "run_id": champ["run_id"],
        "fp32_file": fp32_path.as_posix(),
        "fp32_sha256": sha256_file(fp32_path),
        "int8_file": int8_path.as_posix(),
        "int8_sha256": sha256_file(int8_path),
        "bytes": {"fp32": fp32_path.stat().st_size, "int8": int8_path.stat().st_size},
        "settings": {
            "pre_process": "onnxruntime.quantization.shape_inference.quant_pre_process",
            "format": QuantFormat.QDQ.name,
            "per_channel": cfg.per_channel,
            "activation_type": cfg.activation_type,
            "weight_type": cfg.weight_type,
            "reduce_range": cfg.reduce_range,
            "calibrate_method": cfg.calibrate_method,
            "calibration_split": cfg.calibration_split,
            "calibration_images": int(sum(len(b) for b in batches)),
        },
        "compared_on": val_split,
        "n_images": int(len(y)),
        "classes": classes,
        "fp32": {**s_fp32, "ece": ece(softmax(lf, temperature), y, n_bins), "routing": r_fp32},
        "int8": {**s_int8, "ece": ece(softmax(li, temperature), y, n_bins), "routing": r_int8},
        "agreement": {
            "same_class": int((lf.argmax(axis=1) == li.argmax(axis=1)).sum()),
            "same_routing": int(((conf_f < threshold) == (conf_i < threshold)).sum()),
            "max_abs_logit_change": float(gap.max()),
            "mean_abs_logit_change": float(gap.mean()),
            "max_abs_calibrated_confidence_change": float(np.abs(conf_i - conf_f).max()),
        },
        "applied_with": {"temperature": temperature, "review_threshold": threshold},
        "rule": decision,
        "serving_precision": decision["serving_precision"],
        "latency_cpu": {
            "note": "rough: one laptop CPU, single process; the load benchmark is TASK-022",
            "fp32": [
                latency_ms(fp32, x_all, b, cfg.latency_warmup, cfg.latency_runs)
                for b in cfg.latency_batch_sizes
            ],
            "int8": [
                latency_ms(int8, x_all, b, cfg.latency_warmup, cfg.latency_runs)
                for b in cfg.latency_batch_sizes
            ],
        },
        "caution": (
            f"{len(y)} val images and {s_fp32['n_errors']} FP32 errors: the rule can only catch "
            "large damage; passing means no visible loss on val, not equality with FP32."
        ),
        "versions": {"onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__},
        "provenance": {"git_commit": commit["sha"], "git_dirty": commit["dirty"], "device": "cpu"},
    }
    cfg.report_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_file.write_bytes((json.dumps(report, indent=2) + "\n").encode())
    out = {"report": report, "int8": int8_path, "report_file": cfg.report_file}
    if log_mlflow:
        out["mlflow_run"] = log_to_mlflow(params, champ, int8_path, cfg.report_file, report)
    return out


def log_to_mlflow(
    params: Mapping[str, Any],
    champ: Mapping[str, Any],
    int8_path: Path,
    report_file: Path,
    report: Mapping[str, Any],
) -> str:
    from src.tracking import mlflow_tracking as mt  # only with --log-mlflow

    cfg = mt.TrackingConfig.from_params(params, smoke=False)
    lat = {d["batch"]: d for d in report["latency_cpu"]["int8"]}
    lat_fp32 = {d["batch"]: d for d in report["latency_cpu"]["fp32"]}
    metrics = {
        "fp32_val_macro_f1": report["fp32"]["macro_f1"],
        "int8_val_macro_f1": report["int8"]["macro_f1"],
        "int8_val_accuracy": report["int8"]["accuracy"],
        "int8_val_max_class_f1_drop": report["rule"]["checks"]["max_class_f1_drop"]["value"],
        "int8_val_same_class": float(report["agreement"]["same_class"]),
        "int8_val_same_routing": float(report["agreement"]["same_routing"]),
        "int8_val_risk_ucb": report["int8"]["routing"]["risk_ucb"],
        "int8_val_ece": report["int8"]["ece"],
        "int8_bytes": float(report["bytes"]["int8"]),
    }
    for b in lat:
        metrics[f"int8_latency_ms_batch{b}"] = lat[b]["median_ms"]
        metrics[f"fp32_latency_ms_batch{b}"] = lat_fp32[b]["median_ms"]
    link = mt.add_to_finished_run(
        cfg,
        champ["run_id"],
        metrics=metrics,
        tags={
            "onnx_int8_sha256": report["int8_sha256"],
            "serving_precision": report["serving_precision"],
            "quantize_decision_record": "ADR-026",
        },
        files=[(int8_path, "onnx"), (report_file, "onnx")],
    )
    mt.set_version_tags(
        cfg,
        champ["version"],
        {
            "onnx_int8_sha256": report["int8_sha256"],
            "onnx_int8_artifact": f"onnx/{int8_path.name}",
            "serving_precision": report["serving_precision"],
        },
    )
    return link


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="log the INT8 file and report to the champion's run and tag serving_precision",
    )
    args = parser.parse_args(argv)
    try:
        out = run(load_params(args.config), log_mlflow=args.log_mlflow)
    except QuantizeError as exc:
        print(f"quantize: {exc}", file=sys.stderr)
        return 1
    r = out["report"]
    f, i, a = r["fp32"], r["int8"], r["agreement"]
    print(
        f"{r['registered_model']} v{r['version']}: INT8 {r['bytes']['int8'] / 1e6:.1f} MB "
        f"(FP32 {r['bytes']['fp32'] / 1e6:.1f} MB) -> {out['int8'].as_posix()}"
    )
    print(
        f"  val ({r['n_images']}): macro-F1 FP32 {f['macro_f1']:.4f} -> INT8 {i['macro_f1']:.4f}; "
        f"same class {a['same_class']}/{r['n_images']}, same routing {a['same_routing']}/"
        f"{r['n_images']}; INT8 risk bound {i['routing']['risk_ucb']:.4f}"
    )
    for name, c in r["rule"]["checks"].items():
        print(f"  {name}: {c['value']:.4f} (limit {c['limit']}) {'ok' if c['passed'] else 'FAIL'}")
    print(f"  serving_precision = {r['serving_precision']} (ADR-026)")
    for fp, q in zip(r["latency_cpu"]["fp32"], r["latency_cpu"]["int8"], strict=True):
        print(
            f"  latency batch {fp['batch']}: FP32 {fp['median_ms']:.1f} ms, "
            f"INT8 {q['median_ms']:.1f} ms"
        )
    print(f"  wrote {out['report_file'].as_posix()}")
    if "mlflow_run" in out:
        print(f"  MLflow: {out['mlflow_run']} (+ version tags)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
