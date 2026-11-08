"""FP32 ONNX export of the ``@champion`` model with a parity check on val (TASK-015, ADR-025).

    python -m src.training.export_onnx --config configs/params.yaml [--log-mlflow]

CPU only, no training. Steps:
  1. resolve ``<registered_model>@<champion_alias>`` in the registry; its run must be the main
     model's run (``mlflow_run.json``) and the local checkpoint's SHA-256 must equal the
     version's ``checkpoint_sha256`` tag;
  2. export raw logits (``input`` [batch, 3, S, S] -> ``logits`` [batch, n]), dynamic batch
     axis, one self-contained file, with metadata (version, checkpoint hash, classes,
     preprocessing). The example input is synthetic: no NEU pixels reach the export;
  3. parity on ``val`` (never ``test_frozen``): ONNX Runtime vs PyTorch logits within
     ``atol + rtol * |torch logit|`` (``export.parity``), identical predicted classes,
     identical review routing at the
     champion's T and threshold, and several batch sizes. Any failure: nothing is kept.

Writes ``<export.out_dir>/<model>-v<version>.fp32.onnx`` (git-ignored) and
``export.report_file``. ``--log-mlflow`` adds both to the champion's MLflow run under ``onnx/``
and tags the version with ``onnx_sha256`` / ``onnx_artifact`` so serving can verify the file.
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
import onnx
import onnxruntime
import onnxscript
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import load_params
from src.serving.onnx_model import INPUT_NAME, OUTPUT_NAME, OnnxClassifier
from src.serving.preprocess import PreprocessConfig
from src.training.compare_val import ModelSpec, load_checkpoint, specs_from_params
from src.training.dataset import PROTECTED_SPLIT, SplitDataset
from src.training.metrics import softmax, top_confidence
from src.training.train import git_commit, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_TAGS = ("model", "checkpoint_sha256", "temperature", "review_threshold")


class ExportError(RuntimeError):
    """The ONNX file must not be kept (no champion, wrong checkpoint, parity failed)."""


@dataclass(frozen=True)
class ExportConfig:
    opset: int
    out_dir: Path
    report_file: Path
    batch_size: int
    batch_sizes: tuple[int, ...]
    atol: float
    rtol: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> ExportConfig:
        e = params["export"]
        return cls(
            opset=int(e["opset"]),
            out_dir=Path(e["out_dir"]),
            report_file=Path(e["report_file"]),
            batch_size=int(e["batch_size"]),
            batch_sizes=tuple(int(b) for b in e["batch_sizes"]),
            atol=float(e["parity"]["atol"]),
            rtol=float(e["parity"]["rtol"]),
        )


# ---------------------------------------------------------------- which model


def champion(params: Mapping[str, Any]) -> dict[str, Any]:
    """The ``@champion`` version (number, run id, tags) from the registry."""
    from src.tracking import mlflow_tracking as mt  # registry lookup

    cfg = mt.TrackingConfig.from_params(params, smoke=False)
    alias = str(params["promotion"]["champion_alias"])
    found = mt.version_by_alias(cfg, alias)
    if found is None:
        raise ExportError(f"no @{alias} on {cfg.registered_model}: run `make champion` first")
    missing = [k for k in REQUIRED_TAGS if k not in found["tags"]]
    if missing:
        raise ExportError(f"@{alias} version {found['version']} lacks tags {missing} (ADR-024)")
    return {**found, "name": cfg.registered_model, "alias": alias}


def checkpoint_for(
    params: Mapping[str, Any], champ: Mapping[str, Any], root: Path = REPO_ROOT
) -> ModelSpec:
    """The local main-model checkpoint, only if it is the champion's (run id and hash)."""
    spec = specs_from_params(params["compare_val"])[0]
    if spec.name != champ["tags"]["model"]:
        raise ExportError(f"@{champ['alias']} is {champ['tags']['model']}, not {spec.name}")
    pointer = json.loads((root / spec.results_dir / "mlflow_run.json").read_text("utf-8"))
    if pointer["run_id"] != champ["run_id"]:
        raise ExportError(f"@{champ['alias']} comes from another run than {spec.results_dir}")
    path = root / spec.checkpoint
    if not path.is_file():
        raise ExportError(f"missing checkpoint {spec.checkpoint}")
    if sha256_file(path) != champ["tags"]["checkpoint_sha256"]:
        raise ExportError(f"{spec.checkpoint} is not the @{champ['alias']} checkpoint (SHA-256)")
    return spec


# ---------------------------------------------------------------- export


def export_model(
    model: nn.Module, path: Path, image_size: int, opset: int, metadata: Mapping[str, str]
) -> None:
    """Raw-logit graph with a dynamic batch axis and metadata, checked by ``onnx.checker``."""
    example = torch.zeros(2, 3, image_size, image_size)  # synthetic: no dataset pixels
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.eval(),
        (example,),
        str(path),
        dynamo=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=4096)},),
        opset_version=opset,
        external_data=False,
    )
    graph = onnx.load(str(path))
    onnx.helper.set_model_props(graph, dict(metadata))
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(path))


# ---------------------------------------------------------------- parity


@torch.inference_mode()
def torch_logits(model: nn.Module, x: np.ndarray) -> np.ndarray:
    return model.eval()(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))).numpy()


def tolerance_ratio(ref: np.ndarray, got: np.ndarray, atol: float, rtol: float) -> float:
    """Largest ``|got - ref| / (atol + rtol * |ref|)``; <= 1 means within tolerance (allclose)."""
    ref64 = ref.astype(np.float64)
    gap = np.abs(got.astype(np.float64) - ref64)
    return float((gap / (atol + rtol * np.abs(ref64))).max())


def parity_report(
    ref: np.ndarray,
    got: np.ndarray,
    temperature: float,
    threshold: float,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """ONNX vs PyTorch: logit gap, predicted classes and review routing (ADR-025)."""
    gap = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    conf_ref = top_confidence(softmax(ref, temperature))
    conf_got = top_confidence(softmax(got, temperature))
    routed_ref, routed_got = conf_ref < threshold, conf_got < threshold
    n = len(ref)
    report = {
        "n_images": int(n),
        "atol": atol,
        "rtol": rtol,
        "rule": "|onnx - torch| <= atol + rtol * |torch| for every logit",
        "max_tolerance_ratio": tolerance_ratio(ref, got, atol, rtol),
        "max_abs_logit_diff": float(gap.max()),
        "mean_abs_logit_diff": float(gap.mean()),
        "same_class": int((ref.argmax(axis=1) == got.argmax(axis=1)).sum()),
        "same_routing": int((routed_ref == routed_got).sum()),
        "routed_torch": int(routed_ref.sum()),
        "max_abs_calibrated_confidence_diff": float(np.abs(conf_got - conf_ref).max()),
        "min_distance_to_threshold": float(np.abs(conf_ref - threshold).min()),
    }
    report["passed"] = bool(
        report["max_tolerance_ratio"] <= 1.0
        and report["same_class"] == n
        and report["same_routing"] == n
    )
    return report


def batch_check(
    session: OnnxClassifier,
    model: nn.Module,
    x: np.ndarray,
    sizes: Sequence[int],
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    """The dynamic batch axis: each batch size gives the PyTorch logits within tolerance."""
    out = []
    for n in sizes:
        got, ref = session.logits(x[:n]), torch_logits(model, x[:n])
        ratio = tolerance_ratio(ref, got, atol, rtol)
        out.append(
            {
                "batch": int(n),
                "max_abs_logit_diff": float(np.abs(got - ref).max()),
                "max_tolerance_ratio": ratio,
                "passed": ratio <= 1.0,
            }
        )
    return out


def val_outputs(
    model: nn.Module, session: OnnxClassifier, loader: DataLoader, keep: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Labels, PyTorch logits, ONNX logits for every val image, and the first ``keep`` inputs."""
    ys, refs, gots, xs = [], [], [], []
    kept = 0
    for x, y in loader:
        arr = x.numpy()
        refs.append(torch_logits(model, arr))
        gots.append(session.logits(arr))
        ys.append(y.numpy())
        if kept < keep:
            xs.append(arr[: keep - kept])
            kept += len(xs[-1])
    return np.concatenate(ys), np.concatenate(refs), np.concatenate(gots), np.concatenate(xs)


# ---------------------------------------------------------------- run


def run(params: Mapping[str, Any], *, log_mlflow: bool = False) -> dict[str, Any]:
    val_split = str(params["train"]["val_split"])
    if val_split == PROTECTED_SPLIT:
        raise ExportError(f"the parity check never uses '{PROTECTED_SPLIT}'")
    cfg = ExportConfig.from_params(params)
    champ = champion(params)
    spec = checkpoint_for(params, champ)
    model, classes, timm_name = load_checkpoint(spec.checkpoint)
    pre = PreprocessConfig.from_params(params)
    temperature, threshold = (
        float(champ["tags"]["temperature"]),
        float(champ["tags"]["review_threshold"]),
    )
    commit = git_commit(REPO_ROOT)
    metadata = {
        "registered_model": champ["name"],
        "registry_version": champ["version"],
        "exported_from_alias": champ["alias"],
        "checkpoint_sha256": champ["tags"]["checkpoint_sha256"],
        "timm_model": timm_name,
        "classes": json.dumps(classes),
        "preprocess": json.dumps(dict(params["preprocess"])),
        "input": f"{INPUT_NAME} [batch, 3, {pre.image_size}, {pre.image_size}] float32 from "
        "src/serving/preprocess.py",
        "output": f"{OUTPUT_NAME} [batch, {len(classes)}]: raw logits; serving applies "
        "softmax(logits / T) and the review threshold from the @champion tags (ADR-025)",
        "git_commit": commit["sha"],
    }
    path = cfg.out_dir / f"{champ['name']}-v{champ['version']}.fp32.onnx"
    part = path.with_name(path.name + ".part")
    export_model(model, part, pre.image_size, cfg.opset, metadata)
    try:
        session = OnnxClassifier(part)
        if session.classes != classes:
            raise ExportError("class order in the ONNX metadata differs from the checkpoint")
        ds = SplitDataset(
            Path(params["data"]["split_file"]),
            Path(params["data"]["raw_dir"]),
            [val_split],
            classes,
            pre,
        )
        y, ref, got, x = val_outputs(
            model, session, DataLoader(ds, batch_size=cfg.batch_size), max(cfg.batch_sizes)
        )
        parity = parity_report(ref, got, temperature, threshold, cfg.atol, cfg.rtol)
        parity["batch_check"] = batch_check(session, model, x, cfg.batch_sizes, cfg.atol, cfg.rtol)
        parity["val_accuracy_torch"] = float((ref.argmax(axis=1) == y).mean())
        parity["val_accuracy_onnx"] = float((got.argmax(axis=1) == y).mean())
        if not (parity["passed"] and all(b["passed"] for b in parity["batch_check"])):
            raise ExportError(f"parity failed, nothing kept: {json.dumps(parity)}")
    except Exception:
        part.unlink(missing_ok=True)
        raise
    part.replace(path)
    report = {
        "decision_record": "ADR-025",
        "registered_model": champ["name"],
        "version": champ["version"],
        "alias": champ["alias"],
        "run_id": champ["run_id"],
        "model": spec.name,
        "timm_model": timm_name,
        "checkpoint_sha256": champ["tags"]["checkpoint_sha256"],
        "onnx_file": path.as_posix(),
        "onnx_sha256": sha256_file(path),
        "onnx_bytes": path.stat().st_size,
        "opset": cfg.opset,
        "graph": {
            "input": [INPUT_NAME, "batch", 3, pre.image_size, pre.image_size],
            "output": [OUTPUT_NAME, "batch", len(classes)],
            "output_is": "raw logits (no softmax, no temperature)",
            "example_input": "synthetic zeros (no dataset pixels)",
        },
        "classes": classes,
        "parity_split": val_split,
        "parity_applied_with": {"temperature": temperature, "review_threshold": threshold},
        "parity": parity,
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxscript": onnxscript.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "provenance": {"git_commit": commit["sha"], "git_dirty": commit["dirty"], "device": "cpu"},
    }
    cfg.report_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.report_file.write_bytes((json.dumps(report, indent=2) + "\n").encode())
    out = {"report": report, "onnx": path, "report_file": cfg.report_file}
    if log_mlflow:
        out["mlflow_run"] = log_to_mlflow(params, champ, path, cfg.report_file, report)
    return out


def log_to_mlflow(
    params: Mapping[str, Any],
    champ: Mapping[str, Any],
    onnx_path: Path,
    report_file: Path,
    report: Mapping[str, Any],
) -> str:
    from src.tracking import mlflow_tracking as mt  # only with --log-mlflow

    cfg = mt.TrackingConfig.from_params(params, smoke=False)
    p = report["parity"]
    link = mt.add_to_finished_run(
        cfg,
        champ["run_id"],
        metrics={
            "onnx_parity_max_abs_logit_diff": p["max_abs_logit_diff"],
            "onnx_parity_mean_abs_logit_diff": p["mean_abs_logit_diff"],
            "onnx_parity_same_class": float(p["same_class"]),
            "onnx_parity_same_routing": float(p["same_routing"]),
            "onnx_bytes": float(report["onnx_bytes"]),
        },
        tags={
            "onnx_sha256": report["onnx_sha256"],
            "onnx_opset": report["opset"],
            "onnx_decision_record": "ADR-025",
        },
        files=[(onnx_path, "onnx"), (report_file, "onnx")],
    )
    mt.set_version_tags(
        cfg,
        champ["version"],
        {"onnx_sha256": report["onnx_sha256"], "onnx_artifact": f"onnx/{onnx_path.name}"},
    )
    return link


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # the exporter prints emoji (cp1252 consoles)
        stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="log the ONNX file and report to the champion's run and tag the version",
    )
    args = parser.parse_args(argv)
    try:
        out = run(load_params(args.config), log_mlflow=args.log_mlflow)
    except ExportError as exc:
        print(f"export_onnx: {exc}", file=sys.stderr)
        return 1
    r, p = out["report"], out["report"]["parity"]
    print(
        f"{r['registered_model']} v{r['version']} (@{r['alias']}) -> {out['onnx'].as_posix()} "
        f"({r['onnx_bytes'] / 1e6:.1f} MB, opset {r['opset']}, sha256 {r['onnx_sha256'][:12]}...)"
    )
    print(
        f"  parity on {r['parity_split']} ({p['n_images']} images): max |logit diff| "
        f"{p['max_abs_logit_diff']:.2e}; tolerance atol {p['atol']:.0e} + rtol {p['rtol']:.0e} "
        f"x |logit|, worst ratio {p['max_tolerance_ratio']:.2f} (<= 1 passes); same class "
        f"{p['same_class']}/{p['n_images']}, same routing {p['same_routing']}/{p['n_images']}"
    )
    for b in p["batch_check"]:
        print(f"  batch {b['batch']}: max |logit diff| {b['max_abs_logit_diff']:.2e}")
    print(f"  wrote {out['report_file'].as_posix()}")
    if "mlflow_run" in out:
        print(f"  MLflow: {out['mlflow_run']} (+ version tags onnx_sha256, onnx_artifact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
