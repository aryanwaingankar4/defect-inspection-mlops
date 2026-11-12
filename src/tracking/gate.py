"""Promotion gate (TASK-027, ADR-033): NON-INFERIORITY of ``@challenger`` against ``@champion``.

    make gate ARGS="--challenger-checkpoint PATH"            # dry run: prints, writes nothing
    make gate ARGS="--challenger-checkpoint PATH --record"   # gate file only (ADR-034)
    make gate ARGS="--challenger-checkpoint PATH --apply"    # files, registry, log

Both models are scored from their checkpoints (each checked against its version's
``checkpoint_sha256`` tag) on ``promotion.gate.split`` (``val``; never ``test_frozen``, never an
image of the feedback export). Promote only if ALL hold (thresholds in ``promotion.gate``):

  a) challenger macro-F1 >= champion macro-F1 - ``max_macro_f1_drop``;
  b) no class recall drops by more than ``max_class_recall_drop``;
  c) exact McNemar: blocked if p < ``mcnemar_alpha`` and the champion is right on more images where
     the challenger is wrong than the other way round (significantly worse);
  d) the challenger's FP32 ONNX export passes the ADR-025 parity check (``export_onnx.run``).

Val has 180 images and the champion makes few errors, so a significant improvement cannot be
shown; the gate asks "is the challenger at least as good?" (ADR-033). With ``--apply``: write
``results/promotion/gate_<challenger version>.json``, then promote (tag the challenger with the
previous champion version, move ``@champion``, remove ``@challenger``) or reject (tag only; both
aliases stay), then append the decision to ``promotion.log_file``. The reviewer that produced the
feedback is SIMULATED in this project.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_params
from src.training.compare_val import Paired, paired_table
from src.training.metrics import accuracy, macro_f1, mcnemar_exact, per_class_precision_recall_f1

REPO_ROOT = Path(__file__).resolve().parents[2]
DECISION_RECORD = "ADR-033"
TEST_SPLIT = "test_frozen"
# a promoted version must be servable: serving_bundle reads these tags (ADR-028, ADR-034)
SERVING_TAGS = ("ood_energy_threshold", "serving_precision")
CHECKS = ("macro_f1", "class_recall", "mcnemar", "onnx_parity")


class GateError(RuntimeError):
    """The gate cannot decide (missing alias, wrong checkpoint, forbidden data)."""


@dataclass(frozen=True)
class GateConfig:
    split: str
    max_macro_f1_drop: float
    max_class_recall_drop: float
    mcnemar_alpha: float
    batch_size: int
    out_dir: Path

    def __post_init__(self) -> None:
        if self.split == TEST_SPLIT:
            raise GateError(f"the promotion gate never uses '{TEST_SPLIT}' (ADR-033)")
        for name in ("max_macro_f1_drop", "max_class_recall_drop", "mcnemar_alpha"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise GateError(f"promotion.gate.{name} must be in [0, 1)")

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> GateConfig:
        g = params["promotion"]["gate"]
        return cls(
            split=str(g["split"]),
            max_macro_f1_drop=float(g["max_macro_f1_drop"]),
            max_class_recall_drop=float(g["max_class_recall_drop"]),
            mcnemar_alpha=float(g["mcnemar_alpha"]),
            batch_size=int(g["batch_size"]),
            out_dir=Path(g["out_dir"]),
        )

    def rules(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "max_macro_f1_drop": self.max_macro_f1_drop,
            "max_class_recall_drop": self.max_class_recall_drop,
            "mcnemar_alpha": self.mcnemar_alpha,
        }


# ---------------------------------------------------------------- the four checks (pure)


def model_metrics(y: np.ndarray, pred: np.ndarray, classes: Sequence[str]) -> dict[str, Any]:
    _, recall, _ = per_class_precision_recall_f1(y, pred, len(classes))
    return {
        "accuracy": accuracy(y, pred),
        "macro_f1": macro_f1(y, pred, len(classes)),
        "n_correct": int((y == pred).sum()),
        "recall": {c: float(r) for c, r in zip(classes, recall, strict=True)},
    }


def check_macro_f1(champ: Mapping[str, Any], chall: Mapping[str, Any], cfg: GateConfig) -> dict:
    drop = champ["macro_f1"] - chall["macro_f1"]
    passed = drop <= cfg.max_macro_f1_drop
    return {
        "rule": "challenger macro-F1 >= champion macro-F1 - max_macro_f1_drop",
        "value": drop,
        "limit": cfg.max_macro_f1_drop,
        "passed": bool(passed),
        "reason": f"macro-F1 drop {drop:+.4f} "
        + ("within" if passed else "exceeds")
        + f" {cfg.max_macro_f1_drop}",
    }


def check_class_recall(
    champ: Mapping[str, Any], chall: Mapping[str, Any], cfg: GateConfig
) -> dict[str, Any]:
    drops = {c: champ["recall"][c] - chall["recall"][c] for c in champ["recall"]}
    worst = max(drops, key=lambda c: drops[c])
    failed = sorted(c for c, d in drops.items() if d > cfg.max_class_recall_drop)
    reason = (
        f"recall drops above {cfg.max_class_recall_drop} in {failed}"
        if failed
        else f"largest recall drop {drops[worst]:+.4f} ({worst}) within {cfg.max_class_recall_drop}"
    )
    return {
        "rule": "no class recall drops by more than max_class_recall_drop",
        "value": drops[worst],
        "limit": cfg.max_class_recall_drop,
        "per_class_drop": drops,
        "failed_classes": failed,
        "passed": not failed,
        "reason": reason,
    }


def check_mcnemar(table: Paired, cfg: GateConfig) -> dict[str, Any]:
    """Champion = A, challenger = B. Blocks only a significantly WORSE challenger."""
    p = mcnemar_exact(table.a_only, table.b_only)
    worse = table.a_only > table.b_only
    blocked = p < cfg.mcnemar_alpha and worse
    if blocked:
        reason = (
            f"challenger significantly worse: p = {p:.4g} < {cfg.mcnemar_alpha}, "
            f"{table.a_only} champion-only vs {table.b_only} challenger-only correct"
        )
    else:
        reason = (
            f"not significantly worse: p = {p:.4g}, {table.a_only} champion-only vs "
            f"{table.b_only} challenger-only correct"
        )
    return {
        "rule": "block if exact McNemar p < mcnemar_alpha and champion-only > challenger-only",
        "value": p,
        "limit": cfg.mcnemar_alpha,
        "champion_only_correct": table.a_only,
        "challenger_only_correct": table.b_only,
        "passed": not blocked,
        "reason": reason,
    }


def check_parity(parity: Mapping[str, Any] | None) -> dict[str, Any]:
    """``parity`` is the ADR-025 result from ``export_onnx.run`` (``None``: not available)."""
    if parity is None:
        return {
            "rule": "challenger FP32 ONNX passes the ADR-025 parity check on the gate split",
            "value": None,
            "limit": 1.0,
            "passed": False,
            "reason": "no parity result",
        }
    batches_ok = all(b["passed"] for b in parity.get("batch_check", []))
    passed = bool(parity["passed"] and batches_ok)
    n = parity["n_images"]
    reason = (
        f"max tolerance ratio {parity['max_tolerance_ratio']:.3f} (<= 1 passes), same class "
        f"{parity['same_class']}/{n}, same routing {parity['same_routing']}/{n}, "
        f"batch sizes {'ok' if batches_ok else 'FAILED'}"
    )
    return {
        "rule": "challenger FP32 ONNX passes the ADR-025 parity check on the gate split",
        "value": parity["max_tolerance_ratio"],
        "limit": 1.0,
        "passed": passed,
        "reason": ("parity passed: " if passed else "parity FAILED: ") + reason,
    }


def evaluate(
    y: np.ndarray,
    champion_pred: np.ndarray,
    challenger_pred: np.ndarray,
    classes: Sequence[str],
    parity: Mapping[str, Any] | None,
    cfg: GateConfig,
) -> dict[str, Any]:
    """All four checks on one set of paired predictions; ``decision`` is promote only if all
    pass."""
    champ = model_metrics(y, champion_pred, classes)
    chall = model_metrics(y, challenger_pred, classes)
    table = paired_table(y, champion_pred, challenger_pred)
    checks = {
        "macro_f1": check_macro_f1(champ, chall, cfg),
        "class_recall": check_class_recall(champ, chall, cfg),
        "mcnemar": check_mcnemar(table, cfg),
        "onnx_parity": check_parity(parity),
    }
    promote = all(c["passed"] for c in checks.values())
    return {
        "decision": "promote" if promote else "reject",
        "failed_checks": [k for k in CHECKS if not checks[k]["passed"]],
        "checks": checks,
        "champion_metrics": champ,
        "challenger_metrics": chall,
        "table": {
            "both_right": table.both_right,
            "only_champion_right": table.a_only,
            "only_challenger_right": table.b_only,
            "both_wrong": table.both_wrong,
        },
        "n_images": int(len(y)),
    }


# ---------------------------------------------------------------- data guards and scoring


def feedback_paths(params: Mapping[str, Any]) -> set[str]:
    """Image paths of the current feedback export (empty if there is none)."""
    path = Path(params["review"]["export"]["out_dir"]) / "feedback.csv"
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        return {row["path"] for row in csv.DictReader(f)}


def check_no_feedback(paths: Sequence[str], params: Mapping[str, Any]) -> None:
    overlap = sorted(set(paths) & feedback_paths(params))
    if overlap:
        n, first = len(overlap), overlap[0]
        raise GateError(f"{n} gate images are feedback rows (ADR-033), e.g. {first}")


def score(checkpoint: Path, params: Mapping[str, Any], cfg: GateConfig) -> dict[str, Any]:
    """Labels, predictions, classes and image paths of one checkpoint on the gate split (CPU)."""
    import torch
    from torch.utils.data import DataLoader

    from src.serving.preprocess import PreprocessConfig
    from src.training.compare_val import load_checkpoint
    from src.training.dataset import SplitDataset
    from src.training.train import predict

    model, classes, _ = load_checkpoint(checkpoint)
    ds = SplitDataset(
        Path(params["data"]["split_file"]),
        Path(params["data"]["raw_dir"]),
        [cfg.split],
        classes,
        PreprocessConfig.from_params(params),
    )
    _, y, pred = predict(model, DataLoader(ds, batch_size=cfg.batch_size), torch.device("cpu"))
    return {"y": y, "pred": pred, "classes": classes, "paths": list(ds.paths)}


def onnx_parity(
    params: Mapping[str, Any], alias: str, checkpoint: Path, report: Path, *, log_mlflow: bool
) -> dict[str, Any]:
    """Export the challenger and return its parity result, passed or failed (a failed parity is
    a gate decision). Any other export error stops the gate."""
    from src.training import export_onnx as ex

    try:
        out = ex.run(
            params, alias=alias, checkpoint=checkpoint, report_file=report, log_mlflow=log_mlflow
        )
    except ex.ParityError as exc:
        return exc.parity
    except ex.ExportError as exc:
        raise GateError(f"ONNX export of @{alias} failed: {exc}") from exc
    return out["report"]["parity"]


# ---------------------------------------------------------------- registry


def resolve(params: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The ``@champion`` and ``@challenger`` versions; they must differ and carry serving tags."""
    from src.training import export_onnx as ex

    p = params["promotion"]
    try:
        champ = ex.registered_version(params, str(p["champion_alias"]))
        chall = ex.registered_version(params, str(p["challenger_alias"]))
    except ex.ExportError as exc:
        raise GateError(str(exc)) from exc
    if champ["version"] == chall["version"]:
        raise GateError(f"@champion and @challenger are both version {champ['version']}")
    return champ, chall


def verified_checkpoint(version: Mapping[str, Any], path: Path) -> Path:
    from src.training import export_onnx as ex

    try:
        return ex.checkpoint_at(version, path).checkpoint
    except ex.ExportError as exc:
        raise GateError(str(exc)) from exc


def apply_decision(
    client: Any,
    name: str,
    params: Mapping[str, Any],
    champ: Mapping[str, Any],
    chall: Mapping[str, Any],
    decision: str,
    gate_file: str,
    gate_sha256: str,
) -> None:
    """Tags first, aliases last. Refuses if either alias moved since the gate scored."""
    p = params["promotion"]
    champion_alias, challenger_alias = str(p["champion_alias"]), str(p["challenger_alias"])
    aliases = client.get_registered_model(name).aliases
    if str(aliases.get(champion_alias)) != champ["version"] or (
        str(aliases.get(challenger_alias)) != chall["version"]
    ):
        raise GateError("@champion or @challenger moved while the gate ran: run it again")
    missing = [t for t in SERVING_TAGS if not chall["tags"].get(t)]
    if decision == "promote" and missing:
        raise GateError(
            f"challenger v{chall['version']} passed the gate but is not servable (no {missing}): "
            "not promoted (ADR-034)"
        )
    tags = {
        "gate_decision": decision,
        "gate_decision_record": DECISION_RECORD,
        "gate_file": gate_file,
        "gate_file_sha256": gate_sha256,
    }
    if decision == "promote":
        tags["previous_champion_version"] = champ["version"]
    for key, value in tags.items():
        client.set_model_version_tag(name, chall["version"], key, value)
    if decision == "promote":
        client.set_registered_model_alias(name, champion_alias, chall["version"])
        client.delete_registered_model_alias(name, challenger_alias)


def log_entry(
    name: str,
    params: Mapping[str, Any],
    champ: Mapping[str, Any],
    chall: Mapping[str, Any],
    gate: Mapping[str, Any],
    gate_file: str,
    gate_sha256: str,
) -> dict[str, Any]:
    promoted = gate["decision"] == "promote"
    return {
        "time_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": gate["decision"],
        "decision_record": DECISION_RECORD,
        "registered_model": name,
        "alias": str(params["promotion"]["champion_alias"]),
        "version": chall["version"] if promoted else champ["version"],
        "previous_version": champ["version"] if promoted else None,
        "challenger_version": chall["version"],
        "challenger_run_id": chall["run_id"],
        "gate": "non-inferiority on val (ADR-033)",
        "checks": {k: gate["checks"][k]["passed"] for k in CHECKS},
        "reasons": {k: gate["checks"][k]["reason"] for k in CHECKS},
        "gate_file": gate_file,
        "gate_file_sha256": gate_sha256,
    }


# ---------------------------------------------------------------- run


def next_free(folder: Path, stem: str, suffix: str = ".json") -> Path:
    """``<stem>.json``, else ``<stem>.2.json``, ``<stem>.3.json``...: decision files are never
    overwritten (ADR-035)."""
    path, k = folder / f"{stem}{suffix}", 1
    while path.exists():
        k += 1
        path = folder / f"{stem}.{k}{suffix}"
    return path


def version_checkpoint(params: Mapping[str, Any], version: str) -> Path:
    """Where a registered version's checkpoint is kept (DVC-tracked, ADR-034)."""
    return Path(params["retrain"]["versions_dir"]) / f"v{version}" / "best.pt"


def default_champion_checkpoint(params: Mapping[str, Any], version: str) -> Path:
    """``models/versions/v<champion>/best.pt``, else the first model's local checkpoint."""
    kept = version_checkpoint(params, version)
    return kept if kept.is_file() else Path(params["compare_val"]["models"][0]["checkpoint"])


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data, indent=2) + "\n").encode())


def run(
    params: Mapping[str, Any],
    challenger_checkpoint: Path,
    champion_checkpoint: Path | None = None,
    *,
    do_apply: bool = False,
    record_only: bool = False,
) -> dict[str, Any]:
    """Dry run by default. ``record_only``: also write the gate file (``applied: false``) and
    nothing else (ADR-034). ``do_apply``: gate file, registry and promotion log (ADR-033)."""
    import mlflow

    from src.tracking import mlflow_tracking as mt
    from src.tracking.promote import append_log
    from src.training.train import git_commit, sha256_file

    cfg = GateConfig.from_params(params)
    champ, chall = resolve(params)
    if champion_checkpoint is None:
        champion_checkpoint = default_champion_checkpoint(params, champ["version"])
    champ_ckpt = verified_checkpoint(champ, Path(champion_checkpoint))
    chall_ckpt = verified_checkpoint(chall, Path(challenger_checkpoint))

    a, b = score(champ_ckpt, params, cfg), score(chall_ckpt, params, cfg)
    if a["classes"] != b["classes"]:
        raise GateError("champion and challenger use different class orders")
    if not (np.array_equal(a["y"], b["y"]) and a["paths"] == b["paths"]):
        raise GateError("champion and challenger were not scored on the same images")
    check_no_feedback(a["paths"], params)

    alias = str(params["promotion"]["challenger_alias"])
    export_params = copy.deepcopy(dict(params))
    export_params["train"] = {**params["train"], "val_split": cfg.split}
    if do_apply and not chall["tags"].get("onnx_sha256"):
        report = next_free(cfg.out_dir, f"onnx_parity_{chall['version']}")
        parity = onnx_parity(export_params, alias, chall_ckpt, report, log_mlflow=True)
    else:  # dry run, or the readiness step (ADR-035) already logged the ONNX file: check
        # parity again in a temporary folder and log nothing
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            export_params["export"] = {**params["export"], "out_dir": tmp}
            parity = onnx_parity(
                export_params, alias, chall_ckpt, Path(tmp) / "parity.json", log_mlflow=False
            )

    gate = evaluate(a["y"], a["pred"], b["pred"], a["classes"], parity, cfg)
    split_csv = Path(params["data"]["split_file"])
    commit = git_commit(REPO_ROOT)
    name = mt.TrackingConfig.from_params(params, smoke=False).registered_model  # as resolve()
    record = {
        "decision_record": DECISION_RECORD,
        "gate": "non-inferiority (ADR-033): val has too few images to show a significant gain",
        "decision": gate["decision"],
        "failed_checks": gate["failed_checks"],
        "time_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "registered_model": name,
        "champion": {k: champ[k] for k in ("version", "run_id")}
        | {"checkpoint_sha256": champ["tags"]["checkpoint_sha256"]},
        "challenger": {k: chall[k] for k in ("version", "run_id")}
        | {"checkpoint_sha256": chall["tags"]["checkpoint_sha256"]},
        "split": cfg.split,
        "split_file": split_csv.name,
        "split_sha256": sha256_file(split_csv),
        "n_images": gate["n_images"],
        "classes": a["classes"],
        "rules": cfg.rules(),
        "checks": gate["checks"],
        "champion_metrics": gate["champion_metrics"],
        "challenger_metrics": gate["challenger_metrics"],
        "table": gate["table"],
        "onnx_parity": parity,
        "note": "feedback labels come from a SIMULATED reviewer and a SIMULATED stream",
        "provenance": {"git_commit": commit["sha"], "git_dirty": commit["dirty"], "device": "cpu"},
        "applied": do_apply,
    }
    out: dict[str, Any] = {"record": record}
    gate_file = next_free(cfg.out_dir, f"gate_{chall['version']}")
    if not do_apply:
        if record_only:
            write_json(gate_file, record)
            out["gate_file"] = gate_file
        return out

    write_json(gate_file, record)
    rel = gate_file.as_posix()
    sha = sha256_file(gate_file)
    tracking = mt.TrackingConfig.from_params(params, smoke=False)
    client = mlflow.MlflowClient(tracking_uri=mt.resolve_tracking_uri(tracking.local_uri))
    apply_decision(client, name, params, champ, chall, gate["decision"], rel, sha)
    log_path = Path(params["promotion"]["log_file"])
    append_log(log_path, log_entry(name, params, champ, chall, gate, rel, sha))
    out.update({"gate_file": gate_file, "log_file": log_path})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # the ONNX exporter prints emoji (cp1252 consoles)
        stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--challenger-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--champion-checkpoint",
        type=Path,
        help="default: models/versions/v<champion>/best.pt, else compare_val.models[0].checkpoint",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="write files, registry and log")
    mode.add_argument("--record", action="store_true", help="write the gate file only")
    args = parser.parse_args(argv)
    try:
        out = run(
            load_params(args.config),
            args.challenger_checkpoint,
            args.champion_checkpoint,
            do_apply=args.apply,
            record_only=args.record,
        )
    except GateError as exc:
        print(f"gate: {exc}", file=sys.stderr)
        return 1
    r = out["record"]
    if args.apply:
        mode = "APPLIED"
    elif args.record:
        mode = "RECORDED (gate file only; registry and log unchanged)"
    else:
        mode = "DRY RUN (nothing written; pass --record or --apply)"
    print(
        f"{mode}: {r['decision'].upper()} challenger v{r['challenger']['version']} vs champion "
        f"v{r['champion']['version']} on {r['split']} ({r['n_images']} images, non-inferiority)"
    )
    for key in CHECKS:
        c = r["checks"][key]
        print(f"  [{'pass' if c['passed'] else 'FAIL'}] {key}: {c['reason']}")
    if "log_file" in out:
        print(f"  wrote {out['gate_file'].as_posix()}; appended to {out['log_file'].as_posix()}")
    elif "gate_file" in out:
        print(f"  wrote {out['gate_file'].as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
