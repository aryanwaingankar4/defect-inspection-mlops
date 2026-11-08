"""Set the first ``@champion`` with its evidence linked (TASK-014, ADR-024).

    python -m src.tracking.promote --config configs/params.yaml [--apply]

Dry run by default: prints what would change. ``--apply`` writes to the registry (public on
DagsHub: text only, no images) and appends the decision to ``promotion.log_file``.

Local checks first (nothing talks to MLflow before they pass):
  - the ADR-022 model choice is settled for this split, main model first;
  - ``calibration.json`` and ``threshold.json`` match the main checkpoint and split;
  - the test report used the same checkpoint, temperature and threshold, on a clean tree;
  - the evidence files are committed, unchanged, and their commit is on the remote (links).
Then in the registry:
  - the version is found by the run id in ``mlflow_run.json`` (never a typed number), is
    READY, and its run carries the same checkpoint hash and training run id;
  - the run's copies of the evidence files have the same SHA-256 as the local files;
  - ``@champion`` is free (set it), already on this version (no change), or on another
    version (refuse: moving it is the promotion gate's job, ADR-005).
Tags and description are written before the alias, so the alias never points at a version
without its evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow

from src.config import load_params
from src.tracking import mlflow_tracking as mt
from src.tracking.publish import RUN_UID_TAG
from src.training.evaluate import (
    EvalError,
    check_model_choice,
    load_temperature,
    load_threshold,
)
from src.training.train import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DECISION_RECORDS = "ADR-006,ADR-021,ADR-022,ADR-023,ADR-024"
# evidence name -> artifact path inside the training run (logged by calibrate/threshold/evaluate)
RUN_ARTIFACTS = {
    "calibration": "calibration/calibration.json",
    "threshold": "threshold/threshold.json",
    "test_metrics": "eval/test_metrics.json",
}


class PromotionError(RuntimeError):
    """The alias must not be set (unsettled choice, mismatched evidence, alias taken)."""


# ---------------------------------------------------------------- local evidence


def _git(args: Sequence[str], root: Path) -> str:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def evidence_commit(paths: Sequence[Path], root: Path = REPO_ROOT) -> str:
    """Last commit touching ``paths``; they must be unchanged and the commit pushed."""
    rel = [p.relative_to(root).as_posix() for p in paths]
    if _git(["status", "--porcelain", "--", *rel], root):
        raise PromotionError("evidence files have uncommitted changes: commit them first")
    sha = _git(["log", "-1", "--format=%H", "--", *rel], root)
    if not sha:
        raise PromotionError("evidence files are not committed")
    if not _git(["branch", "-r", "--contains", sha], root):
        raise PromotionError(f"commit {sha[:7]} is not on the remote: push it first (links)")
    return sha


def check_test_report(
    report: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    split_sha256: str,
    temperature: float,
    threshold: float,
) -> None:
    """The test report must belong to exactly this checkpoint, split, T and threshold."""
    prov = report.get("provenance", {})
    if report.get("role") != "main":
        raise PromotionError("the test report is not for the main model")
    if prov.get("checkpoint_sha256") != checkpoint_sha256:
        raise PromotionError("the test report is for another checkpoint")
    if report.get("split_sha256") != split_sha256:
        raise PromotionError("the test report is for another split")
    if report.get("calibration", {}).get("temperature") != temperature:
        raise PromotionError("the test report used another temperature")
    if report.get("review", {}).get("threshold") != threshold:
        raise PromotionError("the test report used another review threshold")
    if prov.get("git_dirty") is not False:
        raise PromotionError("the test report was made from a dirty tree")


def collect_evidence(params: Mapping[str, Any], root: Path = REPO_ROOT) -> dict[str, Any]:
    """Every local check; returns what will be linked on the registry version."""
    try:
        spec = check_model_choice(params, root)[0]
        temperature, cal_path = load_temperature(params, spec, root)
        threshold, thr_path = load_threshold(params, spec, cal_path, root)
    except EvalError as exc:
        raise PromotionError(str(exc)) from exc
    res = root / spec.results_dir
    val = json.loads((res / "val_metrics.json").read_text(encoding="utf-8"))
    pointer = json.loads((res / "mlflow_run.json").read_text(encoding="utf-8"))
    split_csv = root / params["data"]["split_file"]
    test_path = root / params["evaluate"]["out_dir"] / spec.name / "test_metrics.json"
    if not test_path.is_file():
        raise PromotionError(f"no test report for {spec.name}: run `make evaluate` first")
    test = json.loads(test_path.read_text(encoding="utf-8"))
    check_test_report(
        test,
        checkpoint_sha256=val["checkpoint_sha256"],
        split_sha256=sha256_file(split_csv),
        temperature=temperature,
        threshold=threshold,
    )
    files = {"calibration": cal_path, "threshold": thr_path, "test_metrics": test_path}
    thr = json.loads(thr_path.read_text(encoding="utf-8"))
    return {
        "model": spec.name,
        "run_id": pointer["run_id"],
        "run_url": pointer.get("run_url", ""),
        "train_run_uid": val["run_uid"],
        "checkpoint_sha256": val["checkpoint_sha256"],
        "split_file": split_csv.name,
        "split_sha256": sha256_file(split_csv),
        "temperature": temperature,
        "threshold": threshold,
        "files": {k: p.relative_to(root).as_posix() for k, p in files.items()},
        "sha256": {k: sha256_file(p) for k, p in files.items()},
        "results_commit": evidence_commit(list(files.values()), root),
        "test": test,
        "val_review": thr["val"],
    }


# ---------------------------------------------------------------- what is written


def version_tags(ev: Mapping[str, Any]) -> dict[str, str]:
    tags = {
        "role": "main",
        "model": ev["model"],
        "split_file": ev["split_file"],
        "split_sha256": ev["split_sha256"],
        "checkpoint_sha256": ev["checkpoint_sha256"],
        "temperature": repr(ev["temperature"]),
        "review_threshold": repr(ev["threshold"]),
        "results_commit": ev["results_commit"],
        "decision_records": DECISION_RECORDS,
    }
    for name, artifact in RUN_ARTIFACTS.items():
        tags[f"{name}_sha256"] = ev["sha256"][name]
        tags[f"{name}_artifact"] = artifact
    return tags


def _ci(m: Mapping[str, float]) -> str:
    return f"{m['value']:.4f} [{m['ci_low']:.4f}, {m['ci_high']:.4f}]"


def version_description(ev: Mapping[str, Any], code_url: str) -> str:
    test, rv, vr = ev["test"], ev["test"]["review"], ev["val_review"]
    blob = f"{code_url.rstrip('/')}/blob/{ev['results_commit']}"
    links = ", ".join(f"[{k}]({blob}/{p})" for k, p in ev["files"].items())
    return (
        f"{ev['model']} trained on `{ev['split_file']}` (group-aware split, ADR-021); main "
        "model chosen on val (ADR-022). First `@champion` (ADR-024).\n\n"
        f"Serving: probabilities = softmax(logits / T) with T = {ev['temperature']:.4f} (tag "
        "`temperature`, ADR-006); an image goes to review if its calibrated confidence is "
        f"below {ev['threshold']:.4f} (tag `review_threshold`, ADR-023).\n\n"
        f"Val (sets the threshold): {vr['n_routed']} of {vr['n_images']} images routed; "
        f"95% upper bound on risk among accepted images {vr['risk_ucb']['value']:.1%}.\n\n"
        f"test_frozen (report only, {test['n_images']} images, 95% intervals): accuracy "
        f"{_ci(test['accuracy'])}, macro-F1 {_ci(test['macro_f1'])}; at the threshold "
        f"{rv['n_routed']} routed, 95% upper bound on risk among accepted images "
        f"{rv['risk_ucb']['value']:.1%}.\n\n"
        f"Evidence (commit {ev['results_commit'][:7]}): {links}. "
        f"MLflow run: {ev['run_url'] or ev['run_id']}.\n\n"
        "In this project the reviewer and the drift are simulated."
    )


MODEL_DESCRIPTION = (
    "Steel surface defect classifier (NEU, 6 classes) with calibrated confidence and a human "
    "review threshold. `@champion` is the version that serves predictions; only the "
    "promotion gate (ADR-005) moves it after the first one (ADR-024). Version 1 was trained "
    "on a leaky split and never serves (ADR-021)."
)


# ---------------------------------------------------------------- registry


def find_version(client: mlflow.MlflowClient, name: str, run_id: str) -> Any:
    versions = [v for v in client.search_model_versions(f"name = '{name}'") if v.run_id == run_id]
    if len(versions) != 1:
        raise PromotionError(
            f"expected one version of {name} from run {run_id}, found {len(versions)}"
        )
    if versions[0].status != "READY":
        raise PromotionError(f"version {versions[0].version} is {versions[0].status}, not READY")
    return versions[0]


def check_run(client: mlflow.MlflowClient, ev: Mapping[str, Any]) -> None:
    """The run must be the training run of this checkpoint and hold the same evidence files."""
    tags = client.get_run(ev["run_id"]).data.tags
    if tags.get("checkpoint_sha256") != ev["checkpoint_sha256"]:
        raise PromotionError("the MLflow run is for another checkpoint")
    if tags.get(RUN_UID_TAG) != ev["train_run_uid"]:
        raise PromotionError("the MLflow run is for another training run")
    with tempfile.TemporaryDirectory() as tmp:
        for name, artifact in RUN_ARTIFACTS.items():
            try:
                local = client.download_artifacts(ev["run_id"], artifact, tmp)
            except Exception as exc:  # noqa: BLE001 - any failure means the file is missing
                raise PromotionError(f"the MLflow run has no {artifact}") from exc
            if sha256_file(Path(local)) != ev["sha256"][name]:
                raise PromotionError(f"{artifact} in the MLflow run differs from the local file")


def plan(client: mlflow.MlflowClient, name: str, alias: str, ev: Mapping[str, Any]) -> dict:
    """What would change. ``action``: 'set' (alias free) or 'none' (already this version)."""
    version = find_version(client, name, ev["run_id"])
    check_run(client, ev)
    holder = client.get_registered_model(name).aliases.get(alias)
    if holder is not None and str(holder) != str(version.version):
        raise PromotionError(
            f"@{alias} is on version {holder}: moving it is the promotion gate's job (ADR-005)"
        )
    return {
        "version": str(version.version),
        "action": "none" if holder is not None else "set",
        "current_tags": dict(version.tags),
    }


def apply(
    client: mlflow.MlflowClient,
    name: str,
    alias: str,
    version: str,
    tags: Mapping[str, str],
    description: str,
) -> None:
    """Evidence first, alias last."""
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, value)
    client.update_model_version(name, version, description=description)
    if not client.get_registered_model(name).description:
        client.update_registered_model(name, description=MODEL_DESCRIPTION)
    client.set_registered_model_alias(name, alias, version)


def append_log(path: Path, entry: Mapping[str, Any]) -> None:
    log = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    log.append(dict(entry))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(log, indent=2) + "\n").encode())


def log_entry(name: str, alias: str, version: str, ev: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "time_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision": "initial_champion",
        "decision_record": "ADR-024",
        "registered_model": name,
        "alias": alias,
        "version": version,
        "previous_version": None,
        "gate": "not applied: no previous champion to compare with (ADR-024)",
        "run_id": ev["run_id"],
        "model": ev["model"],
        "checkpoint_sha256": ev["checkpoint_sha256"],
        "split_file": ev["split_file"],
        "temperature": ev["temperature"],
        "review_threshold": ev["threshold"],
        "evidence": {k: {"file": ev["files"][k], "sha256": ev["sha256"][k]} for k in RUN_ARTIFACTS},
        "results_commit": ev["results_commit"],
    }


def run(params: Mapping[str, Any], *, do_apply: bool = False) -> dict[str, Any]:
    ev = collect_evidence(params)  # local only
    p = params["promotion"]
    cfg = mt.TrackingConfig.from_params(params, smoke=False)
    name, alias = cfg.registered_model, str(p["champion_alias"])
    client = mlflow.MlflowClient(tracking_uri=mt.resolve_tracking_uri(cfg.local_uri))
    planned = plan(client, name, alias, ev)
    tags = version_tags(ev)
    description = version_description(ev, str(p["code_url"]))
    out = {"name": name, "alias": alias, **planned, "tags": tags, "description": description}
    if do_apply and planned["action"] == "set":
        apply(client, name, alias, planned["version"], tags, description)
        log_path = REPO_ROOT / p["log_file"]
        append_log(log_path, log_entry(name, alias, planned["version"], ev))
        out["log_file"] = log_path
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument(
        "--apply", action="store_true", help="write to the registry (default: dry run)"
    )
    args = parser.parse_args(argv)
    params = load_params(args.config)
    try:
        out = run(params, do_apply=args.apply)
    except PromotionError as exc:
        print(f"promote: {exc}", file=sys.stderr)
        return 1
    mode = "APPLIED" if args.apply else "DRY RUN (nothing written; pass --apply)"
    print(f"{mode}: {out['name']} version {out['version']}")
    print("  local checks passed; run, checkpoint and evidence files match the registry")
    if out["action"] == "none":
        print(f"  @{out['alias']} is already on version {out['version']}: no change")
        return 0
    verb = "set" if args.apply else "would set"
    print(f"  {verb} @{out['alias']} -> version {out['version']} (no version holds it now)")
    print(f"  {verb} version tags:")
    for key, value in out["tags"].items():
        print(f"    {key} = {value}")
    print(f"  {verb} version description:")
    for line in out["description"].splitlines():
        print(f"    {line}" if line else "")
    if "log_file" in out:
        print(f"  appended the decision to {out['log_file'].relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
