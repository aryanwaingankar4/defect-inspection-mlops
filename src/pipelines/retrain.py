"""Retrain flow (TASK-028, ADR-034): feedback export -> Kaggle GPU -> @challenger -> gate record.

    make retrain                  # run or resume the cycle
    make retrain ARGS=--fresh     # forget the saved progress and start over

THE REVIEWER AND THE STREAM ARE SIMULATED in this project. Plain Python, one step after another;
progress is saved in ``retrain.state_dir/state.json`` so a rerun continues from the first
unfinished step (the laptop waits for Kaggle; an interrupted wait resumes).

Steps:
  preflight         clean pushed commit, clean split, @champion, committed export with at least
                    ``retrain.min_feedback_rows`` rows whose MD5 matches the CSV; the champion's
                    checkpoint is kept in ``models/versions/v<version>/``
  feedback_dataset  feedback.csv -> new version of the PRIVATE Kaggle feedback dataset
  push              launcher pinned to the commit, both private datasets, ``--feedback-md5``
  wait              poll the Kaggle run until it completes (wall time recorded)
  download          the output; it must come from the commit, the MD5 and its checkpoint hash
  register          publish + register the run (results to ``results/retrain/<run_uid>/``) and
                    keep the checkpoint in ``models/versions/v<version>/``
  calibrate         temperature + review threshold on val for the challenger (ADR-006/023)
                    (a T fit on a range bound inherits the champion's T, ADR-037)
  challenger        version tags, then ``@challenger`` -> the new version
  serving           FP32 ONNX + parity, ``serving_precision``, OOD threshold on val (ADR-035)
  versions          ``dvc add models/versions`` (commit models/versions.dvc, then ``make push``)
  gate              promotion gate in RECORD mode: gate file only, @champion never moves here
  record            ``results/retrain/cycle_<run_uid>.json``

``--fault-injection`` (TASK-030, ADR-036) runs the same steps on the BROKEN-reviewer copy of the
export (``make fault-export``): its own state folder and private Kaggle dataset, the version tagged
``fault_injection=true`` from registration on, the gate in record mode only (``gate --apply``
refuses such a version), and a last ``release`` step that gives ``@challenger`` back to its
previous holder (or removes it).

The Kaggle job holds no credentials (ADR-020): only the laptop talks to MLflow and Kaggle's API.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import load_params
from src.training.feedback import file_md5

STEPS = (
    "preflight",
    "feedback_dataset",
    "push",
    "wait",
    "download",
    "register",
    "calibrate",
    "challenger",
    "serving",
    "versions",
    "gate",
    "record",
)
FAULT_STEPS = (*STEPS, "release")  # ADR-036: the fault-injection run never keeps @challenger
ACTIVE = {"queued", "running", "new", "pending"}
FAILED = {"error", "cancelrequested", "cancelacknowledged", "cancelled", "canceled"}
DECISION_RECORD = "ADR-034"
FAULT_TAG = "fault_injection"  # same tag as src.tracking.gate.FAULT_TAG (ADR-036)


class RetrainError(RuntimeError):
    """The cycle stops here; fix the cause and rerun (it resumes)."""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    from src.training.train import sha256_file

    return sha256_file(path)


# ---------------------------------------------------------------- state


class State:
    """``state.json``: the cycle's facts and which steps are done."""

    def __init__(self, path: Path, data: dict[str, Any] | None = None) -> None:
        self.path = path
        self.data = data if data is not None else {"steps": {}}

    @classmethod
    def load(cls, path: Path) -> State:
        if path.is_file():
            return cls(path, json.loads(path.read_text(encoding="utf-8")))
        return cls(path)

    def done(self, step: str) -> bool:
        return step in self.data["steps"]

    def finish(self, step: str, **facts: Any) -> None:
        self.data.update(facts)
        self.data["steps"][step] = now()
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes((json.dumps(self.data, indent=2) + "\n").encode())

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


# ---------------------------------------------------------------- pure checks


def check_export(params: Mapping[str, Any]) -> dict[str, Any]:
    """The committed export report and its CSV: same MD5, enough rows (ADR-032/034)."""
    report_path = Path(params["review"]["export"]["report"])
    csv_path = Path(params["review"]["export"]["out_dir"]) / "feedback.csv"
    if not report_path.is_file() or not csv_path.is_file():
        raise RetrainError("no feedback export: run `make feedback-export` and commit it")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    md5 = file_md5(csv_path)
    if md5 != report["csv_md5"]:
        raise RetrainError("feedback.csv differs from the committed export report: re-export")
    minimum = int(params["retrain"]["min_feedback_rows"])
    if int(report["rows"]) < minimum:
        raise RetrainError(
            f"only {report['rows']} feedback rows (retrain.min_feedback_rows = {minimum})"
        )
    return {"csv": csv_path, "md5": md5, "rows": int(report["rows"]), "report": report}


def verify_output(metrics: Mapping[str, Any], checkpoint: Path, commit: str, md5: str) -> None:
    """The downloaded run must be THIS cycle's run (never an older Kaggle output)."""
    if metrics.get("smoke") is not False:
        raise RetrainError("the Kaggle output is a smoke run")
    got_commit = metrics.get("provenance", {}).get("git_commit")
    if got_commit != commit:
        raise RetrainError(f"the Kaggle output is from commit {got_commit}, not {commit}")
    if (metrics.get("feedback") or {}).get("md5") != md5:
        raise RetrainError("the Kaggle output was not trained on this feedback export (MD5)")
    if not checkpoint.is_file() or sha256(checkpoint) != metrics.get("checkpoint_sha256"):
        raise RetrainError("the downloaded checkpoint does not match its recorded SHA-256")


def wait_for_kaggle(
    status: Callable[[], str],
    *,
    poll_s: float,
    timeout_s: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    stale_polls: int = 3,
) -> str:
    """Poll until the run completes; returns the final status.

    Right after a push Kaggle can still report the previous run as complete, so ``complete`` is
    accepted only after an active status was seen, or after ``stale_polls`` polls (the download
    check then rejects an old output)."""
    deadline = clock() + timeout_s
    seen_active, polls = False, 0
    while True:
        sleep(poll_s)
        polls += 1
        s = status()
        if s in ACTIVE:
            seen_active = True
        elif s in FAILED:
            raise RetrainError(f"the Kaggle run ended with status {s}")
        elif s == "complete" and (seen_active or polls >= stale_polls):
            return s
        if clock() > deadline:
            raise RetrainError(f"Kaggle run not finished after {timeout_s / 60:.0f} min ({s})")


def fault_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """``params`` pointed at the fault-injection copy: its export, Kaggle dataset and state folder
    (ADR-036). The real export, ``neu-feedback-private`` and ``build/retrain`` are not touched."""
    f = params["loop"]["fault_injection"]
    p = copy.deepcopy(dict(params))
    p["review"]["export"] = {
        **p["review"]["export"],
        "out_dir": f["out_dir"],
        "report": f["report"],
    }
    p["kaggle"] = {
        **p["kaggle"],
        "feedback_dataset_slug": f["feedback_dataset_slug"],
        "feedback_dataset_title": f["feedback_dataset_title"],
        "feedback_staging_dir": f["feedback_staging_dir"],
    }
    p["retrain"] = {**p["retrain"], "state_dir": f["state_dir"]}
    return p


def challenger_tags(
    metrics: Mapping[str, Any],
    temperature: float,
    threshold: float,
    fault: bool = False,
    calibration: str = "fitted",
) -> dict:
    fb = metrics["feedback"]
    extra = {FAULT_TAG: "true"} if fault else {}
    return {
        "role": "challenger",
        "model": metrics["model"],
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "temperature": repr(float(temperature)),
        "review_threshold": repr(float(threshold)),
        "calibration": calibration,  # fitted | inherited_T (ADR-037)
        "feedback_md5": fb["md5"],
        "feedback_rows": str(fb["rows"]),
        "train_run_uid": metrics["run_uid"],
        "decision_record": DECISION_RECORD,
        **extra,
    }


def cycle_record(
    state: State, gate_record: Mapping[str, Any] | None, gate_file: str | None
) -> dict[str, Any]:
    d = state.data
    fault = bool(d.get("fault_injection"))
    note = "SIMULATED stream and SIMULATED reviewer (ADR-031); real Kaggle GPU training"
    if fault:
        note = (
            "FAULT INJECTION (ADR-036): trained on a BROKEN simulated reviewer's labels (every "
            "label shifted to the next class); never promoted. " + note
        )
    return {
        "decision_record": DECISION_RECORD,
        "fault_injection": fault,
        "note": note,
        "commit": d["commit"],
        "feedback": {"md5": d["feedback_md5"], "rows": d["feedback_rows"]},
        "feedback_per_class": d["feedback_per_class"],
        "kaggle": {
            "pushed_at": d["kaggle_pushed_at"],
            "completed_seen_at": d["kaggle_completed_at"],
            "wall_time_s": d["kaggle_wall_time_s"],
            "training_time_s": d["training_time_s"],
            "device": d["device"],
        },
        "run_uid": d["run_uid"],
        "n_train": d["n_train"],
        "champion_version": d["champion_version"],
        "challenger_version": d["version"],
        "challenger_val_macro_f1_at_training": d["train_val_macro_f1"],
        "calibration": {
            "source": d.get("calibration", "fitted"),
            "temperature": d.get("temperature"),
            "fitted_temperature": d.get("fitted_temperature"),
            "decision_record": "ADR-037" if d.get("calibration") == "inherited_T" else "ADR-006",
        },
        "gate": gate_block(d, gate_record, gate_file),
    }


STOPPED = "stopped_before_gate"
SKIPPED_WHEN_STOPPED = ("challenger", "serving", "gate")


def gate_block(
    d: Mapping[str, Any], gate_record: Mapping[str, Any] | None, gate_file: str | None
) -> dict[str, Any]:
    stop = d.get(STOPPED)
    if stop:  # fault run refused before the gate (ADR-036 amendment)
        return {
            "mode": "not reached: refused before the gate; the gate's checks were not evaluated",
            "decision": STOPPED,
            "failed_checks": [stop["check"]],
            "stopped_at": stop["step"],
            "reason": stop["reason"],
            "file": None,
        }
    assert gate_record is not None
    return {
        "mode": "record (applied: false); @champion not moved (ADR-034)",
        "decision": gate_record["decision"],
        "failed_checks": gate_record["failed_checks"],
        "champion_val_macro_f1": gate_record["champion_metrics"]["macro_f1"],
        "challenger_val_macro_f1": gate_record["challenger_metrics"]["macro_f1"],
        "file": gate_file,
    }


# ---------------------------------------------------------------- steps (side effects)


def step_preflight(params: Mapping[str, Any], state: State, fault: bool = False) -> None:
    from src.pipelines import kaggle_job as kj
    from src.training import export_onnx as ex

    kj.check_split_is_clean(params)
    commit = kj.pinned_commit()
    export = check_export(params)
    champ = ex.champion(params)
    kept = keep_champion_checkpoint(params, champ)
    state.finish(
        "preflight",
        commit=commit,
        feedback_md5=export["md5"],
        feedback_rows=export["rows"],
        feedback_per_class=export["report"]["per_class"],
        champion_version=champ["version"],
        champion_checkpoint=kept.as_posix(),
        fault_injection=fault,
    )


def keep_champion_checkpoint(params: Mapping[str, Any], champ: Mapping[str, Any]) -> Path:
    """``models/versions/v<champion>/best.pt``; copied once from the first model's checkpoint
    if that file is the champion's (SHA-256 tag)."""
    from src.tracking.gate import version_checkpoint

    kept = version_checkpoint(params, champ["version"])
    want = champ["tags"]["checkpoint_sha256"]
    if kept.is_file():
        if sha256(kept) != want:
            raise RetrainError(f"{kept} is not the @champion checkpoint (SHA-256)")
        return kept
    source = Path(params["compare_val"]["models"][0]["checkpoint"])
    if not source.is_file() or sha256(source) != want:
        raise RetrainError(f"no checkpoint of @champion v{champ['version']} on this laptop")
    kept.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, kept)
    return kept


def step_feedback_dataset(params: Mapping[str, Any], state: State, api: Any) -> None:
    from src.data import kaggle_dataset as kd

    csv_path = Path(params["review"]["export"]["out_dir"]) / "feedback.csv"
    if file_md5(csv_path) != state["feedback_md5"]:
        raise RetrainError("feedback.csv changed since preflight: run with --fresh")
    ref = kd.upload_feedback(api, dict(params), csv_path, state["feedback_md5"])
    state.finish("feedback_dataset", feedback_dataset=ref)


def step_push(params: Mapping[str, Any], state: State, api: Any) -> None:
    from src.pipelines import kaggle_job as kj

    args = kj.train_args(None, state["feedback_md5"])
    url = kj.push(api, params, state["commit"], args, [state["feedback_dataset"]])
    state.finish("push", kaggle_url=url, kaggle_pushed_at=now(), kaggle_pushed_ts=time.time())


def step_wait(params: Mapping[str, Any], state: State, api: Any) -> None:
    from src.pipelines import kaggle_job as kj

    r = params["retrain"]
    final = wait_for_kaggle(
        lambda: kj.status(api, params),
        poll_s=float(r["poll_seconds"]),
        timeout_s=float(r["timeout_minutes"]) * 60,
    )
    wall = time.time() - float(state["kaggle_pushed_ts"])
    state.finish(
        "wait", kaggle_status=final, kaggle_completed_at=now(), kaggle_wall_time_s=round(wall)
    )


def output_paths(params: Mapping[str, Any]) -> tuple[Path, Path]:
    k = params["kaggle"]
    out = Path(params["retrain"]["state_dir"]) / "kaggle-output"
    return out / k["output_results"], out / k["output_checkpoint"]


def step_download(params: Mapping[str, Any], state: State, api: Any) -> None:
    from src.pipelines import kaggle_job as kj

    out = Path(params["retrain"]["state_dir"]) / "kaggle-output"
    if out.exists():
        shutil.rmtree(out)
    kj.download_output(api, params, out)
    run_dir, ckpt = output_paths(params)
    metrics = json.loads((run_dir / "val_metrics.json").read_text(encoding="utf-8"))
    verify_output(metrics, ckpt, state["commit"], state["feedback_md5"])
    t = metrics["timing"]
    state.finish(
        "download",
        run_uid=metrics["run_uid"],
        n_train=metrics["n_train"],
        device=metrics["device"],
        train_val_macro_f1=metrics["val_macro_f1"],
        training_time_s=round((t["finished_at_ms"] - t["started_at_ms"]) / 1000),
    )


def step_register(params: Mapping[str, Any], state: State) -> None:
    from src.tracking import publish as pb
    from src.tracking.gate import version_checkpoint

    run_dir, ckpt = output_paths(params)
    record = pb.publish(params, run_dir, ckpt, role="candidate", register=True)
    version = str(record["registered_version"])
    if state.data.get("fault_injection"):  # tagged before anything else can use the version
        from src.tracking import mlflow_tracking as mt

        cfg = mt.TrackingConfig.from_params(params, smoke=False)
        mt.set_version_tags(cfg, version, {FAULT_TAG: "true"})
    results = Path(params["retrain"]["results_dir"]) / state["run_uid"]
    pb.copy_results(run_dir, results)
    kept = version_checkpoint(params, version)
    kept.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, kept)
    state.finish(
        "register",
        version=version,
        mlflow_run_id=record["run_id"],
        results_dir=results.as_posix(),
        challenger_checkpoint=kept.as_posix(),
    )


def challenger_params(
    params: Mapping[str, Any], model: str, checkpoint: Path | str, results_dir: Path | str
) -> dict[str, Any]:
    """``params`` with the challenger as the main model (calibrate, threshold and OOD read it
    there)."""
    p = copy.deepcopy(dict(params))
    main = {"name": model, "checkpoint": str(checkpoint), "results_dir": str(results_dir)}
    p["compare_val"]["models"] = [main, *p["compare_val"]["models"][1:]]
    p["train"] = {**p["train"], "model": model}
    return p


def champion_temperature(params: Mapping[str, Any], version: str) -> dict[str, Any]:
    """The champion's validated T (its ``temperature`` tag), for ADR-037."""
    from src.tracking import mlflow_tracking as mt

    cfg = mt.TrackingConfig.from_params(params, smoke=False)
    tags = mt.version_tags(cfg, version)
    if not tags.get("temperature"):
        raise RetrainError(f"@champion v{version} has no temperature tag")
    return {"version": str(version), "temperature": float(tags["temperature"])}


def step_calibrate(params: Mapping[str, Any], state: State) -> None:
    from src.training import calibrate as cal
    from src.training import threshold as th
    from src.training.compare_val import ModelSpec

    metrics = json.loads(
        (Path(state["results_dir"]) / "val_metrics.json").read_text(encoding="utf-8")
    )
    spec = ModelSpec(
        metrics["model"], Path(state["challenger_checkpoint"]), Path(state["results_dir"])
    )
    val_split = str(params["train"]["val_split"])
    inherit = champion_temperature(params, state["champion_version"])
    result = cal.calibrate_model(
        spec, params, cal.CalibConfig.from_params(params), val_split, inherit=inherit
    )
    if result["cal"]["at_bound"]:  # cannot happen with ``inherit``; kept as a guard
        raise RetrainError(f"challenger temperature {result['cal']['temperature']} on its bound")
    files = cal.write_outputs(result)
    cal.log_to_mlflow(params, spec, files)
    p = challenger_params(
        params, metrics["model"], state["challenger_checkpoint"], state["results_dir"]
    )
    try:
        thr = th.run(p, log_mlflow=True)
    except th.ThresholdError as exc:
        if not state.data.get("fault_injection"):
            raise RetrainError(f"review threshold refused (ADR-023): {exc}") from exc
        state.finish(  # ADR-036 amendment: the fault run's outcome, recorded
            "calibrate",
            calibration=result["cal"]["calibration"],
            fitted_temperature=float(result["cal"]["fit"]["temperature"]),
            temperature=float(result["cal"]["temperature"]),
            review_threshold=None,
            stopped_before_gate={
                "step": "calibrate",
                "check": "review_threshold_risk (ADR-023)",
                "reason": str(exc),
            },
        )
        return
    state.finish(
        "calibrate",
        calibration=result["cal"]["calibration"],
        fitted_temperature=float(result["cal"]["fit"]["temperature"]),
        temperature=float(result["cal"]["temperature"]),
        review_threshold=float(thr["result"]["threshold"]),
    )


def step_challenger(params: Mapping[str, Any], state: State) -> None:
    from src.tracking import mlflow_tracking as mt  # the only MLflow entry point

    cfg = mt.TrackingConfig.from_params(params, smoke=False)
    metrics = json.loads(
        (Path(state["results_dir"]) / "val_metrics.json").read_text(encoding="utf-8")
    )
    mt.set_version_tags(
        cfg,
        state["version"],
        challenger_tags(
            metrics,
            state["temperature"],
            state["review_threshold"],
            fault=bool(state.data.get("fault_injection")),
            calibration=state.data.get("calibration", "fitted"),
        ),
    )
    alias = str(params["promotion"]["challenger_alias"])
    before = mt.move_alias(cfg, alias, state["version"])  # tags first, alias last
    state.finish("challenger", previous_challenger=before)


SERVING_PRECISION = "fp32"  # challengers serve FP32 like v2 (ADR-026, ADR-035)


def prepare_serving(
    params: Mapping[str, Any], version: Mapping[str, Any], checkpoint: Path, results_dir: Path
) -> dict[str, Any]:
    """Make a registered version servable (ADR-035): FP32 ONNX with parity on val (tags only if
    it passes), ``serving_precision``, and the energy OOD threshold on val at its own T."""
    from src.tracking import mlflow_tracking as mt
    from src.training import export_onnx as ex
    from src.training import ood

    if not checkpoint.is_file() or sha256(checkpoint) != version["tags"]["checkpoint_sha256"]:
        raise RetrainError(f"{checkpoint} is not the checkpoint of v{version['version']}")
    model = str(version["tags"]["model"])
    p = challenger_params(params, model, checkpoint, results_dir)
    alias = str(version["alias"])
    try:
        export = ex.run(
            p, alias=alias, checkpoint=checkpoint, report_file=results_dir / "onnx_export.json",
            log_mlflow=True,
        )  # fmt: skip
    except ex.ExportError as exc:
        raise RetrainError(f"ONNX export of v{version['version']} failed: {exc}") from exc
    cfg = mt.TrackingConfig.from_params(params, smoke=False)
    mt.set_version_tags(cfg, version["version"], {"serving_precision": SERVING_PRECISION})
    try:
        out = ood.run(p, alias=alias, onnx_path=export["onnx"], log_mlflow=True)
    except ood.OodError as exc:
        raise RetrainError(f"OOD threshold of v{version['version']} failed: {exc}") from exc
    return {
        "onnx_sha256": export["report"]["onnx_sha256"],
        "parity_passed": export["report"]["parity"]["passed"],
        "ood_energy_threshold": out["result"]["ood_energy_threshold"],
        "ood_file": Path(out["out"]).as_posix(),
        "onnx_report": (results_dir / "onnx_export.json").as_posix(),
    }


def step_serving(params: Mapping[str, Any], state: State) -> None:
    from src.training import export_onnx as ex

    version = ex.registered_version(params, str(params["promotion"]["challenger_alias"]))
    if version["version"] != state["version"]:
        raise RetrainError(f"@challenger is v{version['version']}, not v{state['version']}")
    done = prepare_serving(
        params, version, Path(state["challenger_checkpoint"]), Path(state["results_dir"])
    )
    state.finish("serving", ood_energy_threshold=done["ood_energy_threshold"])


def step_versions(params: Mapping[str, Any], state: State) -> None:
    folder = str(params["retrain"]["versions_dir"])
    subprocess.run(["dvc", "add", folder], check=True)
    state.finish("versions")


def step_gate(params: Mapping[str, Any], state: State) -> None:
    from src.tracking import gate as gt

    out = gt.run(
        params,
        Path(state["challenger_checkpoint"]),
        Path(state["champion_checkpoint"]),
        record_only=True,
    )
    state.finish("gate", gate_file=Path(out["gate_file"]).as_posix())


def step_record(params: Mapping[str, Any], state: State) -> Path:
    gate_record, gate_file = None, None
    if not state.data.get(STOPPED):
        gate_file = Path(state["gate_file"]).as_posix()
        gate_record = json.loads(Path(gate_file).read_text(encoding="utf-8"))
    out = Path(params["retrain"]["results_dir"]) / f"cycle_{state['run_uid']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = cycle_record(state, gate_record, gate_file)
    out.write_bytes((json.dumps(data, indent=2) + "\n").encode())
    state.finish("record", cycle_file=out.as_posix())
    return out


def step_release(params: Mapping[str, Any], state: State) -> None:
    """Fault injection only: ``@challenger`` back to its previous holder, or removed (ADR-036)."""
    from src.tracking import mlflow_tracking as mt

    if state.data.get(STOPPED):  # @challenger was never moved to the fault version
        state.finish("release", challenger_after_release="never moved")
        return
    cfg = mt.TrackingConfig.from_params(params, smoke=False)
    alias = str(params["promotion"]["challenger_alias"])
    holder = mt.version_by_alias(cfg, alias)
    if holder is None or holder["version"] != state["version"]:
        raise RetrainError(f"@{alias} no longer points at fault version v{state['version']}")
    before = state.data.get("previous_challenger")
    if before:
        mt.move_alias(cfg, alias, before)
    else:
        mt.delete_alias(cfg, alias)
    state.finish("release", challenger_after_release=before)


# ---------------------------------------------------------------- run


def kaggle_api() -> Any:
    from src.pipelines.kaggle_job import _api

    return _api()


def run(
    params: Mapping[str, Any],
    *,
    fresh: bool = False,
    api_factory=kaggle_api,
    fault: bool = False,
) -> State:
    if fault:
        params = fault_params(params)
    state_path = Path(params["retrain"]["state_dir"]) / "state.json"
    if fresh and state_path.exists():
        state_path.unlink()
    state = State.load(state_path)
    if state.done("release" if fault else "record"):
        raise RetrainError(f"this cycle is finished ({state.data.get('cycle_file')}): use --fresh")
    api = None

    def kaggle() -> Any:
        nonlocal api
        api = api if api is not None else api_factory()
        return api

    steps: dict[str, Callable[[], Any]] = {
        "preflight": lambda: step_preflight(params, state, fault),
        "feedback_dataset": lambda: step_feedback_dataset(params, state, kaggle()),
        "push": lambda: step_push(params, state, kaggle()),
        "wait": lambda: step_wait(params, state, kaggle()),
        "download": lambda: step_download(params, state, kaggle()),
        "register": lambda: step_register(params, state),
        "calibrate": lambda: step_calibrate(params, state),
        "challenger": lambda: step_challenger(params, state),
        "serving": lambda: step_serving(params, state),
        "versions": lambda: step_versions(params, state),
        "gate": lambda: step_gate(params, state),
        "record": lambda: step_record(params, state),
        "release": lambda: step_release(params, state),
    }
    if state.data.get("fault_injection", fault) != fault:
        raise RetrainError("the saved progress is from another kind of run: use --fresh")
    for name in FAULT_STEPS if fault else STEPS:
        if state.done(name):
            print(f"[done] {name}", flush=True)
            continue
        if state.data.get(STOPPED) and name in SKIPPED_WHEN_STOPPED:
            print(f"[skip] {name} (stopped before the gate: {state[STOPPED]['check']})")
            state.finish(name)
            continue
        print(f"[....] {name}", flush=True)
        steps[name]()
        print(f"[ ok ] {name}", flush=True)
    return state


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # MLflow / the ONNX exporter print emoji
        stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--fresh", action="store_true", help="forget saved progress")
    parser.add_argument(
        "--fault-injection", action="store_true", help="BROKEN-reviewer run, never promoted"
    )
    args = parser.parse_args(argv)
    try:
        state = run(load_params(args.config), fresh=args.fresh, fault=args.fault_injection)
    except RetrainError as exc:
        print(f"retrain: {exc} (fix it and rerun: the flow resumes)", file=sys.stderr)
        return 1
    d = state.data
    if args.fault_injection:
        print("FAULT-INJECTION run (ADR-036): never promoted; @challenger released")
    if d.get(STOPPED):
        gate = f"STOPPED before the gate ({d[STOPPED]['check']}): {d[STOPPED]['reason']}"
    else:
        gate = f"gate (record only) -> {d['gate_file']}"
    print(
        f"SIMULATED-loop cycle done: {d['feedback_rows']} feedback rows -> challenger "
        f"v{d['version']}; Kaggle {d['kaggle_wall_time_s']} s; {gate}; "
        f"cycle record {d['cycle_file']}"
    )
    print("Commit results/ + models/versions.dvc and run `make push` (DVC).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
