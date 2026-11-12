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
  challenger        version tags, then ``@challenger`` -> the new version
  versions          ``dvc add models/versions`` (commit models/versions.dvc, then ``make push``)
  gate              promotion gate in RECORD mode: gate file only, @champion never moves here
  record            ``results/retrain/cycle_<run_uid>.json``

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
    "versions",
    "gate",
    "record",
)
ACTIVE = {"queued", "running", "new", "pending"}
FAILED = {"error", "cancelrequested", "cancelacknowledged", "cancelled", "canceled"}
DECISION_RECORD = "ADR-034"


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


def challenger_tags(metrics: Mapping[str, Any], temperature: float, threshold: float) -> dict:
    fb = metrics["feedback"]
    return {
        "role": "challenger",
        "model": metrics["model"],
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "temperature": repr(float(temperature)),
        "review_threshold": repr(float(threshold)),
        "feedback_md5": fb["md5"],
        "feedback_rows": str(fb["rows"]),
        "train_run_uid": metrics["run_uid"],
        "decision_record": DECISION_RECORD,
    }


def cycle_record(state: State, gate_record: Mapping[str, Any], gate_file: str) -> dict[str, Any]:
    d = state.data
    return {
        "decision_record": DECISION_RECORD,
        "note": "SIMULATED stream and SIMULATED reviewer (ADR-031); real Kaggle GPU training",
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
        "gate": {
            "mode": "record (applied: false); @champion not moved (ADR-034)",
            "decision": gate_record["decision"],
            "failed_checks": gate_record["failed_checks"],
            "champion_val_macro_f1": gate_record["champion_metrics"]["macro_f1"],
            "challenger_val_macro_f1": gate_record["challenger_metrics"]["macro_f1"],
            "file": gate_file,
        },
    }


# ---------------------------------------------------------------- steps (side effects)


def step_preflight(params: Mapping[str, Any], state: State) -> None:
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


def challenger_params(params: Mapping[str, Any], state: State, model: str) -> dict[str, Any]:
    """``params`` with the challenger as the main model (calibrate/threshold read it there)."""
    p = copy.deepcopy(dict(params))
    main = {
        "name": model,
        "checkpoint": state["challenger_checkpoint"],
        "results_dir": state["results_dir"],
    }
    p["compare_val"]["models"] = [main, *p["compare_val"]["models"][1:]]
    p["train"] = {**p["train"], "model": model}
    return p


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
    result = cal.calibrate_model(spec, params, cal.CalibConfig.from_params(params), val_split)
    if result["cal"]["at_bound"]:
        raise RetrainError(f"challenger temperature {result['cal']['temperature']} on its bound")
    files = cal.write_outputs(result)
    cal.log_to_mlflow(params, spec, files)
    thr = th.run(challenger_params(params, state, metrics["model"]), log_mlflow=True)
    state.finish(
        "calibrate",
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
        challenger_tags(metrics, state["temperature"], state["review_threshold"]),
    )
    alias = str(params["promotion"]["challenger_alias"])
    before = mt.move_alias(cfg, alias, state["version"])  # tags first, alias last
    state.finish("challenger", previous_challenger=before)


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
    gate_file = Path(state["gate_file"])
    gate_record = json.loads(gate_file.read_text(encoding="utf-8"))
    out = Path(params["retrain"]["results_dir"]) / f"cycle_{state['run_uid']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = cycle_record(state, gate_record, gate_file.as_posix())
    out.write_bytes((json.dumps(data, indent=2) + "\n").encode())
    state.finish("record", cycle_file=out.as_posix())
    return out


# ---------------------------------------------------------------- run


def kaggle_api() -> Any:
    from src.pipelines.kaggle_job import _api

    return _api()


def run(params: Mapping[str, Any], *, fresh: bool = False, api_factory=kaggle_api) -> State:
    state_path = Path(params["retrain"]["state_dir"]) / "state.json"
    if fresh and state_path.exists():
        state_path.unlink()
    state = State.load(state_path)
    if state.done("record"):
        raise RetrainError(f"this cycle is finished ({state['cycle_file']}): use --fresh")
    api = None

    def kaggle() -> Any:
        nonlocal api
        api = api if api is not None else api_factory()
        return api

    steps: dict[str, Callable[[], Any]] = {
        "preflight": lambda: step_preflight(params, state),
        "feedback_dataset": lambda: step_feedback_dataset(params, state, kaggle()),
        "push": lambda: step_push(params, state, kaggle()),
        "wait": lambda: step_wait(params, state, kaggle()),
        "download": lambda: step_download(params, state, kaggle()),
        "register": lambda: step_register(params, state),
        "calibrate": lambda: step_calibrate(params, state),
        "challenger": lambda: step_challenger(params, state),
        "versions": lambda: step_versions(params, state),
        "gate": lambda: step_gate(params, state),
        "record": lambda: step_record(params, state),
    }
    for name in STEPS:
        if state.done(name):
            print(f"[done] {name}", flush=True)
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
    args = parser.parse_args(argv)
    try:
        state = run(load_params(args.config), fresh=args.fresh)
    except RetrainError as exc:
        print(f"retrain: {exc} (fix it and rerun: the flow resumes)", file=sys.stderr)
        return 1
    d = state.data
    print(
        f"SIMULATED-loop cycle done: {d['feedback_rows']} feedback rows -> challenger "
        f"v{d['version']}; Kaggle {d['kaggle_wall_time_s']} s; gate (record only) -> "
        f"{d['gate_file']}; cycle record {d['cycle_file']}"
    )
    print("Commit results/ + models/versions.dvc and run `make push` (DVC).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
