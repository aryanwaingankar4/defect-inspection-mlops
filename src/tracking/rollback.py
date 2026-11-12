"""Rollback (TASK-029, ADR-035): move ``@champion`` back, only to a version that can be served.

    make rollback                 # to the current champion's previous_champion_version tag
    make rollback VERSION=2       # to an explicit version
    make challenger VERSION=3     # point @challenger at a version again (to re-run the gate)

A target must have its kept checkpoint (``models/versions/v<v>/best.pt``, matching its
``checkpoint_sha256`` tag), its ONNX parity result (the ONNX file tags of its ``serving_precision``,
written only after parity passed, ADR-025) and ``temperature``, ``review_threshold`` and
``ood_energy_threshold``: the tags ``make serving-bundle`` needs. Otherwise nothing changes.

A rollback writes ``results/promotion/rollback_<UTC time>.json``, tags the target, moves the alias
and appends to the promotion log. It does not redeploy: it prints the next step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow

from src.config import load_params
from src.tracking import mlflow_tracking as mt
from src.tracking.gate import is_fault_injection, next_free, version_checkpoint
from src.tracking.promote import append_log
from src.tracking.serving_bundle import FILE_TAGS, VALUE_TAGS

DECISION_RECORD = "ADR-035"
NEXT_STEP = "make serving-bundle && make docker-smoke  (then deploy the new image)"


class RollbackError(RuntimeError):
    """Nothing was changed."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def readiness(params: Mapping[str, Any], version: str, tags: Mapping[str, str]) -> dict[str, Any]:
    """Each check with its result; ``problems`` lists the failures (empty = servable)."""
    problems = []
    if is_fault_injection(tags):
        problems.append("it is a fault-injection version (fault_injection=true, ADR-036)")
    ckpt = version_checkpoint(params, version)
    if not ckpt.is_file():
        problems.append(f"no kept checkpoint {ckpt.as_posix()}")
    elif _sha256(ckpt) != tags.get("checkpoint_sha256"):
        problems.append(f"{ckpt.as_posix()} does not match the checkpoint_sha256 tag")
    precision = tags.get("serving_precision")
    if precision not in FILE_TAGS:
        problems.append("no valid serving_precision tag")
        file_tags: tuple[str, ...] = ()
    else:
        file_tags = FILE_TAGS[precision]
        missing = [t for t in file_tags if not tags.get(t)]
        if missing:
            problems.append(f"no ONNX parity result (missing tags {missing})")
    missing_values = [t for t in VALUE_TAGS if not tags.get(t)]
    if missing_values:
        problems.append(f"missing tags {missing_values}")
    return {
        "checkpoint": ckpt.as_posix(),
        "serving_precision": precision,
        "onnx_tags": {t: tags.get(t) for t in file_tags},
        "values": {t: tags.get(t) for t in VALUE_TAGS},
        "problems": problems,
    }


def _client(params: Mapping[str, Any]) -> tuple[Any, str]:
    cfg = mt.TrackingConfig.from_params(params, smoke=False)
    client = mlflow.MlflowClient(tracking_uri=mt.resolve_tracking_uri(cfg.local_uri))
    return client, cfg.registered_model


def _tags(client: Any, name: str, version: str) -> dict[str, str]:
    try:
        return dict(client.get_model_version(name, version).tags)
    except mlflow.exceptions.MlflowException as exc:
        raise RollbackError(f"{name} has no version {version}") from exc


def plan(params: Mapping[str, Any], client: Any, name: str, version: str | None) -> dict:
    """Current champion, target and why; refuses an unservable or identical target."""
    alias = str(params["promotion"]["champion_alias"])
    current = client.get_registered_model(name).aliases.get(alias)
    if current is None:
        raise RollbackError(f"no @{alias} to roll back")
    current = str(current)
    if version is None:
        target = _tags(client, name, current).get("previous_champion_version")
        if not target:
            raise RollbackError(
                f"@{alias} v{current} has no previous_champion_version tag: pass VERSION="
            )
        reason = f"previous_champion_version tag on v{current}"
    else:
        target, reason = str(version), "explicit VERSION"
    if target == current:
        raise RollbackError(f"v{target} already holds @{alias}")
    checks = readiness(params, target, _tags(client, name, target))
    if checks["problems"]:
        raise RollbackError(f"v{target} cannot be served: {'; '.join(checks['problems'])}")
    return {"alias": alias, "from": current, "to": target, "reason": reason, "checks": checks}


def rollback(
    params: Mapping[str, Any],
    version: str | None = None,
    *,
    client: Any = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    if client is None:
        client, name = _client(params)
    else:
        name = mt.TrackingConfig.from_params(params, smoke=False).registered_model
    p = plan(params, client, name, version)
    at = now or datetime.now(UTC)
    record = {
        "decision_record": DECISION_RECORD,
        "action": "rollback",
        "time_utc": at.isoformat(timespec="seconds"),
        "registered_model": name,
        "alias": p["alias"],
        "from_version": p["from"],
        "to_version": p["to"],
        "reason": p["reason"],
        "checks": p["checks"],
        "next_step": NEXT_STEP,
        "note": "rollback moves the alias only; the serving bundle and image are rebuilt next",
    }
    out_dir = Path(params["promotion"]["gate"]["out_dir"])
    out = next_free(out_dir, f"rollback_{at.strftime('%Y%m%dT%H%M%SZ')}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((json.dumps(record, indent=2) + "\n").encode())
    sha = _sha256(out)
    client.set_model_version_tag(name, p["to"], "rolled_back_to_at", record["time_utc"])
    client.set_registered_model_alias(name, p["alias"], p["to"])  # tags first, alias last
    entry = {
        "time_utc": record["time_utc"],
        "decision": "rollback",
        "decision_record": DECISION_RECORD,
        "registered_model": name,
        "alias": p["alias"],
        "version": p["to"],
        "previous_version": p["from"],
        "reason": p["reason"],
        "rollback_file": out.as_posix(),
        "rollback_file_sha256": sha,
    }
    append_log(Path(params["promotion"]["log_file"]), entry)
    return {"record": record, "file": out}


def set_challenger(params: Mapping[str, Any], version: str, *, client: Any = None) -> dict:
    """Point ``@challenger`` at a servable version that is not the champion (to re-gate it)."""
    if client is None:
        client, name = _client(params)
    else:
        name = mt.TrackingConfig.from_params(params, smoke=False).registered_model
    p = params["promotion"]
    aliases = client.get_registered_model(name).aliases
    if str(aliases.get(str(p["champion_alias"]))) == str(version):
        raise RollbackError(f"v{version} holds @{p['champion_alias']}: nothing to gate")
    checks = readiness(params, str(version), _tags(client, name, str(version)))
    if checks["problems"]:
        raise RollbackError(f"v{version} cannot be served: {'; '.join(checks['problems'])}")
    before = aliases.get(str(p["challenger_alias"]))
    client.set_registered_model_alias(name, str(p["challenger_alias"]), str(version))
    return {"version": str(version), "previous_challenger": before, "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--version", help="target version (default: previous_champion_version)")
    parser.add_argument("--set-challenger", metavar="VERSION", help="point @challenger instead")
    args = parser.parse_args(argv)
    params = load_params(args.config)
    try:
        if args.set_challenger:
            out = set_challenger(params, args.set_challenger)
            print(f"@challenger -> v{out['version']} (was {out['previous_challenger']})")
            print("  next: make gate ARGS=\"--challenger-checkpoint models/versions/v"
                  f"{out['version']}/best.pt --apply\"")  # fmt: skip
            return 0
        out = rollback(params, args.version)
    except RollbackError as exc:
        print(f"rollback: {exc} (nothing changed)", file=sys.stderr)
        return 1
    r = out["record"]
    print(f"ROLLED BACK @{r['alias']}: v{r['from_version']} -> v{r['to_version']} ({r['reason']})")
    print(f"  wrote {out['file'].as_posix()}; appended to {params['promotion']['log_file']}")
    print(f"  next step (not done here): {NEXT_STEP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
