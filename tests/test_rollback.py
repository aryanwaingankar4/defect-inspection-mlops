"""TASK-029 (ADR-035): rollback and re-gating on a local MLflow registry (no network)."""

import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("mlflow")

from mlflow import MlflowClient  # noqa: E402

from src.config import load_params  # noqa: E402
from src.tracking import rollback as rb  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
NOW = datetime(2026, 11, 12, 10, 0, 0, tzinfo=UTC)


def servable_tags(ckpt: Path) -> dict[str, str]:
    return {
        "model": "efficientnet_b0",
        "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "temperature": "1.5",
        "review_threshold": "0.8",
        "ood_energy_threshold": "-6.1",
        "serving_precision": "fp32",
        "onnx_sha256": "a" * 64,
        "onnx_artifact": "onnx/model.fp32.onnx",
    }


@pytest.fixture
def reg(offline_mlflow, tmp_path):
    """v1 and v2 servable; v2 holds @champion and was promoted over v1."""
    client = MlflowClient(tracking_uri=offline_mlflow)
    exp = client.create_experiment("rollback-test")
    name = PARAMS["tracking"]["registered_model"]
    client.create_registered_model(name)
    params = copy.deepcopy(PARAMS)
    params["retrain"]["versions_dir"] = str(tmp_path / "versions")
    params["promotion"]["log_file"] = str(tmp_path / "promotion_log.json")
    params["promotion"]["gate"]["out_dir"] = str(tmp_path / "promotion")
    for v in ("1", "2", "3"):
        run = client.create_run(exp)
        client.set_terminated(run.info.run_id)
        client.create_model_version(name, f"runs:/{run.info.run_id}/model", run.info.run_id)
        ckpt = tmp_path / "versions" / f"v{v}" / "best.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(f"weights {v}".encode())
        for k, val in servable_tags(ckpt).items():
            client.set_model_version_tag(name, v, k, val)
    client.set_model_version_tag(name, "2", "previous_champion_version", "1")
    client.set_registered_model_alias(name, "champion", "2")
    return {"client": client, "name": name, "params": params, "tmp": tmp_path}


def aliases(reg):
    return {k: str(v) for k, v in reg["client"].get_registered_model(reg["name"]).aliases.items()}


def log(reg):
    path = Path(reg["params"]["promotion"]["log_file"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def test_rollback_to_the_previous_champion(reg):
    out = rb.rollback(reg["params"], client=reg["client"], now=NOW)
    assert aliases(reg) == {"champion": "1"}
    assert out["file"].name == "rollback_20261112T100000Z.json"
    record = json.loads(out["file"].read_text(encoding="utf-8"))
    assert (record["from_version"], record["to_version"]) == ("2", "1")
    assert "previous_champion_version" in record["reason"] and record["checks"]["problems"] == []
    assert "serving-bundle" in record["next_step"]
    entry = log(reg)[-1]
    assert entry["decision"] == "rollback" and entry["decision_record"] == "ADR-035"
    assert (entry["version"], entry["previous_version"]) == ("1", "2")
    assert entry["rollback_file_sha256"] == hashlib.sha256(out["file"].read_bytes()).hexdigest()
    tags = reg["client"].get_model_version(reg["name"], "1").tags
    assert tags["rolled_back_to_at"] == record["time_utc"]


def test_rollback_to_an_explicit_version_and_files_are_never_overwritten(reg):
    first = rb.rollback(reg["params"], "3", client=reg["client"], now=NOW)
    second = rb.rollback(reg["params"], "2", client=reg["client"], now=NOW)
    assert aliases(reg) == {"champion": "2"}
    assert first["file"].name == "rollback_20261112T100000Z.json"
    assert second["file"].name == "rollback_20261112T100000Z.2.json"
    assert first["record"]["reason"] == "explicit VERSION" and len(log(reg)) == 2


def break_version(reg, version, what):
    client, name = reg["client"], reg["name"]
    ckpt = Path(reg["params"]["retrain"]["versions_dir"]) / f"v{version}" / "best.pt"
    if what == "checkpoint_missing":
        ckpt.unlink()
    elif what == "checkpoint_changed":
        ckpt.write_bytes(b"other weights")
    else:
        client.delete_model_version_tag(name, version, what)


@pytest.mark.parametrize(
    ("what", "message"),
    [
        ("checkpoint_missing", "no kept checkpoint"),
        ("checkpoint_changed", "does not match"),
        ("onnx_sha256", "ONNX parity"),
        ("onnx_artifact", "ONNX parity"),
        ("serving_precision", "serving_precision"),
        ("ood_energy_threshold", "ood_energy_threshold"),
        ("review_threshold", "review_threshold"),
        ("temperature", "temperature"),
    ],
)
def test_rollback_refuses_a_target_that_cannot_be_served(reg, what, message):
    break_version(reg, "1", what)
    with pytest.raises(rb.RollbackError, match=message):
        rb.rollback(reg["params"], client=reg["client"], now=NOW)
    assert aliases(reg) == {"champion": "2"} and log(reg) == []
    assert not Path(reg["params"]["promotion"]["gate"]["out_dir"]).exists()


@pytest.mark.parametrize(
    ("version", "message"),
    [("2", "already holds"), ("9", "no version 9")],
)
def test_rollback_refuses_bad_explicit_targets(reg, version, message):
    with pytest.raises(rb.RollbackError, match=message):
        rb.rollback(reg["params"], version, client=reg["client"], now=NOW)
    assert aliases(reg) == {"champion": "2"}


def test_rollback_needs_a_previous_version_or_an_explicit_one(reg):
    reg["client"].delete_model_version_tag(reg["name"], "2", "previous_champion_version")
    with pytest.raises(rb.RollbackError, match="pass VERSION"):
        rb.rollback(reg["params"], client=reg["client"], now=NOW)


def test_set_challenger_for_a_new_gate(reg):
    out = rb.set_challenger(reg["params"], "3", client=reg["client"])
    assert out["previous_challenger"] is None and aliases(reg) == {
        "champion": "2",
        "challenger": "3",
    }
    with pytest.raises(rb.RollbackError, match="holds @champion"):
        rb.set_challenger(reg["params"], "2", client=reg["client"])
    break_version(reg, "1", "ood_energy_threshold")
    with pytest.raises(rb.RollbackError, match="cannot be served"):
        rb.set_challenger(reg["params"], "1", client=reg["client"])
    assert aliases(reg)["challenger"] == "3"


def test_readiness_uses_the_serving_bundle_tags():
    from src.tracking.serving_bundle import VALUE_TAGS

    assert set(VALUE_TAGS) == {"temperature", "review_threshold", "ood_energy_threshold"}


def test_rollback_and_regating_refuse_a_fault_injection_version(reg):
    reg["client"].set_model_version_tag(reg["name"], "1", "fault_injection", "true")
    with pytest.raises(rb.RollbackError, match="fault-injection"):
        rb.rollback(reg["params"], client=reg["client"], now=NOW)
    with pytest.raises(rb.RollbackError, match="fault-injection"):
        rb.set_challenger(reg["params"], "1", client=reg["client"])
    assert aliases(reg) == {"champion": "2"} and log(reg) == []
