"""TASK-014: first @champion (ADR-024). Local MLflow store only (conftest.offline_mlflow)."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("mlflow")

from mlflow import MlflowClient  # noqa: E402

from src.config import load_params  # noqa: E402
from src.tracking import promote as pr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
NAME = "steel-defect-classifier-test"
ALIAS = "champion"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ci(v: float) -> dict:
    return {"value": v, "ci_low": v - 0.01, "ci_high": min(1.0, v + 0.01)}


def make_run(tmp_path: Path, tag: str = "a") -> dict:
    """A finished run with the three evidence files, registered as a new version."""
    client = MlflowClient()
    exp = client.get_experiment_by_name("promote-test")
    exp_id = exp.experiment_id if exp else client.create_experiment("promote-test")
    run = client.create_run(
        exp_id, tags={"checkpoint_sha256": f"ck{tag}", "train_run_uid": f"u{tag}"}
    )
    run_id = run.info.run_id
    sha = {}
    for name, artifact in pr.RUN_ARTIFACTS.items():
        folder, fname = artifact.split("/")
        local = tmp_path / tag / fname
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps({"name": name, "tag": tag}), encoding="utf-8")
        client.log_artifact(run_id, str(local), folder)
        sha[name] = _sha(local)
    client.set_terminated(run_id)
    if not client.search_registered_models(f"name = '{NAME}'"):
        client.create_registered_model(NAME)
    version = client.create_model_version(NAME, f"runs:/{run_id}/model", run_id)
    return {
        "model": "efficientnet_b0",
        "run_id": run_id,
        "run_url": f"https://example.invalid/runs/{run_id}",
        "train_run_uid": f"u{tag}",
        "checkpoint_sha256": f"ck{tag}",
        "split_file": "neu_split_v2.csv",
        "split_sha256": "s" * 64,
        "temperature": 1.6424999999,
        "threshold": 0.81612345678,
        "files": {
            "calibration": "results/train/calibration.json",
            "threshold": "results/train/threshold.json",
            "test_metrics": "results/eval/efficientnet_b0/test_metrics.json",
        },
        "sha256": sha,
        "results_commit": "c" * 40,
        "test": {
            "n_images": 360,
            "accuracy": _ci(0.99),
            "macro_f1": _ci(0.99),
            "review": {"n_routed": 8, "risk_ucb": {"value": 0.018}},
        },
        "val_review": {"n_routed": 9, "n_images": 180, "risk_ucb": {"value": 0.027}},
        "version": str(version.version),
    }


# ---------------------------------------------------------------- registry


def test_plan_is_a_dry_run(tmp_path):
    ev = make_run(tmp_path)
    p = pr.plan(MlflowClient(), NAME, ALIAS, ev)
    assert p["action"] == "set" and p["version"] == ev["version"]
    client = MlflowClient()
    assert client.get_registered_model(NAME).aliases == {}
    assert client.get_model_version(NAME, ev["version"]).tags == {}


def test_apply_writes_evidence_then_alias_and_is_idempotent(tmp_path):
    ev = make_run(tmp_path)
    client = MlflowClient()
    tags = pr.version_tags(ev)
    desc = pr.version_description(ev, "https://github.com/o/r")
    pr.apply(client, NAME, ALIAS, ev["version"], tags, desc)
    mv = client.get_model_version_by_alias(NAME, ALIAS)
    assert str(mv.version) == ev["version"]  # int in the local store
    assert mv.tags == tags and mv.description == desc
    assert client.get_registered_model(NAME).description == pr.MODEL_DESCRIPTION
    assert pr.plan(client, NAME, ALIAS, ev)["action"] == "none"


def test_plan_refuses_when_another_version_holds_the_alias(tmp_path):
    first = make_run(tmp_path, "a")
    second = make_run(tmp_path, "b")
    MlflowClient().set_registered_model_alias(NAME, ALIAS, first["version"])
    with pytest.raises(pr.PromotionError, match="promotion gate"):
        pr.plan(MlflowClient(), NAME, ALIAS, second)


def test_find_version_refuses_a_run_without_a_version(tmp_path):
    make_run(tmp_path)
    with pytest.raises(pr.PromotionError, match="found 0"):
        pr.find_version(MlflowClient(), NAME, "0" * 32)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"checkpoint_sha256": "other"}, "another checkpoint"),
        ({"train_run_uid": "other"}, "another training run"),
    ],
)
def test_check_run_refuses_another_run(tmp_path, change, message):
    ev = {**make_run(tmp_path), **change}
    with pytest.raises(pr.PromotionError, match=message):
        pr.check_run(MlflowClient(), ev)


def test_check_run_refuses_changed_evidence(tmp_path):
    ev = make_run(tmp_path)
    ev["sha256"] = {**ev["sha256"], "threshold": "0" * 64}
    with pytest.raises(pr.PromotionError, match="differs from the local file"):
        pr.check_run(MlflowClient(), ev)


def test_run_dry_run_then_apply(tmp_path, monkeypatch):
    ev = make_run(tmp_path)
    params = copy.deepcopy(PARAMS)
    params["tracking"]["registered_model"] = NAME
    log = tmp_path / "promotion_log.json"
    params["promotion"]["log_file"] = str(log)
    monkeypatch.setattr(pr, "collect_evidence", lambda p: ev)

    out = pr.run(params)
    assert out["action"] == "set" and not log.exists()
    assert MlflowClient().get_registered_model(NAME).aliases == {}

    out = pr.run(params, do_apply=True)
    aliases = MlflowClient().get_registered_model(NAME).aliases
    assert {k: str(v) for k, v in aliases.items()} == {ALIAS: ev["version"]}
    entries = json.loads(log.read_text(encoding="utf-8"))
    assert len(entries) == 1 and entries[0]["version"] == ev["version"]
    assert entries[0]["previous_version"] is None

    out = pr.run(params, do_apply=True)  # second apply: nothing changes, nothing logged
    assert out["action"] == "none"
    assert len(json.loads(log.read_text(encoding="utf-8"))) == 1


# ---------------------------------------------------------------- local checks + text


def _report() -> dict:
    return {
        "role": "main",
        "split_sha256": "s",
        "calibration": {"temperature": 1.5},
        "review": {"threshold": 0.8},
        "provenance": {"checkpoint_sha256": "ck", "git_dirty": False},
    }


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("role",), "lighter alternative", "main model"),
        (("provenance", "checkpoint_sha256"), "x", "another checkpoint"),
        (("split_sha256",), "x", "another split"),
        (("calibration", "temperature"), 1.4, "another temperature"),
        (("review", "threshold"), 0.7, "another review threshold"),
        (("provenance", "git_dirty"), True, "dirty"),
    ],
)
def test_check_test_report_refuses_mismatches(path, value, message):
    report = _report()
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    kwargs = {"checkpoint_sha256": "ck", "split_sha256": "s", "temperature": 1.5, "threshold": 0.8}
    pr.check_test_report(_report(), **kwargs)  # the unchanged report passes
    with pytest.raises(pr.PromotionError, match=message):
        pr.check_test_report(report, **kwargs)


def test_tags_keep_full_precision_and_description_links_pin_the_commit(tmp_path):
    ev = make_run(tmp_path)
    tags = pr.version_tags(ev)
    assert float(tags["temperature"]) == ev["temperature"]
    assert float(tags["review_threshold"]) == ev["threshold"]
    assert all(isinstance(v, str) for v in tags.values())
    assert tags["threshold_artifact"] == "threshold/threshold.json"
    desc = pr.version_description(ev, "https://github.com/o/r/")
    assert f"https://github.com/o/r/blob/{'c' * 40}/results/train/threshold.json" in desc
    assert "simulated" in desc and "ADR-024" in desc


def test_append_log_appends(tmp_path):
    path = tmp_path / "log.json"
    pr.append_log(path, {"a": 1})
    pr.append_log(path, {"b": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == [{"a": 1}, {"b": 2}]


# ---------------------------------------------------------------- committed state


def test_committed_promotion_log_matches_the_evidence():
    log_path = REPO_ROOT / PARAMS["promotion"]["log_file"]
    entries = json.loads(log_path.read_text(encoding="utf-8"))
    first = entries[0]
    pointer = json.loads(
        (
            REPO_ROOT / PARAMS["compare_val"]["models"][0]["results_dir"] / "mlflow_run.json"
        ).read_text(encoding="utf-8")
    )
    assert first["decision"] == "initial_champion" and first["decision_record"] == "ADR-024"
    assert first["alias"] == PARAMS["promotion"]["champion_alias"]
    assert first["run_id"] == pointer["run_id"]
    assert first["version"] == pointer["registered_version"]
    assert first["model"] == PARAMS["train"]["model"]
    for item in first["evidence"].values():
        assert _sha(REPO_ROOT / item["file"]) == item["sha256"]


@pytest.mark.needs_model
def test_committed_evidence_passes_the_local_checks():
    ev = pr.collect_evidence(PARAMS)
    assert ev["model"] == PARAMS["train"]["model"]
    assert set(ev["sha256"]) == set(pr.RUN_ARTIFACTS)
