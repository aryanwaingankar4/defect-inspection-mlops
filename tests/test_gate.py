"""TASK-027 (ADR-033): the non-inferiority promotion gate. Synthetic predictions (no data, no
model) for the four checks, and a local MLflow registry for the alias moves and the log."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlflow")
pytest.importorskip("torch")

from mlflow import MlflowClient  # noqa: E402

from src.config import load_params  # noqa: E402
from src.tracking import gate as gt  # noqa: E402
from src.training import export_onnx as ex  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
CFG = gt.GateConfig.from_params(PARAMS)
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def labels(per_class: int = 30) -> np.ndarray:
    return np.repeat(np.arange(len(CLASSES)), per_class)


def wrong_at(y: np.ndarray, idx) -> np.ndarray:
    """``y`` with the images at ``idx`` predicted as the next class."""
    pred = y.copy()
    idx = np.asarray(list(idx), dtype=int)
    pred[idx] = (y[idx] + 1) % len(CLASSES)
    return pred


def spread(y: np.ndarray, n: int, offset: int = 0) -> list[int]:
    """``n`` indices spread over all classes (round robin), starting at ``offset`` per class."""
    per_class = len(y) // len(CLASSES)
    return [(i % len(CLASSES)) * per_class + offset + i // len(CLASSES) for i in range(n)]


PARITY_OK = {
    "n_images": 180,
    "max_tolerance_ratio": 0.6,
    "same_class": 180,
    "same_routing": 180,
    "passed": True,
    "batch_check": [{"batch": b, "passed": True} for b in (1, 7, 32)],
}

Y = labels()
CHAMPION = wrong_at(Y, [0, 31, 62, 93])  # 4 errors on 180 val images, one in each of 4 classes


def decide(challenger, parity=PARITY_OK, y=Y, champion=CHAMPION, cfg=CFG):
    return gt.evaluate(y, champion, challenger, CLASSES, parity, cfg)


# ---------------------------------------------------------------- the five owner scenarios


def test_better_challenger_is_promoted():
    out = decide(Y.copy())  # fixes all 4 champion errors
    assert out["decision"] == "promote" and out["failed_checks"] == []
    assert out["table"]["only_challenger_right"] == 4 and out["table"]["only_champion_right"] == 0
    # 4 discordant images can never be significant: why the gate is non-inferiority (ADR-033)
    assert out["checks"]["mcnemar"]["value"] > CFG.mcnemar_alpha


def test_equal_challenger_is_promoted():
    out = decide(CHAMPION.copy())
    assert out["decision"] == "promote"
    assert out["checks"]["macro_f1"]["value"] == 0.0
    assert out["checks"]["mcnemar"]["value"] == 1.0


def test_one_collapsed_class_is_rejected():
    challenger = CHAMPION.copy()
    challenger[Y == 2] = 3  # every "patches" image predicted as pitted_surface
    out = decide(challenger)
    assert out["decision"] == "reject"
    rec = out["checks"]["class_recall"]
    assert not rec["passed"] and rec["failed_classes"] == ["patches"]
    assert rec["per_class_drop"]["patches"] == pytest.approx(29 / 30)
    assert "patches" in rec["reason"]


def test_significantly_worse_challenger_is_blocked_by_mcnemar_alone():
    # 6000 images: 40 champion-only vs 15 challenger-only correct is a small macro-F1 and recall
    # change but a significant paired difference, so only check c) rejects.
    y = labels(1000)
    champion = wrong_at(y, spread(y, 15))
    challenger = wrong_at(y, spread(y, 40, offset=500))
    out = decide(challenger, y=y, champion=champion, parity={**PARITY_OK, "n_images": 6000})
    assert out["decision"] == "reject" and out["failed_checks"] == ["mcnemar"]
    m = out["checks"]["mcnemar"]
    assert m["value"] < CFG.mcnemar_alpha
    assert (m["champion_only_correct"], m["challenger_only_correct"]) == (40, 15)
    assert "significantly worse" in m["reason"]


def test_significantly_better_challenger_is_not_blocked():
    y = labels(1000)
    champion = wrong_at(y, spread(y, 40, offset=500))
    challenger = wrong_at(y, spread(y, 15))
    out = decide(challenger, y=y, champion=champion, parity={**PARITY_OK, "n_images": 6000})
    assert out["checks"]["mcnemar"]["value"] < CFG.mcnemar_alpha
    assert out["checks"]["mcnemar"]["passed"] and out["decision"] == "promote"


@pytest.mark.parametrize(
    "parity",
    [
        {**PARITY_OK, "passed": False, "same_class": 179, "max_tolerance_ratio": 3.2},
        {**PARITY_OK, "batch_check": [{"batch": 1, "passed": True}, {"batch": 7, "passed": False}]},
        None,
    ],
)
def test_parity_failure_is_rejected(parity):
    out = decide(CHAMPION.copy(), parity=parity)
    assert out["decision"] == "reject" and out["failed_checks"] == ["onnx_parity"]
    assert not out["checks"]["onnx_parity"]["passed"]


# ---------------------------------------------------------------- thresholds at the edges


def test_recall_allows_one_more_miss_per_class_but_not_two():
    one_more = wrong_at(Y, [0, 31, 62, 93, 1])  # crazing: 28/30 instead of 29/30
    assert decide(one_more)["checks"]["class_recall"]["passed"]
    two_more = wrong_at(Y, [0, 31, 62, 93, 1, 2])  # crazing: 27/30
    out = decide(two_more)
    assert not out["checks"]["class_recall"]["passed"] and out["decision"] == "reject"


def test_macro_f1_margin_comes_from_params():
    strict = gt.GateConfig(**{**CFG.__dict__, "max_macro_f1_drop": 0.0})
    one_more = wrong_at(Y, [0, 31, 62, 93, 1])
    assert decide(one_more)["checks"]["macro_f1"]["passed"]
    assert not decide(one_more, cfg=strict)["checks"]["macro_f1"]["passed"]


def test_gate_params():
    g = PARAMS["promotion"]["gate"]
    assert g["split"] == "val"
    assert (g["max_macro_f1_drop"], g["max_class_recall_drop"], g["mcnemar_alpha"]) == (
        0.01,
        0.05,
        0.05,
    )


# ---------------------------------------------------------------- data guards


def test_gate_refuses_test_frozen():
    params = copy.deepcopy(PARAMS)
    params["promotion"]["gate"]["split"] = "test_frozen"
    with pytest.raises(gt.GateError, match="test_frozen"):
        gt.GateConfig.from_params(params)


def test_gate_refuses_feedback_images(tmp_path):
    params = copy.deepcopy(PARAMS)
    params["review"]["export"]["out_dir"] = str(tmp_path)
    gt.check_no_feedback(["a/1.jpg"], params)  # no export yet: fine
    (tmp_path / "feedback.csv").write_text(
        "path,label,reviewer_kind,request_id\na/1.jpg,crazing,simulated,x\n", encoding="utf-8"
    )
    gt.check_no_feedback(["a/2.jpg"], params)
    with pytest.raises(gt.GateError, match="feedback rows"):
        gt.check_no_feedback(["a/1.jpg", "a/2.jpg"], params)


# ---------------------------------------------------------------- registry (local store)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def registry(offline_mlflow, tmp_path):
    """Version 1 holds @champion, version 2 holds @challenger; both with serving tags."""
    client = MlflowClient(tracking_uri=offline_mlflow)
    exp = client.create_experiment("gate-test")
    name = PARAMS["tracking"]["registered_model"]
    client.create_registered_model(name)
    out = {"client": client, "name": name, "ckpt": {}, "run": {}}
    for version, alias in (("1", "champion"), ("2", "challenger")):
        run = client.create_run(exp)
        client.set_terminated(run.info.run_id)
        mv = client.create_model_version(name, f"runs:/{run.info.run_id}/model", run.info.run_id)
        assert str(mv.version) == version
        ckpt = tmp_path / f"v{version}.pt"
        ckpt.write_bytes(f"weights {version}".encode())
        tags = {
            "model": "efficientnet_b0",
            "checkpoint_sha256": _sha(ckpt),
            "temperature": "1.5",
            "review_threshold": "0.8",
            "ood_energy_threshold": "-6.1",  # servable: promote is allowed (ADR-034)
            "serving_precision": "fp32",
        }
        for k, v in tags.items():
            client.set_model_version_tag(name, version, k, v)
        client.set_registered_model_alias(name, alias, version)
        out["ckpt"][version], out["run"][version] = ckpt, run.info.run_id
    params = copy.deepcopy(PARAMS)
    params["promotion"]["log_file"] = str(tmp_path / "promotion_log.json")
    params["promotion"]["gate"]["out_dir"] = str(tmp_path / "promotion")
    params["review"]["export"]["out_dir"] = str(tmp_path / "no-feedback")
    out["params"] = params
    return out


def aliases(reg) -> dict[str, str]:
    return {k: str(v) for k, v in reg["client"].get_registered_model(reg["name"]).aliases.items()}


def test_resolve_and_checkpoint_checks(registry):
    champ, chall = gt.resolve(registry["params"])
    assert (champ["version"], chall["version"]) == ("1", "2") and chall["alias"] == "challenger"
    assert gt.verified_checkpoint(chall, registry["ckpt"]["2"]) == registry["ckpt"]["2"]
    with pytest.raises(gt.GateError, match="SHA-256"):
        gt.verified_checkpoint(chall, registry["ckpt"]["1"])  # the champion's file


def test_resolve_refuses_missing_or_identical_aliases(registry):
    client, name = registry["client"], registry["name"]
    client.set_registered_model_alias(name, "challenger", "1")
    with pytest.raises(gt.GateError, match="both version 1"):
        gt.resolve(registry["params"])
    client.delete_registered_model_alias(name, "challenger")
    with pytest.raises(gt.GateError, match="no @challenger"):
        gt.resolve(registry["params"])


def run_gate(registry, monkeypatch, challenger_pred, parity=PARITY_OK, apply=True, record=False):
    """The full run with scoring and the ONNX export replaced by synthetic results."""
    preds = {registry["ckpt"]["1"]: CHAMPION, registry["ckpt"]["2"]: challenger_pred}
    paths = [f"x/{i}.jpg" for i in range(len(Y))]

    def fake_score(ckpt, params, cfg):
        return {"y": Y, "pred": preds[ckpt], "classes": CLASSES, "paths": paths}

    calls = []

    def fake_parity(params, alias, ckpt, report, *, log_mlflow):
        calls.append({"alias": alias, "ckpt": ckpt, "log_mlflow": log_mlflow})
        return parity

    monkeypatch.setattr(gt, "score", fake_score)
    monkeypatch.setattr(gt, "onnx_parity", fake_parity)
    out = gt.run(
        registry["params"],
        registry["ckpt"]["2"],
        registry["ckpt"]["1"],
        do_apply=apply,
        record_only=record,
    )
    assert calls == [{"alias": "challenger", "ckpt": registry["ckpt"]["2"], "log_mlflow": apply}]
    return out


def test_promote_moves_champion_records_previous_and_drops_challenger(registry, monkeypatch):
    out = run_gate(registry, monkeypatch, Y.copy())
    assert out["record"]["decision"] == "promote"
    assert aliases(registry) == {"champion": "2"}  # @challenger removed: it now serves
    tags = registry["client"].get_model_version(registry["name"], "2").tags
    assert tags["gate_decision"] == "promote" and tags["previous_champion_version"] == "1"
    gate_file = Path(registry["params"]["promotion"]["gate"]["out_dir"]) / "gate_2.json"
    record = json.loads(gate_file.read_text(encoding="utf-8"))
    assert tags["gate_file_sha256"] == _sha(gate_file)
    assert record["applied"] and record["split"] == "val" and record["n_images"] == 180
    assert set(record["checks"]) == set(gt.CHECKS)
    for check in record["checks"].values():
        assert {"value", "limit", "passed", "reason", "rule"} <= set(check)
    log = json.loads(Path(registry["params"]["promotion"]["log_file"]).read_text("utf-8"))
    assert len(log) == 1
    assert log[0]["decision"] == "promote" and log[0]["decision_record"] == "ADR-033"
    assert (log[0]["version"], log[0]["previous_version"]) == ("2", "1")
    assert log[0]["gate_file_sha256"] == _sha(gate_file)
    assert all(log[0]["checks"].values())


def test_reject_keeps_both_aliases_and_logs_the_reasons(registry, monkeypatch):
    collapsed = CHAMPION.copy()
    collapsed[Y == 2] = 3
    out = run_gate(registry, monkeypatch, collapsed)
    assert out["record"]["decision"] == "reject"
    assert aliases(registry) == {"champion": "1", "challenger": "2"}
    tags = registry["client"].get_model_version(registry["name"], "2").tags
    assert tags["gate_decision"] == "reject" and "previous_champion_version" not in tags
    log = json.loads(Path(registry["params"]["promotion"]["log_file"]).read_text("utf-8"))
    assert log[-1]["decision"] == "reject" and log[-1]["previous_version"] is None
    assert log[-1]["version"] == "1" and log[-1]["challenger_version"] == "2"
    assert log[-1]["checks"]["class_recall"] is False
    assert "patches" in log[-1]["reasons"]["class_recall"]


def test_dry_run_writes_nothing_and_changes_nothing(registry, monkeypatch):
    out = run_gate(registry, monkeypatch, Y.copy(), apply=False)
    assert out["record"]["decision"] == "promote" and not out["record"]["applied"]
    assert aliases(registry) == {"champion": "1", "challenger": "2"}
    assert not Path(registry["params"]["promotion"]["log_file"]).exists()
    assert not Path(registry["params"]["promotion"]["gate"]["out_dir"]).exists()


def test_record_mode_writes_only_the_gate_file(registry, monkeypatch):
    out = run_gate(registry, monkeypatch, Y.copy(), apply=False, record=True)
    gate_file = Path(registry["params"]["promotion"]["gate"]["out_dir"]) / "gate_2.json"
    assert out["gate_file"] == gate_file and "log_file" not in out
    record = json.loads(gate_file.read_text(encoding="utf-8"))
    assert record["decision"] == "promote" and record["applied"] is False
    assert aliases(registry) == {"champion": "1", "challenger": "2"}  # nothing moved
    assert "gate_decision" not in registry["client"].get_model_version(registry["name"], "2").tags
    assert not Path(registry["params"]["promotion"]["log_file"]).exists()


def test_promote_refuses_a_challenger_that_cannot_be_served(registry, monkeypatch):
    registry["client"].delete_model_version_tag(registry["name"], "2", "ood_energy_threshold")
    with pytest.raises(gt.GateError, match="not servable"):
        run_gate(registry, monkeypatch, Y.copy())
    assert aliases(registry) == {"champion": "1", "challenger": "2"}
    assert not Path(registry["params"]["promotion"]["log_file"]).exists()


def test_champion_checkpoint_defaults_to_the_kept_version(tmp_path):
    params = copy.deepcopy(PARAMS)
    params["retrain"]["versions_dir"] = str(tmp_path)
    fallback = Path(params["compare_val"]["models"][0]["checkpoint"])
    assert gt.default_champion_checkpoint(params, "2") == fallback
    kept = tmp_path / "v2" / "best.pt"
    kept.parent.mkdir()
    kept.write_bytes(b"w")
    assert gt.default_champion_checkpoint(params, "2") == kept


def test_apply_refuses_if_an_alias_moved(registry):
    champ, chall = gt.resolve(registry["params"])
    registry["client"].set_registered_model_alias(registry["name"], "challenger", "1")
    with pytest.raises(gt.GateError, match="moved"):
        gt.apply_decision(
            registry["client"], registry["name"], registry["params"], champ, chall,
            "promote", "g.json", "0" * 64,
        )  # fmt: skip


def test_export_resolves_any_alias(registry):
    v = ex.registered_version(registry["params"], "challenger")
    assert v["version"] == "2" and v["alias"] == "challenger"
    assert ex.checkpoint_at(v, registry["ckpt"]["2"]).name == "efficientnet_b0"
    with pytest.raises(ex.ExportError, match="needs its checkpoint"):
        ex.run(registry["params"], alias="challenger")


# ---------------------------------------------------------------- ADR-035 additions


def test_gate_files_are_never_overwritten(registry, monkeypatch):
    run_gate(registry, monkeypatch, Y.copy(), apply=False, record=True)
    out_dir = Path(registry["params"]["promotion"]["gate"]["out_dir"])
    assert (out_dir / "gate_2.json").exists()
    out = run_gate(registry, monkeypatch, Y.copy(), apply=False, record=True)
    assert out["gate_file"] == out_dir / "gate_2.2.json"
    assert gt.next_free(out_dir, "gate_2") == out_dir / "gate_2.3.json"


def test_apply_does_not_log_a_second_onnx_file(registry, monkeypatch):
    registry["client"].set_model_version_tag(registry["name"], "2", "onnx_sha256", "a" * 64)
    preds = {registry["ckpt"]["1"]: CHAMPION, registry["ckpt"]["2"]: Y.copy()}
    paths = [f"x/{i}.jpg" for i in range(len(Y))]
    monkeypatch.setattr(
        gt,
        "score",
        lambda c, p, cfg: {"y": Y, "pred": preds[c], "classes": CLASSES, "paths": paths},
    )
    calls = []

    def fake_parity(params, alias, ckpt, report, *, log_mlflow):
        calls.append(log_mlflow)
        return PARITY_OK

    monkeypatch.setattr(gt, "onnx_parity", fake_parity)
    out = gt.run(registry["params"], registry["ckpt"]["2"], registry["ckpt"]["1"], do_apply=True)
    assert calls == [False] and out["record"]["decision"] == "promote"
    assert aliases(registry) == {"champion": "2"}


def test_apply_refuses_a_fault_injection_challenger(registry, monkeypatch):
    """ADR-036: a model trained on broken labels is gated in record mode only."""
    registry["client"].set_model_version_tag(registry["name"], "2", "fault_injection", "true")
    with pytest.raises(gt.GateError, match="FAULT-INJECTION"):
        gt.run(registry["params"], registry["ckpt"]["2"], registry["ckpt"]["1"], do_apply=True)
    assert aliases(registry) == {"champion": "1", "challenger": "2"}
    assert not Path(registry["params"]["promotion"]["gate"]["out_dir"]).exists()
    out = run_gate(registry, monkeypatch, Y.copy(), apply=False, record=True)
    assert out["record"]["applied"] is False and out["gate_file"].is_file()
