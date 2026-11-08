"""TASK-011: evaluation on test_frozen (ADR-022). Fast tests never read test_frozen images."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from src.config import load_params  # noqa: E402
from src.data.groups import Overlap, write_overlaps  # noqa: E402
from src.tracking.artifact_policy import is_allowed_artifact  # noqa: E402
from src.training import evaluate as ev  # noqa: E402
from src.training.compare_val import ModelSpec  # noqa: E402
from src.training.metrics import (  # noqa: E402
    bootstrap_metrics,
    grouped_bootstrap_indices,
    per_class_precision_recall_f1,
    percentile_interval,
    softmax,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
CLASSES = ["a", "b", "c"]


# ---------------------------------------------------------------- metrics


def test_precision_recall_f1_known_values():
    y = np.array([0, 0, 0, 1, 1, 2])
    p = np.array([0, 0, 1, 1, 2, 2])
    precision, recall, f1 = per_class_precision_recall_f1(y, p, 3)
    np.testing.assert_allclose(precision, [1.0, 1 / 2, 1 / 2])
    np.testing.assert_allclose(recall, [2 / 3, 1 / 2, 1.0])
    np.testing.assert_allclose(f1, [0.8, 0.5, 2 / 3])


def test_undefined_precision_is_zero():
    precision, recall, _ = per_class_precision_recall_f1(np.array([0, 1]), np.array([0, 0]), 2)
    assert precision[1] == 0.0 and recall[1] == 0.0


def test_grouped_bootstrap_keeps_groups_together_and_unit_counts():
    labels = np.array([0, 0, 0, 0, 1, 1, 1])
    units = np.array([10, 10, 11, 12, 20, 20, 20])  # class 0: 3 units, class 1: 1 unit
    for idx in grouped_bootstrap_indices(labels, units, n_resamples=200, seed=3):
        counts = np.bincount(idx, minlength=len(labels))
        assert counts[0] == counts[1]  # unit 10 drawn as a whole
        assert counts[4] == counts[5] == counts[6] == 1  # class 1 has one unit: always drawn once
        drawn_units_class0 = counts[0] + counts[2] + counts[3]  # unit 10 counted once per draw
        assert drawn_units_class0 == 3


def test_grouped_bootstrap_is_repeatable():
    labels, units = np.array([0, 0, 1, 1]), np.arange(4)
    a = grouped_bootstrap_indices(labels, units, 20, seed=5)
    b = grouped_bootstrap_indices(labels, units, 20, seed=5)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_grouped_bootstrap_rejects_mixed_class_unit():
    with pytest.raises(ValueError, match="several classes"):
        grouped_bootstrap_indices(np.array([0, 1]), np.array([7, 7]), 1, seed=0)


def test_bootstrap_on_the_full_sample_equals_point_metrics():
    y, p = np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 1, 2, 0])
    boot = bootstrap_metrics(y, p, 3, [np.arange(6)])
    _, _, f1 = per_class_precision_recall_f1(y, p, 3)
    assert boot["accuracy"][0] == pytest.approx(4 / 6)
    assert boot["macro_f1"][0] == pytest.approx(f1.mean())


def test_percentile_interval():
    lo, hi = percentile_interval(np.arange(101, dtype=float), 0.9)
    assert (lo, hi) == (pytest.approx(5.0), pytest.approx(95.0))


def test_metrics_report_intervals_contain_values_and_perfect_class_collapses():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 20)
    p = y.copy()
    p[rng.choice(np.arange(20, 60), size=6, replace=False)] = 0  # errors only in classes b, c
    samples = grouped_bootstrap_indices(y, np.arange(60), 500, seed=1)
    r = ev.metrics_report(y, p, CLASSES, samples, 0.95)
    assert r["n_errors"] == 6
    for m in [r["accuracy"], r["macro_f1"]] + [
        c[k] for c in r["per_class"].values() for k in ("precision", "recall", "f1")
    ]:
        assert m["ci_low"] <= m["value"] <= m["ci_high"]
    rec_a = r["per_class"]["a"]["recall"]
    assert rec_a == {"value": 1.0, "ci_low": 1.0, "ci_high": 1.0}
    assert r["accuracy"]["ci_low"] < r["accuracy"]["value"] < r["accuracy"]["ci_high"]
    assert np.array(r["confusion_matrix"]["counts"]).sum() == 60


# ---------------------------------------------------------------- groups


def test_unit_ids_share_ids_within_groups_only():
    paths = ["p1", "p2", "p3", "p4"]
    ids = ev.unit_ids(paths, [["p1", "p3"], ["x9", "y9"]])
    assert ids[0] == ids[2]
    assert len({ids[0], ids[1], ids[3]}) == 3


def test_near_copy_groups_use_confirmed_edges_inside_the_set(tmp_path):
    pairs = tmp_path / "pairs.csv"
    write_overlaps(
        [
            Overlap(0.9, 0.95, "t1", "a", "t2", "a"),  # both in set, confirmed
            Overlap(0.2, 0.95, "t3", "a", "t4", "a"),  # below the threshold
            Overlap(0.9, 0.95, "t1", "a", "other", "a"),  # crosses out of the set
        ],
        pairs,
    )
    cfg = ev.EvalConfig(32, 10, 0, 0.95, tmp_path, pairs, 0.3, 15)
    assert ev.near_copy_groups_within(["t1", "t2", "t3", "t4"], cfg) == [["t1", "t2"]]


# ---------------------------------------------------------------- checks


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_setup(root: Path, *, significant=False, order=("m1", "m2")) -> dict:
    """A tiny repo layout with a settled ADR-022 comparison under ``root``."""
    (root / "split.csv").write_text("path,class,split\n", encoding="utf-8")
    params = copy.deepcopy(PARAMS)
    params["data"]["split_file"] = "split.csv"
    params["train"]["model"] = "m1"
    models, report_models = [], []
    for name in ("m1", "m2"):
        ckpt, res = root / f"{name}.pt", root / f"res_{name}"
        ckpt.write_bytes(name.encode())
        res.mkdir()
        (res / "val_metrics.json").write_text(json.dumps({"checkpoint_sha256": _sha(ckpt)}))
        cal = {
            "temperature": 1.5,
            "at_bound": False,
            "checkpoint_sha256": _sha(ckpt),
            "fitted_on": {
                "split": "val",
                "split_file": "split.csv",
                "split_sha256": _sha(root / "split.csv"),
            },
        }
        (res / "calibration.json").write_text(json.dumps(cal), encoding="utf-8")
        if name == "m1":  # the review threshold exists for the main model only
            rule = {
                k: PARAMS["threshold"][k] for k in ("review_budget", "max_risk_ucb", "ucb_level")
            }
            thr = {
                "threshold": 0.9,
                "passed": True,
                "rule": rule,
                "checkpoint_sha256": _sha(ckpt),
                "calibration_sha256": _sha(res / "calibration.json"),
                "fitted_on": cal["fitted_on"],
            }
            (res / "threshold.json").write_text(json.dumps(thr), encoding="utf-8")
        models.append({"name": name, "checkpoint": f"{name}.pt", "results_dir": f"res_{name}"})
    for name in order:
        report_models.append({"name": name, "checkpoint_sha256": _sha(root / f"{name}.pt")})
    params["compare_val"] = {**params["compare_val"], "models": models, "out": "cmp.json"}
    report = {
        "split": "val",
        "split_file": "split.csv",
        "split_sha256": _sha(root / "split.csv"),
        "significant": significant,
        "models": report_models,
    }
    (root / "cmp.json").write_text(json.dumps(report), encoding="utf-8")
    return params


def test_check_model_choice_accepts_a_settled_choice(tmp_path):
    specs = ev.check_model_choice(fake_setup(tmp_path), tmp_path)
    assert [s.name for s in specs] == ["m1", "m2"]


def test_check_model_choice_refuses_a_significant_result(tmp_path):
    with pytest.raises(ev.EvalError, match="owner must decide"):
        ev.check_model_choice(fake_setup(tmp_path, significant=True), tmp_path)


def test_check_model_choice_refuses_another_order(tmp_path):
    with pytest.raises(ev.EvalError, match="order"):
        ev.check_model_choice(fake_setup(tmp_path, order=("m2", "m1")), tmp_path)


def test_check_model_choice_refuses_main_model_not_first(tmp_path):
    params = fake_setup(tmp_path)
    params["train"]["model"] = "m2"
    with pytest.raises(ev.EvalError, match="main model first"):
        ev.check_model_choice(params, tmp_path)


def test_check_model_choice_refuses_a_changed_checkpoint(tmp_path):
    params = fake_setup(tmp_path)
    (tmp_path / "m2.pt").write_bytes(b"retrained")
    with pytest.raises(ev.EvalError, match="not the checkpoint"):
        ev.check_model_choice(params, tmp_path)


def test_check_model_choice_refuses_a_changed_split(tmp_path):
    params = fake_setup(tmp_path)
    (tmp_path / "split.csv").write_text("path,class,split\nx,a,val\n", encoding="utf-8")
    with pytest.raises(ev.EvalError, match="split file changed"):
        ev.check_model_choice(params, tmp_path)


def test_check_model_choice_refuses_without_comparison(tmp_path):
    params = fake_setup(tmp_path)
    (tmp_path / "cmp.json").unlink()
    with pytest.raises(ev.EvalError, match="make compare-val"):
        ev.check_model_choice(params, tmp_path)


def test_check_split_refuses_without_clean_report(tmp_path):
    with pytest.raises(ev.EvalError, match="near-copy report"):
        ev.check_split(PARAMS, tmp_path)


def test_run_checks_before_reading_test_frozen(monkeypatch):
    def no_test_read(*args, **kwargs):
        raise AssertionError("test_frozen was read before the checks passed")

    def refuse(*args, **kwargs):
        raise ev.EvalError("not settled")

    monkeypatch.setattr(ev, "SplitDataset", no_test_read)
    monkeypatch.setattr(ev, "check_model_choice", refuse)
    with pytest.raises(ev.EvalError, match="not settled"):
        ev.run(PARAMS)


def _spec(root: Path, name: str = "m1") -> ModelSpec:
    return ModelSpec(name, Path(f"{name}.pt"), Path(f"res_{name}"))


def test_load_temperature_accepts_a_matching_calibration(tmp_path):
    params = fake_setup(tmp_path)
    t, path = ev.load_temperature(params, _spec(tmp_path), tmp_path)
    assert t == 1.5 and path.name == ev.CALIBRATION_FILE


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"checkpoint_sha256": "0" * 64}, "another checkpoint"),
        ({"at_bound": True}, "range bound"),
        ({"fitted_on": {"split": "test_frozen", "split_file": "split.csv"}}, "not fitted on val"),
        (
            {"fitted_on": {"split": "val", "split_file": "split.csv", "split_sha256": "x"}},
            "changed",
        ),
    ],
)
def test_load_temperature_refuses_a_mismatched_calibration(tmp_path, change, message):
    params = fake_setup(tmp_path)
    path = tmp_path / "res_m1" / ev.CALIBRATION_FILE
    cal = {**json.loads(path.read_text(encoding="utf-8")), **change}
    path.write_text(json.dumps(cal), encoding="utf-8")
    with pytest.raises(ev.EvalError, match=message):
        ev.load_temperature(params, _spec(tmp_path), tmp_path)


def test_load_temperature_refuses_without_calibration(tmp_path):
    params = fake_setup(tmp_path)
    (tmp_path / "res_m1" / ev.CALIBRATION_FILE).unlink()
    with pytest.raises(ev.EvalError, match="make calibrate"):
        ev.load_temperature(params, _spec(tmp_path), tmp_path)


def test_load_threshold_accepts_a_matching_threshold(tmp_path):
    params = fake_setup(tmp_path)
    cal = tmp_path / "res_m1" / ev.CALIBRATION_FILE
    t, path = ev.load_threshold(params, _spec(tmp_path), cal, tmp_path)
    assert t == 0.9 and path.name == "threshold.json"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"checkpoint_sha256": "0" * 64}, "another checkpoint"),
        ({"calibration_sha256": "0" * 64}, "another calibration"),
        ({"passed": False}, "risk check"),
        ({"rule": {"review_budget": 0.2, "max_risk_ucb": 0.05, "ucb_level": 0.95}}, "changed"),
        ({"fitted_on": {"split": "test_frozen", "split_file": "split.csv"}}, "not set on val"),
    ],
)
def test_load_threshold_refuses_a_mismatched_threshold(tmp_path, change, message):
    params = fake_setup(tmp_path)
    path = tmp_path / "res_m1" / "threshold.json"
    thr = {**json.loads(path.read_text(encoding="utf-8")), **change}
    path.write_text(json.dumps(thr), encoding="utf-8")
    with pytest.raises(ev.EvalError, match=message):
        ev.load_threshold(
            params, _spec(tmp_path), tmp_path / "res_m1" / "calibration.json", tmp_path
        )


def test_load_threshold_refuses_a_recalibrated_model(tmp_path):
    params = fake_setup(tmp_path)
    cal = tmp_path / "res_m1" / ev.CALIBRATION_FILE
    cal.write_text(cal.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ev.EvalError, match="another calibration"):
        ev.load_threshold(params, _spec(tmp_path), cal, tmp_path)


def test_load_threshold_refuses_without_threshold(tmp_path):
    params = fake_setup(tmp_path)
    (tmp_path / "res_m1" / "threshold.json").unlink()
    with pytest.raises(ev.EvalError, match="make threshold"):
        ev.load_threshold(params, _spec(tmp_path), tmp_path / "res_m1" / "x", tmp_path)


def test_calibration_report_keeps_classes_and_counts_errors():
    y = np.array([0, 1, 2, 0])
    logits = np.eye(3)[[0, 1, 1, 0]] * 4.0  # one error
    before, after = softmax(logits), softmax(logits, 2.0)
    samples = grouped_bootstrap_indices(y, np.arange(4), 20, seed=0)
    r = ev.calibration_report(y, before, after, samples, 10, 0.95)
    assert r["n_errors"] == 1 and "Only 1 of 4" in r["caution"]
    assert set(r["verdict"]) == {"ece", "nll", "brier"}
    assert sum(r["reliability_after"]["count"]) == 4


def test_committed_state_passes_the_checks_and_main_model_is_first():
    ev.check_split(PARAMS)
    cfg = PARAMS["compare_val"]
    assert cfg["models"][0]["name"] == PARAMS["train"]["model"]
    report = json.loads((REPO_ROOT / cfg["out"]).read_text(encoding="utf-8"))
    assert report["significant"] is False
    for i, spec in enumerate(ev.specs_from_params(cfg)):
        if (REPO_ROOT / spec.checkpoint).exists():
            _, cal_path = ev.load_temperature(PARAMS, spec)
            if i == 0:
                ev.load_threshold(PARAMS, spec, cal_path)


# ---------------------------------------------------------------- outputs + MLflow


def fake_report(name="m1") -> tuple[dict, np.ndarray, np.ndarray]:
    y = np.repeat([0, 1, 2], 4)
    probs = np.eye(3)[y] * 0.9 + 0.05
    probs[0] = [0.1, 0.8, 0.1]  # one error
    samples = grouped_bootstrap_indices(y, np.arange(len(y)), 50, seed=0)
    calibration = {
        "temperature": 1.0,
        **ev.calibration_report(y, probs, probs, samples, 10, 0.95),
    }
    report = {
        "model": name,
        "role": "main",
        "split_file": "split.csv",
        "n_images": len(y),
        **ev.metrics_report(y, probs.argmax(axis=1), CLASSES, samples, 0.95),
        "bootstrap": {"method": "cluster", "n_resamples": 50},
        "calibration": calibration,
        "review": {
            "source": "res/threshold.json",
            **ev.review_report(
                y,
                probs.argmax(axis=1),
                probs.max(axis=1),
                0.85,
                np.arange(len(y)),
                samples,
                0.95,
                0.95,
            ),
        },
        "provenance": {"git_commit": "abc"},
    }
    return report, y, probs


def test_write_outputs_and_plot_names_are_allowlisted(tmp_path):
    report, y, probs = fake_report()
    paths = [f"img_{i}.jpg" for i in range(len(y))]
    files = ev.write_outputs(tmp_path, report, paths, y, probs, probs * 0 + 0.5, CLASSES)
    assert all(p.is_file() for p in files.values())
    for key in ("confusion_matrix", "confusion_matrix_normalized"):
        assert is_allowed_artifact(f"{ev.PLOT_DIR}/{files[key].name}")
    for key in ("reliability_before", "reliability_after"):
        assert is_allowed_artifact(f"{ev.PLOT_DIR}/{files[key].name}")
    lines = files["predictions"].read_text(encoding="utf-8").splitlines()
    assert lines[0] == "path,true,pred,confidence,calibrated_confidence,correct"
    assert lines[1] == "img_0.jpg,a,b,0.8000,0.5000,0"
    assert b"\r\n" not in files["metrics"].read_bytes()
    routed = probs.max(axis=1) < 0.85
    files = ev.write_outputs(tmp_path, report, paths, y, probs, probs, CLASSES, routed)
    lines = files["predictions"].read_text(encoding="utf-8").splitlines()
    assert lines[0].endswith(",correct,routed_to_review")
    assert lines[1] == "img_0.jpg,a,b,0.8000,0.8000,0,1"


def test_mlflow_payload_names():
    report, _, _ = fake_report()
    metrics, tags = ev.mlflow_payload(report)
    assert {"test_accuracy", "test_macro_f1_ci_low", "test_recall.a", "test_f1.c"} <= set(metrics)
    assert all(isinstance(v, float) for v in metrics.values())
    assert tags["test_eval_role"] == "main"
    assert {"test_temperature_applied", "test_ece_before", "test_nll_after"} <= set(metrics)
    assert {"test_review_threshold", "test_review_risk_ucb", "test_review_coverage"} <= set(metrics)
    assert tags["test_eval_threshold_record"] == "ADR-023"
    report.pop("review")
    metrics, tags = ev.mlflow_payload(report)
    assert "test_review_threshold" not in metrics and "test_eval_threshold_record" not in tags


def test_log_to_mlflow_adds_to_the_finished_run_locally(tmp_path):
    pytest.importorskip("mlflow")
    from mlflow import MlflowClient

    from src.tracking import mlflow_tracking as mt

    cfg = mt.TrackingConfig.from_params(PARAMS, smoke=False)
    with mt.RunTracker(cfg) as tracker:
        tracker.log_metrics({"best_val_macro_f1": 0.9})
    run_id = tracker.run_id
    before = MlflowClient().get_run(run_id).info

    res = tmp_path / "res"
    res.mkdir()
    (res / "mlflow_run.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    report, y, probs = fake_report()
    files = ev.write_outputs(
        tmp_path / "out", report, [str(i) for i in y], y, probs, probs, CLASSES
    )
    link = ev.log_to_mlflow(PARAMS, ModelSpec("m1", tmp_path / "x.pt", res), files)

    run = MlflowClient().get_run(run_id)
    assert run_id in link
    assert run.info.status == "FINISHED" and run.info.end_time == before.end_time
    assert run.data.metrics["test_accuracy"] == pytest.approx(report["accuracy"]["value"])
    assert run.data.tags["test_eval_decision_record"] == "ADR-022"
    plots = {a.path for a in MlflowClient().list_artifacts(run_id, ev.PLOT_DIR)}
    assert plots == {
        "plots/confusion_matrix.png",
        "plots/confusion_matrix_normalized.png",
        "plots/reliability_diagram_test_before.png",
        "plots/reliability_diagram_test_after.png",
    }
    assert [a.path for a in MlflowClient().list_artifacts(run_id, "eval")] == [
        "eval/test_metrics.json"
    ]


# ---------------------------------------------------------------- committed results


def _committed(name: str) -> dict:
    out = REPO_ROOT / PARAMS["evaluate"]["out_dir"] / name / "test_metrics.json"
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.parametrize("index", [0, 1])
def test_committed_results_are_consistent(index):
    spec = PARAMS["compare_val"]["models"][index]
    r = _committed(spec["name"])
    boot = PARAMS["evaluate"]["bootstrap"]
    assert r["role"] == ev.ROLES[index]
    assert r["split"] == "test_frozen"
    assert r["split_file"] == Path(PARAMS["data"]["split_file"]).name
    assert (r["bootstrap"]["n_resamples"], r["bootstrap"]["seed"]) == (
        boot["n_resamples"],
        boot["seed"],
    )
    assert r["bootstrap"]["level"] == boot["level"]
    assert r["provenance"]["git_dirty"] is False
    counts = np.array(r["confusion_matrix"]["counts"])
    assert counts.sum() == r["n_images"]
    assert r["n_errors"] == r["n_images"] - np.trace(counts)
    for m in [r["accuracy"], r["macro_f1"]]:
        assert m["ci_low"] <= m["value"] <= m["ci_high"]
    recorded = json.loads(
        (REPO_ROOT / spec["results_dir"] / "val_metrics.json").read_text(encoding="utf-8")
    )
    assert r["provenance"]["checkpoint_sha256"] == recorded["checkpoint_sha256"]
    cal = json.loads(
        (REPO_ROOT / spec["results_dir"] / "calibration.json").read_text(encoding="utf-8")
    )
    c = r["calibration"]
    assert c["temperature"] == cal["temperature"]
    assert c["source"] == f"{spec['results_dir']}/calibration.json"
    assert c["n_errors"] == r["n_errors"]
    assert c["reliability_before"]["n_bins"] == PARAMS["calibrate"]["n_bins"]
    rows = REPO_ROOT / PARAMS["evaluate"]["out_dir"] / spec["name"] / "test_predictions.csv"
    lines = rows.read_text(encoding="utf-8").splitlines()
    header = "path,true,pred,confidence,calibrated_confidence,correct"
    if index == 0:  # main model: the val threshold is applied (ADR-023)
        thr = json.loads(
            (REPO_ROOT / spec["results_dir"] / "threshold.json").read_text(encoding="utf-8")
        )
        rv = r["review"]
        assert rv["threshold"] == thr["threshold"]
        assert rv["source"] == f"{spec['results_dir']}/threshold.json"
        assert rv["n_errors"] == r["n_errors"] == rv["errors_routed"] + rv["errors_accepted"]
        assert rv["n_routed"] == sum(line.endswith(",1") for line in lines[1:])
        assert lines[0] == header + ",routed_to_review"
    else:
        assert "review" not in r
        assert lines[0] == header


@pytest.mark.needs_data
@pytest.mark.needs_model
def test_run_reproduces_committed_results(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    for m in PARAMS["compare_val"]["models"]:
        if not Path(m["checkpoint"]).exists():
            pytest.skip(f"missing checkpoint {m['checkpoint']}")
    params = copy.deepcopy(PARAMS)
    params["evaluate"]["out_dir"] = str(tmp_path)
    for s in ev.run(params):
        new, old = s["report"], _committed(s["model"])
        keys = ("accuracy", "macro_f1", "per_class", "confusion_matrix", "bootstrap", "calibration")
        for key in keys + (("review",) if s["role"] == "main" else ()):
            assert new[key] == old[key], key
        name = "test_predictions.csv"
        committed_csv = REPO_ROOT / PARAMS["evaluate"]["out_dir"] / s["model"] / name
        assert (tmp_path / s["model"] / name).read_bytes() == committed_csv.read_bytes()
