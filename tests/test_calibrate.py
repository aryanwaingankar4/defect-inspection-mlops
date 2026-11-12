"""TASK-012: temperature scaling on val (ADR-006). Fast tests never read images."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from src.config import load_params  # noqa: E402
from src.training import calibrate as cb  # noqa: E402
from src.training.compare_val import ModelSpec  # noqa: E402
from src.training.metrics import (  # noqa: E402
    brier,
    compare_calibration,
    ece,
    fit_temperature,
    grouped_bootstrap_indices,
    nll,
    reliability_bins,
    softmax,
    stratified_group_folds,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
CFG = cb.CalibConfig.from_params(PARAMS)


# ---------------------------------------------------------------- scores


def test_softmax_temperature_keeps_the_predicted_class():
    logits = np.random.default_rng(0).normal(size=(50, 6)) * 4
    for t in (0.25, 0.7, 1.0, 1.6, 4.0):
        p = softmax(logits, t)
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        assert np.array_equal(p.argmax(axis=1), logits.argmax(axis=1))
    assert softmax(logits, 2.0).max(axis=1).mean() < softmax(logits, 1.0).max(axis=1).mean()


def test_scores_match_hand_computed_values():
    probs = np.array([[0.8, 0.2], [0.4, 0.6], [0.9, 0.1]])
    y = np.array([0, 0, 0])  # second image is wrong (predicts class 1 at 0.6)
    assert nll(probs, y) == pytest.approx(-(math.log(0.8) + math.log(0.4) + math.log(0.9)) / 3)
    assert brier(probs, y) == pytest.approx((0.08 + 0.72 + 0.02) / 3)
    assert ece(probs, y, 10) == pytest.approx((0.2 + 0.6 + 0.1) / 3)


def test_reliability_bins_edges():
    probs = np.array([[1.0, 0.0], [0.2, 0.8], [0.5, 0.5]])
    b = reliability_bins(probs, np.array([0, 1, 0]), 5)
    assert b["count"].tolist() == [
        0,
        0,
        1,
        1,
        1,
    ]  # 0.5 -> (0.4, 0.6]; 0.8 -> (0.6, 0.8]; 1.0 -> last
    assert np.isnan(b["accuracy"][0])


def test_fit_temperature_recovers_a_known_temperature():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(6000, 4)) * 2.0
    y = np.array([rng.choice(4, p=p) for p in softmax(z)])  # labels drawn from softmax(z)
    t, at_bound = fit_temperature(z * 2.0, y, 0.25, 4.0, 81)  # model twice as confident
    assert not at_bound
    assert t == pytest.approx(2.0, rel=0.06)


def test_fit_temperature_reports_the_bound_when_there_are_no_errors():
    logits = np.eye(3)[[0, 1, 2, 0]] * 5.0
    t, at_bound = fit_temperature(logits, np.array([0, 1, 2, 0]), 0.25, 4.0, 81)
    assert at_bound and t == pytest.approx(0.25)


def test_fit_temperature_rejects_a_range_without_one():
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((2, 2)), np.array([0, 1]), 1.5, 4.0, 81)


def test_group_folds_keep_units_together_and_spread_each_class():
    labels = np.repeat([0, 1], 10)
    units = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8] + list(range(20, 30)))
    folds = stratified_group_folds(labels, units, 5, seed=2)
    assert folds[0] == folds[1]
    for cls in (0, 1):
        assert set(folds[labels == cls]) == set(range(5))
    assert np.array_equal(folds, stratified_group_folds(labels, units, 5, seed=2))


def test_compare_calibration_verdicts():
    y = np.repeat(np.arange(3), 40)
    logits = np.eye(3)[y] * 6.0
    logits[::7] = np.eye(3)[(y[::7] + 1) % 3] * 6.0  # confidently wrong on 1 in 7
    samples = grouped_bootstrap_indices(y, np.arange(len(y)), 300, seed=4)
    same = compare_calibration(y, softmax(logits), softmax(logits), samples, 10, 0.95)
    assert all(v.startswith("no clear change") for v in same["verdict"].values())
    better = compare_calibration(y, softmax(logits), softmax(logits, 3.0), samples, 10, 0.95)
    assert better["verdict"]["nll"].startswith("lower")
    assert better["change"]["nll"]["ci_high"] < 0


# ---------------------------------------------------------------- calibrate.py


def test_cross_fitted_uses_one_temperature_per_fold():
    rng = np.random.default_rng(5)
    y = np.repeat(np.arange(3), 30)
    logits = np.eye(3)[y] * 3 + rng.normal(size=(90, 3)) * 2
    folds = stratified_group_folds(y, np.arange(90), CFG.cv_folds, seed=0)
    probs, temps, bounds = cb.cross_fitted_probs(logits, y, folds, CFG)
    assert probs.shape == logits.shape and len(temps) == len(bounds) == CFG.cv_folds
    for f, t in enumerate(temps):
        np.testing.assert_allclose(probs[folds == f], softmax(logits[folds == f], t))


def test_run_refuses_test_frozen():
    params = copy.deepcopy(PARAMS)
    params["train"]["val_split"] = "test_frozen"
    with pytest.raises(cb.CalibrationError, match="test_frozen"):
        cb.run(params)


def test_check_checkpoint_refuses_another_checkpoint(tmp_path):
    (tmp_path / "m.pt").write_bytes(b"weights")
    (tmp_path / "res").mkdir()
    (tmp_path / "res" / "val_metrics.json").write_text(json.dumps({"checkpoint_sha256": "0" * 64}))
    with pytest.raises(cb.CalibrationError, match="not the checkpoint"):
        cb.check_checkpoint(ModelSpec("m", Path("m.pt"), Path("res")), tmp_path)


def _fake_cal(share_on_bound=0.0, fold_at_bound=(False,) * 5) -> dict:
    report = {"n_errors": 1, "n_images": 180, "verdict": {"ece": "x", "nll": "y", "brier": "z"}}
    return {
        "temperature": 1.2,
        "temperature_interval": {"ci_low": 0.9, "ci_high": 1.5, "share_on_bound": share_on_bound},
        "val": {"cross_fitted": {**report, "fold_at_bound": list(fold_at_bound)}},
    }


def test_summary_says_when_t_is_poorly_determined():
    assert "poorly determined" not in cb.summary_text(_fake_cal())
    text = cb.summary_text(_fake_cal(0.3, (True, False, False, False, False)))
    assert "poorly determined" in text and "30%" in text and "1 of 5" in text


def test_params_range_and_bins():
    c = PARAMS["calibrate"]
    assert 0 < c["t_min"] < 1 < c["t_max"]
    assert c["n_bins"] >= 5 and c["cv_folds"] >= 2


# ---------------------------------------------------------------- committed results


def _models():
    return PARAMS["compare_val"]["models"]


@pytest.mark.parametrize("index", [0, 1])
def test_committed_calibration_is_consistent(index):
    spec = _models()[index]
    res = REPO_ROOT / spec["results_dir"]
    cal = json.loads((res / "calibration.json").read_text(encoding="utf-8"))
    recorded = json.loads((res / "val_metrics.json").read_text(encoding="utf-8"))
    split = Path(PARAMS["data"]["split_file"])
    assert cal["model"] == spec["name"]
    assert cal["fitted_on"]["split"] == "val" == PARAMS["train"]["val_split"]
    assert cal["fitted_on"]["split_file"] == split.name
    assert cal["checkpoint_sha256"] == recorded["checkpoint_sha256"]
    assert cal["at_bound"] is False
    assert CFG.t_min < cal["temperature"] < CFG.t_max
    assert cal["t_range"] == [CFG.t_min, CFG.t_max]
    assert cal["provenance"]["git_dirty"] is False
    for name in ("in_sample", "cross_fitted"):
        rep = cal["val"][name]
        assert rep["reliability_before"]["n_bins"] == CFG.n_bins
        assert sum(rep["reliability_after"]["count"]) == rep["n_images"]
        assert rep["n_errors"] == cal["fitted_on"]["n_errors"]
    rows = (res / "val_predictions.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "path,true,pred,confidence,calibrated_confidence,correct"
    assert len(rows) - 1 == cal["fitted_on"]["n_images"]


def test_mlflow_payload_names():
    cal = json.loads(
        (REPO_ROOT / _models()[0]["results_dir"] / "calibration.json").read_text(encoding="utf-8")
    )
    metrics, tags = cb.mlflow_payload(cal)
    assert {"temperature", "val_ece_before", "val_nll_after_cv", "val_brier_change_cv"} <= set(
        metrics
    )
    assert tags["calibration_decision_record"] == "ADR-006"


@pytest.mark.needs_data
@pytest.mark.needs_model
def test_calibration_reproduces_committed_main_model(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    m = _models()[0]
    if not Path(m["checkpoint"]).exists():
        pytest.skip(f"missing checkpoint {m['checkpoint']}")
    spec = ModelSpec(m["name"], Path(m["checkpoint"]), Path(m["results_dir"]))
    new = cb.calibrate_model(spec, PARAMS, CFG, PARAMS["train"]["val_split"])["cal"]
    old = json.loads((spec.results_dir / "calibration.json").read_text(encoding="utf-8"))
    for key in ("temperature", "at_bound", "temperature_interval", "fitted_on", "val"):
        assert new[key] == old[key], key


# ---------------------------------------------------------------- ADR-037: inherited T


def test_a_fit_on_its_bound_inherits_the_champion_t_only_in_a_retrain():
    champ = {"version": "3", "temperature": 1.5115}
    assert cb.choose_temperature(0.25, True, champ) == (1.5115, cb.INHERITED, False)
    assert cb.choose_temperature(4.0, True, champ) == (1.5115, cb.INHERITED, False)  # upper bound
    assert cb.choose_temperature(1.2, False, champ) == (1.2, cb.FITTED, False)  # fit kept
    assert cb.choose_temperature(0.25, True, None) == (0.25, cb.FITTED, True)  # ADR-006: stop


def test_a_zero_error_val_fit_lands_on_the_lower_bound():
    """Why ADR-037 exists: with no val errors, NLL keeps falling as T shrinks."""
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 20)
    logits = rng.normal(0, 1, (60, 3))
    logits[np.arange(60), y] += 4.0  # every image right
    cfg = cb.CalibConfig.from_params(PARAMS)
    t, at_bound = cb.fit(logits, y, cfg)
    assert at_bound and t == pytest.approx(cfg.t_min)


def test_inherited_payload_tags_its_source():
    cal = json.loads(
        (REPO_ROOT / _models()[0]["results_dir"] / "calibration.json").read_text(encoding="utf-8")
    )
    cal = {**cal, "calibration": cb.INHERITED, "decision_record": "ADR-006, ADR-037"}
    _, tags = cb.mlflow_payload(cal)
    assert (
        tags["calibration_source"] == "inherited_T"
        and "ADR-037" in tags["calibration_decision_record"]
    )
