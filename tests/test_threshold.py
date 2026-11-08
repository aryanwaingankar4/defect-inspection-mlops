"""TASK-013: review threshold on val (ADR-023). Fast tests never read images."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from src.config import load_params  # noqa: E402
from src.tracking.artifact_policy import is_allowed_artifact  # noqa: E402
from src.training import evaluate as ev  # noqa: E402
from src.training import threshold as th  # noqa: E402
from src.training.compare_val import ModelSpec  # noqa: E402
from src.training.metrics import (  # noqa: E402
    binomial_cdf,
    budget_threshold,
    clopper_pearson_upper,
    grouped_bootstrap_indices,
    risk_at_coverage,
    risk_coverage_curve,
    risk_curve_bootstrap,
    threshold_bootstrap,
)
from src.training.train import sha256_file  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
CFG = th.ThresholdConfig.from_params(PARAMS)


# ---------------------------------------------------------------- Clopper-Pearson


def test_binomial_cdf_edges_and_a_hand_value():
    assert binomial_cdf(-1, 5, 0.3) == 0.0 and binomial_cdf(5, 5, 0.3) == 1.0
    assert binomial_cdf(1, 3, 0.5) == pytest.approx(0.5)  # (1 + 3) / 8


def test_clopper_pearson_zero_errors_has_a_closed_form():
    for n in (10, 171, 342):
        assert clopper_pearson_upper(0, n, 0.95) == pytest.approx(1 - 0.05 ** (1 / n), abs=1e-12)


def test_clopper_pearson_known_values_and_monotone():
    # scipy.stats.beta.ppf(0.95, k + 1, 171 - k), computed once and pasted
    for k, expected in enumerate([0.0174, 0.0274, 0.0364, 0.0447, 0.0527]):
        assert clopper_pearson_upper(k, 171, 0.95) == pytest.approx(expected, abs=5e-5)
    bounds = [clopper_pearson_upper(k, 50, 0.95) for k in range(51)]
    assert all(a < b for a, b in zip(bounds, bounds[1:], strict=False))
    assert clopper_pearson_upper(50, 50, 0.95) == 1.0
    # the bound solves P(X <= k | n, p) = 1 - level
    assert binomial_cdf(3, 171, clopper_pearson_upper(3, 171, 0.95)) == pytest.approx(0.05)


@pytest.mark.parametrize(("k", "n", "level"), [(-1, 5, 0.95), (6, 5, 0.95), (1, 5, 1.0)])
def test_clopper_pearson_rejects_bad_input(k, n, level):
    with pytest.raises(ValueError):
        clopper_pearson_upper(k, n, level)


def test_images_needed_for_one_percent():
    assert th.images_needed(0.01, 0.95) == 299
    assert clopper_pearson_upper(0, 299, 0.95) <= 0.01 < clopper_pearson_upper(0, 298, 0.95)


# ---------------------------------------------------------------- threshold + curve


def test_budget_threshold_routes_ceil_budget_times_n():
    conf = np.linspace(0.5, 1.0, 180)[::-1]
    t, m = budget_threshold(conf, 0.05)
    assert m == 9 and int((conf < t).sum()) == 9
    s = np.sort(conf)
    assert t == pytest.approx((s[8] + s[9]) / 2)
    assert budget_threshold(np.arange(10.0), 0.15)[1] == 2  # ceil(1.5)


def test_budget_threshold_refuses_ties_and_bad_budgets():
    with pytest.raises(ValueError, match="tied"):
        budget_threshold(np.array([0.1, 0.5, 0.5, 0.9]), 0.5)
    for bad in (0.0, 1.0, 0.95):  # 0.95 of 10 rounds up to all 10 images
        with pytest.raises(ValueError):
            budget_threshold(np.arange(10.0), bad)


def test_risk_coverage_curve_hand_example_with_a_tie():
    conf = np.array([0.9, 0.8, 0.8, 0.6, 0.5])
    correct = np.array([True, False, True, True, False])
    c = risk_coverage_curve(conf, correct)
    assert c["threshold"].tolist() == [0.9, 0.8, 0.6, 0.5]
    assert c["coverage"].tolist() == [0.2, 0.6, 0.8, 1.0]
    np.testing.assert_allclose(c["risk"], [0.0, 1 / 3, 1 / 4, 2 / 5])
    np.testing.assert_allclose(
        risk_at_coverage(conf, correct, np.array([0.2, 0.4, 1.0])), [0.0, 0.5, 0.4]
    )


def test_bootstraps_on_the_full_sample_equal_the_point_values():
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([True, True, False, True, False])
    full = [np.arange(5)]
    b = threshold_bootstrap(conf, correct, 0.65, full)
    assert b["coverage"][0] == 0.6 and b["risk"][0] == pytest.approx(1 / 3)
    grid = np.array([0.6, 1.0])
    np.testing.assert_allclose(risk_curve_bootstrap(conf, correct, grid, full)[0], [1 / 3, 0.4])


def test_review_report_counts_and_bounds():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pred = np.array([0, 0, 1, 0, 1, 1, 0, 1])  # errors at 2 and 6
    conf = np.array([0.99, 0.98, 0.97, 0.60, 0.95, 0.94, 0.55, 0.93])
    units = np.array([0, 0, 1, 2, 3, 4, 5, 6])  # images 0 and 1 are one near-copy group
    samples = grouped_bootstrap_indices(y, units, 200, seed=0)
    r = ev.review_report(y, pred, conf, 0.7, units, samples, 0.95, 0.95)
    assert (r["n_routed"], r["n_accepted"], r["routed_share"]) == (2, 6, 0.25)
    assert (r["n_errors"], r["errors_routed"], r["errors_accepted"]) == (2, 1, 1)
    assert r["risk"]["value"] == pytest.approx(1 / 6)
    assert r["risk"]["ci_low"] <= r["risk"]["value"] <= r["risk"]["ci_high"]
    assert r["risk_ucb"]["value"] == pytest.approx(clopper_pearson_upper(1, 6, 0.95))
    g = r["risk_ucb_by_group"]
    assert (g["n_units_accepted"], g["n_units_with_error"]) == (5, 1)
    assert g["value"] == pytest.approx(clopper_pearson_upper(1, 5, 0.95))


def test_coverage_grid():
    grid = CFG.coverage_grid()
    assert grid[0] == CFG.min_coverage and grid[-1] == 1.0
    np.testing.assert_allclose(np.diff(grid), CFG.coverage_step)


# ---------------------------------------------------------------- run guards


def test_build_refuses_test_frozen():
    params = copy.deepcopy(PARAMS)
    params["train"]["val_split"] = "test_frozen"
    with pytest.raises(th.ThresholdError, match="test_frozen"):
        th.build(params, CFG)


def test_main_spec_must_be_the_main_model():
    params = copy.deepcopy(PARAMS)
    params["train"]["model"] = "something_else"
    with pytest.raises(th.ThresholdError, match="main model"):
        th.main_spec(params)


def _fake_result(passed=True) -> dict:
    grid = CFG.coverage_grid()
    val = {
        "threshold": 0.8,
        "n_images": 180,
        "n_routed": 9,
        "n_accepted": 171,
        "routed_share": 0.05,
        "n_errors": 4,
        "errors_routed": 3,
        "errors_accepted": 1,
        "coverage": {"value": 0.95, "ci_low": 0.93, "ci_high": 0.97},
        "risk": {"value": 1 / 171, "ci_low": 0.0, "ci_high": 0.02},
        "risk_ucb": {"value": 0.0274 if passed else 0.06, "level": 0.95},
        "risk_ucb_by_group": {"value": 0.03},
    }
    return {
        "model": "m1",
        "threshold": 0.8,
        "passed": passed,
        "rule": {"review_budget": 0.05},
        "fitted_on": {"split_file": "split.csv"},
        "val": val,
        "curve": {"coverage": [0.5, 0.95, 1.0], "risk": [0.0, 1 / 171, 4 / 180]},
        "bands": {
            "level": 0.95,
            "coverage": grid.tolist(),
            "ci_low": [0.0] * len(grid),
            "ci_high": [0.03] * len(grid),
        },
        "provenance": {"git_commit": "abc"},
    }


def test_run_stops_and_writes_nothing_above_max_risk_ucb(tmp_path, monkeypatch):
    spec = ModelSpec("m1", tmp_path / "m1.pt", tmp_path)
    monkeypatch.setattr(th, "build", lambda p, c: {"spec": spec, "result": _fake_result(False)})
    with pytest.raises(th.ThresholdError, match="owner decides"):
        th.run(PARAMS)
    assert list(tmp_path.iterdir()) == []


def test_write_outputs_plot_is_allowlisted(tmp_path):
    files = th.write_outputs(ModelSpec("m1", tmp_path / "m.pt", tmp_path), _fake_result(), CFG)
    assert all(p.is_file() for p in files.values())
    assert is_allowed_artifact(f"{ev.PLOT_DIR}/{files['plot'].name}")
    assert b"\r\n" not in files["threshold"].read_bytes()


def test_claims_state_the_limits():
    c = th.claims(_fake_result()["val"], CFG, 15)
    text = " ".join(c["cannot_claim"])
    assert "299" in text and "15 near-copy groups" in text and "ADR-007" in text
    assert "3 of the 4 val errors" in c["can_claim"][0]


def test_params_rule():
    t = PARAMS["threshold"]
    assert 0 < t["review_budget"] < 1 and 0 < t["max_risk_ucb"] < 1
    assert 0 < t["ucb_level"] < 1 and 0 < t["min_coverage"] < 1


# ---------------------------------------------------------------- committed results


def _main() -> dict:
    return PARAMS["compare_val"]["models"][0]


def _committed() -> dict:
    path = REPO_ROOT / _main()["results_dir"] / CFG.out_file
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_threshold_is_consistent():
    m, r = _main(), _committed()
    res = REPO_ROOT / m["results_dir"]
    recorded = json.loads((res / "val_metrics.json").read_text(encoding="utf-8"))
    split = REPO_ROOT / PARAMS["data"]["split_file"]
    assert r["model"] == m["name"] == PARAMS["train"]["model"]
    assert r["passed"] is True
    assert r["rule"]["review_budget"] == CFG.review_budget
    assert r["rule"]["max_risk_ucb"] == CFG.max_risk_ucb
    assert r["rule"]["ucb_level"] == CFG.ucb_level
    assert r["calibration_sha256"] == sha256_file(res / "calibration.json")
    assert r["checkpoint_sha256"] == recorded["checkpoint_sha256"]
    assert r["fitted_on"]["split"] == "val" and r["fitted_on"]["split_sha256"] == sha256_file(split)
    assert r["provenance"]["git_dirty"] is False
    v = r["val"]
    assert v["n_routed"] == math.ceil(round(CFG.review_budget * v["n_images"], 9))
    assert v["risk_ucb"]["value"] <= CFG.max_risk_ucb
    assert v["errors_routed"] + v["errors_accepted"] == v["n_errors"] == r["fitted_on"]["n_errors"]
    assert r["bands"]["coverage"] == CFG.coverage_grid().tolist()
    assert r["curve"]["coverage"][-1] == 1.0
    assert is_allowed_artifact(f"{ev.PLOT_DIR}/{th.PLOT_NAME}")
    assert (res / th.PLOT_NAME).is_file()


def test_mlflow_payload_names():
    metrics, tags = th.mlflow_payload(_committed())
    assert {"review_threshold", "val_review_risk_ucb", "val_review_coverage"} <= set(metrics)
    assert all(isinstance(v, float) for v in metrics.values())
    assert tags["threshold_decision_record"] == "ADR-023"


@pytest.mark.needs_data
@pytest.mark.needs_model
def test_threshold_reproduces_committed(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    if not Path(_main()["checkpoint"]).exists():
        pytest.skip(f"missing checkpoint {_main()['checkpoint']}")
    new, old = th.build(PARAMS, CFG)["result"], _committed()
    for key in ("threshold", "passed", "fitted_on", "val", "curve", "bands", "claims"):
        assert new[key] == old[key], key
