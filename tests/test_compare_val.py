"""Paired McNemar model comparison on val (ADR-022)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.config import load_params
from src.training.compare_val import (
    CompareError,
    ModelSpec,
    adr022_outcome,
    check_reproduces,
    paired_table,
    run,
)
from src.training.metrics import mcnemar_exact

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
REPORT = REPO_ROOT / PARAMS["compare_val"]["out"]


@pytest.mark.parametrize(
    ("a_only", "b_only", "expected"),
    [
        (0, 0, 1.0),
        (1, 4, 12 / 32),  # 2 * (C(5,0) + C(5,1)) / 2^5
        (0, 6, 2 / 64),
        (5, 5, 1.0),  # clamped at 1
        (10, 0, 2 / 1024),
    ],
)
def test_mcnemar_exact_known_values(a_only, b_only, expected):
    assert mcnemar_exact(a_only, b_only) == pytest.approx(expected)
    assert mcnemar_exact(b_only, a_only) == pytest.approx(expected)  # symmetric


def test_mcnemar_exact_matches_binomial_sum():
    n, k = 25, 7
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    assert mcnemar_exact(k, n - k) == pytest.approx(2 * tail)


def test_mcnemar_exact_rejects_negative_counts():
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 3)


def test_paired_table_counts_each_cell():
    y = np.array([0, 1, 2, 3, 4, 5])
    a = np.array([0, 1, 2, 0, 0, 0])  # right on 0,1,2
    b = np.array([0, 0, 2, 3, 0, 0])  # right on 0,2,3
    t = paired_table(y, a, b)
    assert (t.both_right, t.a_only, t.b_only, t.both_wrong) == (2, 1, 1, 2)


def test_paired_table_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        paired_table(np.zeros(3), np.zeros(3), np.zeros(2))


def test_outcome_follows_adr022():
    names = ["main", "alt"]
    assert "main stays the main model" in adr022_outcome(0.05, 0.05, names)
    assert "owner decides" in adr022_outcome(0.049, 0.05, names)


def test_run_refuses_test_frozen():
    params = {**PARAMS, "compare_val": {**PARAMS["compare_val"], "split": "test_frozen"}}
    with pytest.raises(CompareError, match="test_frozen"):
        run(params)


def test_check_reproduces_refuses_mismatches(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"weights")
    spec = ModelSpec("m", ckpt, tmp_path)
    recorded = {"val_accuracy": 0.9, "val_macro_f1": 0.8}
    check_reproduces(spec, recorded, 0.9, 0.8, 1e-6)  # matches: no error
    with pytest.raises(CompareError, match="val_macro_f1"):
        check_reproduces(spec, recorded, 0.9, 0.81, 1e-6)
    with pytest.raises(CompareError, match="not the checkpoint"):
        check_reproduces(spec, {**recorded, "checkpoint_sha256": "0" * 64}, 0.9, 0.8, 1e-6)


def test_params_put_the_predeclared_main_model_first():
    cfg = PARAMS["compare_val"]
    assert cfg["split"] == "val"
    assert cfg["models"][0]["name"] == PARAMS["train"]["model"]


def test_committed_report_is_consistent():
    r = json.loads(REPORT.read_text(encoding="utf-8"))
    names = [m["name"] for m in r["models"]]
    a_only, b_only = (r["table"][f"only_{n}_right"] for n in names)
    assert r["split"] == "val"
    assert r["split_file"] == Path(PARAMS["data"]["split_file"]).name
    assert sum(r["table"].values()) == r["n_images"]
    assert len(r["discordant_images"]) == r["n_discordant"] == a_only + b_only
    assert r["p_value"] == pytest.approx(mcnemar_exact(a_only, b_only))
    assert r["alpha"] == PARAMS["compare_val"]["alpha"]
    assert r["significant"] == (r["p_value"] < r["alpha"])
    assert r["outcome"] == adr022_outcome(r["p_value"], r["alpha"], names)
    for m in r["models"]:
        recorded = json.loads(
            (REPO_ROOT / m["results_dir"] / "val_metrics.json").read_text(encoding="utf-8")
        )
        assert m["checkpoint_sha256"] == recorded["checkpoint_sha256"]
        assert m["macro_f1"] == pytest.approx(recorded["val_macro_f1"], abs=1e-6)


@pytest.mark.needs_data
@pytest.mark.needs_model
def test_run_reproduces_committed_report(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    for m in PARAMS["compare_val"]["models"]:
        if not Path(m["checkpoint"]).exists():
            pytest.skip(f"missing checkpoint {m['checkpoint']}")
    r = run(PARAMS)
    committed = json.loads(REPORT.read_text(encoding="utf-8"))
    assert r["table"] == committed["table"]
    assert r["p_value"] == committed["p_value"]
    assert r["discordant_images"] == committed["discordant_images"]
