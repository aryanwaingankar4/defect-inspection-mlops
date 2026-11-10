"""Shared scoring: calibrated confidence, energy, routing (ADR-023, ADR-025, ADR-027)."""

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from src.serving import scoring as sc
from src.training.metrics import softmax


def test_calibrated_probs_match_the_training_softmax():
    logits = np.random.default_rng(0).normal(size=(20, 6)) * 8
    for t in (0.5, 1.0, 1.64):
        np.testing.assert_allclose(sc.calibrated_probs(logits, t), softmax(logits, t))


def test_energy_hand_values():
    logits = np.array([[0.0, 0.0], [10.0, 0.0]])
    np.testing.assert_allclose(
        sc.energy(logits, 1.0), [-math.log(2), -(10 + math.log(1 + math.exp(-10)))]
    )
    # with T: -T * logsumexp(logits / T)
    assert sc.energy(np.array([[2.0, 4.0]]), 2.0)[0] == pytest.approx(
        -2 * math.log(math.e + math.e**2)
    )


def test_energy_is_stable_for_large_logits():
    e = sc.energy(np.array([[1000.0, 999.0]]), 1.0)
    assert np.isfinite(e).all() and e[0] == pytest.approx(-(1000 + math.log(1 + math.exp(-1))))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_temperature_must_be_positive(bad):
    with pytest.raises(ValueError):
        sc.energy(np.zeros((1, 2)), bad)


def test_score_routes_on_low_confidence_or_ood():
    th = sc.Thresholds(temperature=1.0, review_threshold=0.9, ood_energy_threshold=-5.0)
    logits = np.array([[10.0, 0.0], [0.5, 0.0], [3.0, 0.0], [0.1, 0.0]])
    s = sc.score(logits, th)
    # image 0: confident, low energy; 1: low confidence and high energy;
    # 2: confident enough (0.95) but energy -3.05 > -5 -> ood; 3: both
    assert s.pred.tolist() == [0, 0, 0, 0]
    assert s.low_confidence.tolist() == [False, True, False, True]
    assert s.ood.tolist() == [False, True, True, True]
    assert s.routed.tolist() == [False, True, True, True]
    assert s.reasons(0) == [] and s.reasons(2) == ["ood"]
    assert s.reasons(1) == ["low_confidence", "ood"]


def test_scoring_imports_only_numpy():
    tree = ast.parse(Path(sc.__file__).read_text(encoding="utf-8"))
    names = {
        a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    names |= {
        n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    }
    assert names <= {"__future__", "dataclasses", "numpy"}
