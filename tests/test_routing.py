"""TASK-023: random audit selection (ADR-007, ADR-030)."""

import random
import uuid
from pathlib import Path

import pytest

from src.config import load_params
from src.serving import routing as rt

REPO_ROOT = Path(__file__).resolve().parents[1]
RATE = load_params(REPO_ROOT / "configs" / "params.yaml")["serving"]["audit_rate"]
SALT = "test-audit-salt-not-secret"


def request_ids(n: int, seed: int = 0) -> list[str]:
    """Server-style random UUID hex ids, from a seeded generator so the test is reproducible."""
    rng = random.Random(seed)
    return [uuid.UUID(int=rng.getrandbits(128), version=4).hex for _ in range(n)]


def test_params_rate_is_the_owner_decision():
    assert RATE == 0.02


def test_draw_is_deterministic_in_range_and_depends_on_the_salt():
    rid = request_ids(1)[0]
    assert rt.audit_draw(rid, SALT) == rt.audit_draw(rid, SALT)
    assert 0.0 <= rt.audit_draw(rid, SALT) < 1.0
    assert rt.audit_draw(rid, SALT) != rt.audit_draw(rid, SALT + "x")


def test_audit_rate_over_a_large_simulated_batch():
    """TEST_PLAN: the sampled share matches params within tolerance (4 binomial SDs)."""
    n = 100_000
    hits = sum(rt.audit_selected(r, SALT, RATE) for r in request_ids(n))
    sd = (RATE * (1 - RATE) / n) ** 0.5
    assert abs(hits / n - RATE) < 4 * sd


def test_different_salts_select_different_requests():
    ids = request_ids(20_000, seed=1)
    a = {r for r in ids if rt.audit_selected(r, SALT, RATE)}
    b = {r for r in ids if rt.audit_selected(r, "another-salt", RATE)}
    assert a and b and len(a & b) < 0.2 * min(len(a), len(b))  # about RATE overlap expected


@pytest.mark.parametrize(("rate", "expected"), [(0.0, False), (1.0, True)])
def test_rate_edges(rate, expected):
    assert all(rt.audit_selected(r, SALT, rate) is expected for r in request_ids(100))


@pytest.mark.parametrize("rate", [-0.01, 1.01])
def test_rate_must_be_a_share(rate):
    with pytest.raises(ValueError):
        rt.audit_selected("a" * 32, SALT, rate)


def test_only_otherwise_accepted_predictions_are_audited():
    rid = request_ids(1)[0]
    assert rt.with_audit([], rid, SALT, 1.0) == ["audit"]
    assert rt.with_audit([], rid, SALT, 0.0) == []
    assert rt.with_audit(["low_confidence"], rid, SALT, 1.0) == ["low_confidence"]
    assert rt.with_audit(["low_confidence", "ood"], rid, SALT, 1.0) == ["low_confidence", "ood"]


def test_routing_uses_only_the_standard_library():
    import ast

    tree = ast.parse(Path(rt.__file__).read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert names <= {"__future__", "hashlib"}
