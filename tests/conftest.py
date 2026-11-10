"""Shared fixtures. Tests must never reach DagsHub (TASK-009)."""

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

LOCAL_PREFIXES = ("sqlite:", "file:")
TEST_AUDIT_SALT = "test-audit-salt-not-secret"  # fixed, so audit decisions are reproducible


@pytest.fixture(autouse=True)
def audit_salt(monkeypatch: pytest.MonkeyPatch) -> str:
    """Every test gets the same non-secret audit salt (ADR-030); tests never read .env."""
    monkeypatch.setenv("AUDIT_SALT", TEST_AUDIT_SALT)
    return TEST_AUDIT_SALT


@pytest.fixture(autouse=True)
def offline_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Strip MLflow/DagsHub settings from the environment and use a per-test local store."""
    for name in [k for k in os.environ if k.startswith(("MLFLOW_", "DAGSHUB_"))]:
        monkeypatch.delenv(name)
    uri = "sqlite:///" + (tmp_path / "mlruns" / "mlflow.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    yield uri
    mlflow = sys.modules.get("mlflow")
    if mlflow is not None:
        current = mlflow.get_tracking_uri()
        # The URI itself is not printed: it could carry credentials.
        assert current.startswith(LOCAL_PREFIXES), "a test used a non-local MLflow URI"
