"""Pure helpers of scripts/azure.py (ADR-039). No Azure call, no network."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("azure_deploy", ROOT / "scripts" / "azure.py")
azure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(azure)


def test_database_url_encodes_password_and_requires_tls() -> None:
    url = azure.database_url("admin", "p@ss:w/rd#1?", "srv.postgres.database.azure.com", "defect")
    assert url.startswith("postgresql+psycopg://admin:")
    assert "p%40ss%3Aw%2Frd%231%3F@srv.postgres.database.azure.com:5432/defect" in url
    assert url.endswith("?sslmode=require")


def _want() -> dict:
    images = json.loads((ROOT / "tests" / "golden" / "expected.json").read_text("utf-8"))
    return next(e for e in images["champion"]["images"] if e["id"] == "synthetic")


def test_golden_match_ignores_only_the_random_audit() -> None:
    want = _want()
    pred = {k: want[k] for k in ("prediction", "calibrated_confidence", "reasons")}
    assert azure.golden_matches(pred, want)
    assert azure.golden_matches({**pred, "reasons": [*want["reasons"], "audit"]}, want)
    assert not azure.golden_matches({**pred, "reasons": ["ood"]}, want)
    assert not azure.golden_matches({**pred, "calibrated_confidence": 0.5}, want)
    assert not azure.golden_matches({**pred, "prediction": "crazing"}, want)


def test_latency_summary_p50() -> None:
    out = azure.latency_summary([30.0, 10.0, 20.0, 40.0])
    assert out == {"n": 4, "p50_ms": 25.0, "min_ms": 10.0, "max_ms": 40.0}


def test_report_refuses_secret_values() -> None:
    azure.assert_no_secrets('{"url": "https://x"}', ["s3cret", ""])
    with pytest.raises(azure.DeployError):
        azure.assert_no_secrets('{"note": "s3cret"}', ["s3cret"])


def test_bicep_has_no_secret_defaults_and_locked_settings() -> None:
    text = "\n".join(
        (ROOT / "infra" / f).read_text("utf-8") for f in ("main.bicep", "postgres.bicep")
    )
    params = json.loads((ROOT / "infra" / "main.parameters.json").read_text("utf-8"))
    for name in ("pgAdminPassword", "apiKey", "auditSalt", "databaseUrl"):
        assert name not in params["parameters"]
    for needle in (
        "adminUserEnabled: false",
        "minReplicas: 0",
        "maxReplicas: 1",
        "'Standard_B1ms'",
        "version: '16'",
        "name: 'require_secure_transport'",
        "targetPort: 8000",
        "path: '/health'",
    ):
        assert needle in text, needle
