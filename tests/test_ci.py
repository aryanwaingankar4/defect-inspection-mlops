"""TASK-035: static checks on the GitHub Actions workflow (no network, no Actions run)."""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
WORKFLOW_TEXT = WORKFLOW_PATH.read_text(encoding="utf-8")
WORKFLOW = yaml.safe_load(WORKFLOW_TEXT)
STEPS = [s for job in WORKFLOW["jobs"].values() for s in job["steps"]]


def run_text() -> str:
    return "\n".join(s.get("run", "") for s in STEPS)


def test_runs_on_push_and_pull_request_to_main():
    on = WORKFLOW[True]  # PyYAML reads the key `on` as boolean True
    assert on["push"]["branches"] == ["main"]
    assert on["pull_request"]["branches"] == ["main"]


def test_every_action_is_pinned_to_a_full_commit_sha():
    uses = [s["uses"] for s in STEPS if "uses" in s]
    assert uses
    for ref in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), ref


def test_read_only_token_and_no_secrets():
    assert WORKFLOW["permissions"] == {"contents": "read"}
    assert "secrets." not in WORKFLOW_TEXT
    assert all(s.get("with", {}).get("persist-credentials") is False for s in STEPS if
               s.get("uses", "").startswith("actions/checkout@"))  # fmt: skip


def test_installs_from_the_lockfile_with_cache_and_runs_make_targets():
    setup = next(s for s in STEPS if s.get("uses", "").startswith("astral-sh/setup-uv@"))
    assert setup["with"]["enable-cache"] is True
    assert setup["with"]["cache-dependency-glob"] == "uv.lock"
    text = run_text()
    assert "uv sync --locked" in text
    assert "make lint" in text and "make test" in text


def test_test_job_checks_out_full_history():
    checkout = WORKFLOW["jobs"]["lint-test"]["steps"][0]
    assert checkout["with"]["fetch-depth"] == 0  # test_promote looks up evidence commits


def test_no_dvc_data_or_model_in_ci():
    text = run_text()
    for forbidden in ("dvc pull", "dvc ", "make data", "serving-bundle", "test-all", "kaggle"):
        assert forbidden not in text, forbidden


def test_trivy_is_checksum_verified_and_fails_only_on_fixable_critical():
    env = WORKFLOW["jobs"]["trivy"]["env"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", env["TRIVY_VERSION"])
    assert re.fullmatch(r"[0-9a-f]{64}", env["TRIVY_SHA256"])
    text = run_text()
    assert "sha256sum -c" in text
    fail_lines = [line for line in text.splitlines() if "--exit-code 1" in line]
    assert any("trivy fs" in line and "--ignore-unfixed" in line for line in fail_lines)
    assert any("trivy config" in line for line in fail_lines)
    assert all("--severity CRITICAL " in line for line in fail_lines)


def test_readme_has_the_ci_badge():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "actions/workflows/ci.yml/badge.svg" in readme


def test_postgres_job_migrates_then_runs_the_db_tests():
    """ADR-038: a pinned Postgres service container, `make db-migrate`, then the DB tests."""
    job = WORKFLOW["jobs"]["postgres"]
    image = job["services"]["postgres"]["image"]
    assert re.fullmatch(r"postgres:16\.\d+@sha256:[0-9a-f]{64}", image)
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert compose["services"]["postgres"]["image"] == image  # the same as local tests
    runs = [s.get("run", "") for s in job["steps"]]
    migrate = next(i for i, r in enumerate(runs) if r.startswith("make db-migrate"))
    tests = next(i for i, r in enumerate(runs) if "TEST_DATABASE_URL" in r)
    assert migrate < tests and "tests/test_migrations.py" in runs[tests]
    assert job["env"]["PG_TEST_URL"].startswith("postgresql+psycopg://")
    assert "@127.0.0.1:" in job["env"]["PG_TEST_URL"]
