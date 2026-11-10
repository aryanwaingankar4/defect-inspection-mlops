"""TASK-021: the API image and compose file. Fast static checks (CI); the real container check is
`make docker-smoke` (manual, needs Docker)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
IGNORE = [
    line.strip()
    for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]
COMPOSE = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
API = COMPOSE["services"]["api"]


def instructions(name: str) -> list[str]:
    """Arguments of every ``name`` instruction (continuation lines joined)."""
    text = re.sub(r"\\\n\s*", " ", DOCKERFILE)
    return [m.group(1).strip() for m in re.finditer(rf"^{name}\s+(.*)$", text, re.MULTILINE)]


def final_stage() -> str:
    return DOCKERFILE.split("\nFROM ")[-1]


def test_runs_as_a_non_root_user():
    users = instructions("USER")
    assert users and users[-1].split(":")[0] not in ("0", "root")
    assert "USER" in final_stage()


def test_installs_only_the_serve_group_from_the_lockfile():
    runs = " ".join(instructions("RUN"))
    assert "uv sync --locked --no-default-groups --group serve" in runs
    assert "--group train" not in runs and "--group dev" not in runs
    assert "pip install" not in runs


def test_copies_only_the_api_files():
    sources = []
    for args in instructions("COPY"):
        if args.startswith("--from="):
            continue
        sources += args.split()[:-1]
    allowed = {
        "pyproject.toml",
        "uv.lock",
        "src/__init__.py",
        "src/config.py",
        "src/db/",
        "src/serving/",
        "configs/params.yaml",
        "models/serving/",
    }
    assert set(sources) <= allowed, set(sources) - allowed


def test_no_secrets_or_data_baked_in():
    for name in ("API_KEY", "DATABASE_URL", "MLFLOW_", "KAGGLE"):
        assert not re.search(rf"^(ENV|ARG)\s+.*{name}", DOCKERFILE, re.MULTILINE), name
    assert "data/" not in " ".join(instructions("COPY"))


def test_healthcheck_calls_health():
    checks = instructions("HEALTHCHECK")
    assert checks and "/health" in checks[0]


def test_dockerignore_is_an_allowlist():
    assert IGNORE[0] == "*"
    reincluded = {line[1:] for line in IGNORE if line.startswith("!")}
    assert reincluded == {
        "pyproject.toml",
        "uv.lock",
        "src/__init__.py",
        "src/config.py",
        "src/db/",
        "src/serving/",
        "configs/params.yaml",
        "models/serving/",
    }
    for never in ("data", ".env", "results", "mlruns", "build", "models/onnx"):
        assert not any(r.startswith(never) for r in reincluded), never


def test_compose_passes_only_the_needed_environment():
    assert "env_file" not in API
    env = API["environment"]
    assert set(env) == {"API_KEY", "DATABASE_URL"}
    assert env["API_KEY"].startswith("${API_KEY:?")  # required, never a default value
    assert env["DATABASE_URL"] == "sqlite:////data/predictions.db"


def test_compose_volume_ports_and_hardening():
    assert "predictions:/data" in API["volumes"] and "predictions" in COMPOSE["volumes"]
    assert all(p.startswith("127.0.0.1:") for p in API["ports"])  # local only
    assert API["read_only"] is True and any(t.startswith("/tmp") for t in API["tmpfs"])
    assert API["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in API["security_opt"]


@pytest.mark.slow
@pytest.mark.manual
def test_docker_smoke():
    """Builds nothing itself: run `make docker-build` first (or just `make docker-smoke`)."""
    if shutil.which("docker") is None:
        pytest.skip("docker not installed")
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip("docker engine not running")
    images = subprocess.run(
        ["docker", "image", "ls", "-q", "defect-inspection-api:local"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not images:
        pytest.skip("image not built: run `make docker-build`")
    r = subprocess.run(
        ["python", "scripts/docker_smoke.py"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr
