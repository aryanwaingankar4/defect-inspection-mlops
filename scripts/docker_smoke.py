"""`make docker-smoke` (TASK-021): start the built API image with docker compose and check it.

Uses a separate compose project, a random throwaway API key and its own volume, and removes
them at the end. Sends only the committed synthetic golden image (ADR-017): no NEU file goes
into or near the container. Checks:
health check healthy, /health, /predict (= tests/golden/expected.json), 401 without key,
non-root user, no torch/mlflow inside, read-only filesystem, one logged row that survives a
restart, the key not in the logs, image labels; and (ADR-038) that the API refuses to start on a
database that was never migrated, while the one-shot ``migrate`` service brings the compose
database to head first.
On success it writes ``results/docker/image_report.json`` (image tag, size, user id, read-only
root filesystem, the git commit the image was built from).

``--postgres``: the same checks with the API on the compose test Postgres (profile ``postgres``,
fixed non-secret local password, ADR-038); writes ``results/docker/image_report_postgres.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "defect-inspection-smoke"
PORT = "18000"
IMAGE = "defect-inspection-api:local"
REPORTS = {
    "sqlite": ROOT / "results" / "docker" / "image_report.json",
    "postgres": ROOT / "results" / "docker" / "image_report_postgres.json",
}
DATABASES = {
    "sqlite": "sqlite:////data/predictions.db",
    # LOCAL TEST database only (ADR-038): fixed, non-secret password, compose network only
    "postgres": "postgresql+psycopg://defect:defect-local-test@postgres:5432/defect",
}
PG_PORT = "15433"  # host port of the smoke Postgres (not the 5433 of `make pg-up`)
PROFILE: list[str] = []
BASE = f"http://127.0.0.1:{PORT}"
GOLDEN = ROOT / "tests" / "golden"
ROW_COUNT = (
    "import os; from src.db.store import PredictionLog; "
    "print(PredictionLog(os.environ['DATABASE_URL']).count())"
)
REVISION = (
    "import os; from src.db import migrate; from src.db.store import make_engine; "
    "print(migrate.check_at_head(make_engine(os.environ['DATABASE_URL'])))"
)


class SmokeError(RuntimeError):
    pass


def compose(*args: str, env: dict[str, str], check: bool = True) -> str:
    cmd = ["docker", "compose", "-p", PROJECT, *PROFILE, *args]
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if check and out.returncode != 0:
        raise SmokeError(f"`docker compose {' '.join(args)}` failed:\n{out.stderr[-2000:]}")
    return out.stdout + out.stderr


def exec_in(env: dict[str, str], *cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", "-p", PROJECT, *PROFILE, "exec", "-T", "api", *cmd],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def wait_healthy(env: dict[str, str], timeout: float = 90.0) -> float:
    container = compose("ps", "-q", "api", env=env).strip()
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        status = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", container],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status == "healthy":
            return time.monotonic() - start
        if status == "unhealthy":
            raise SmokeError("container health check reports unhealthy")
        time.sleep(1)
    raise SmokeError(f"not healthy after {timeout:.0f} s")


def request(path: str, *, data: bytes | None = None, headers: dict[str, str] | None = None):
    req = urllib.request.Request(BASE + path, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def multipart(png: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="g.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode()
        + png
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return body, f"multipart/form-data; boundary={boundary}"


def check(ok: bool, label: str, results: list[str]) -> None:
    results.append(f"  [{'ok' if ok else 'FAIL'}] {label}")
    if not ok:
        print("\n".join(results))
        raise SmokeError(label)


def refuses_unmigrated(key: str) -> tuple[bool, str]:
    """The image on an empty SQLite file (never migrated) must exit with the ADR-038 error."""
    out = subprocess.run(
        [
            "docker", "run", "--rm", "--read-only", "--tmpfs", "/tmp:size=16m",
            "-e", f"API_KEY={key}", "-e", "AUDIT_SALT=smoke-salt",
            "-e", "DATABASE_URL=sqlite:////tmp/never-migrated.db", IMAGE,
        ],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    text = out.stdout + out.stderr
    return out.returncode != 0 and "make db-migrate" in text, text[-300:]


def run(backend: str = "sqlite") -> list[str]:
    key = secrets.token_hex(16)  # throwaway: never printed
    report_file = REPORTS[backend]
    # compose checks every required variable, also for services this smoke does not start
    env = {
        **os.environ,
        "API_KEY": key,
        "API_PORT": PORT,
        "GRAFANA_ADMIN_PASSWORD": secrets.token_hex(16),
        "API_DATABASE_URL": DATABASES[backend],
        "POSTGRES_PORT": PG_PORT,
    }
    out: list[str] = []
    labels = json.loads(
        subprocess.run(
            ["docker", "image", "inspect", IMAGE, "--format", "{{json .Config.Labels}}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    size = subprocess.run(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Size}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    version = labels["io.defect-inspection.model-version"]
    out.append(
        f"image {IMAGE}: {int(size) / 1e6:.0f} MB, model v{version}"
        f", onnx {labels['io.defect-inspection.onnx-sha256'][:12]}..., "
        f"commit {labels['org.opencontainers.image.revision']}"
    )
    refused, text = refuses_unmigrated(key)
    check(refused, "API refuses to start on a never-migrated database (ADR-038)", out)
    try:
        compose("up", "-d", "--no-build", "api", env=env)  # migrate (and postgres) run first
        out.append(f"  healthy after {wait_healthy(env):.1f} s (Docker HEALTHCHECK)")
        rev = exec_in(env, "python", "-c", REVISION).stdout.strip()
        check(rev != "", f"{backend} database migrated to head by the migrate service ({rev})", out)
        status, health = request("/health")
        check(status == 200 and health["database"] == "ok", f"/health 200: {health}", out)

        png = (GOLDEN / "synthetic_golden.png").read_bytes()
        want = next(
            e
            for e in json.loads((GOLDEN / "expected.json").read_text("utf-8"))["champion"]["images"]
            if e["id"] == "synthetic"
        )
        body, ctype = multipart(png)
        status, pred = request(
            "/predict", data=body, headers={"Content-Type": ctype, "X-API-Key": key}
        )
        same = (
            status == 200
            and pred["prediction"] == want["prediction"]
            and pred["reasons"] == want["reasons"]
            and abs(pred["calibrated_confidence"] - want["calibrated_confidence"]) < 1e-4
        )
        check(
            same,
            f"/predict golden image -> {pred.get('prediction')} "
            f"{pred.get('calibrated_confidence', 0):.4f} {pred.get('reasons')} "
            "(matches expected.json)",
            out,
        )
        status, err = request("/predict", data=body, headers={"Content-Type": ctype})
        check(
            status == 401 and err["error"] == "missing_api_key", "/predict without key -> 401", out
        )

        uid = exec_in(env, "id", "-u").stdout.strip()
        check(uid not in ("", "0"), f"runs as non-root (uid {uid})", out)
        no_ml = exec_in(env, "python", "-c", "import torch").returncode != 0
        no_ml &= exec_in(env, "python", "-c", "import mlflow").returncode != 0
        check(no_ml, "torch and mlflow are not installed in the image", out)
        ro = exec_in(env, "sh", "-c", "touch /app/x").returncode != 0
        check(ro, "root filesystem is read-only (/app not writable)", out)
        no_data = exec_in(env, "sh", "-c", "test ! -e /app/data && test ! -e /app/.env")
        check(no_data.returncode == 0, "no data/ and no .env inside the image", out)

        rows = exec_in(env, "python", "-c", ROW_COUNT).stdout.strip()
        check(rows == "1", f"prediction log has {rows} row", out)
        compose("restart", "api", env=env)
        wait_healthy(env)
        rows = exec_in(env, "python", "-c", ROW_COUNT).stdout.strip()
        check(rows == "1", f"after restart the volume still has {rows} row", out)

        logs = compose("logs", "api", env=env)
        check(key not in logs, "API key not in the container logs", out)
        check(pred["request_id"] in logs, "request_id in the JSON logs", out)
        report = {
            "image": IMAGE,
            "size_mb": round(int(size) / 1e6, 1),
            "size_source": "docker image inspect .Size",
            "user_id": int(uid),
            "read_only_root_filesystem": ro,
            "git_commit": labels["org.opencontainers.image.revision"],
            "model_version": version,
            "onnx_sha256": labels["io.defect-inspection.onnx-sha256"],
            "smoke_passed": True,
            "checked_by": "make docker-smoke (scripts/docker_smoke.py)"
            + (" --postgres" if backend == "postgres" else ""),
            "database": backend,
            "schema_revision": rev,
        }
        write_report(report_file, report)
        out.append(f"  wrote {report_file.relative_to(ROOT).as_posix()}")
    finally:
        compose("down", "-v", env=env, check=False)
        out.append("  stopped; smoke volume removed")
    return out


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(report, indent=2) + "\n").encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--postgres", action="store_true", help="API on the compose test Postgres")
    args = parser.parse_args()
    backend = "postgres" if args.postgres else "sqlite"
    if args.postgres:
        PROFILE[:] = ["--profile", "postgres"]
    try:
        lines = run(backend)
    except SmokeError as exc:
        print(f"docker smoke FAILED: {exc}")
        sys.exit(1)
    print("docker smoke passed")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
