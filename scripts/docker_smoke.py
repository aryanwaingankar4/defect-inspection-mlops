"""`make docker-smoke` (TASK-021): start the built API image with docker compose and check it.

Uses a separate compose project, a random throwaway API key and its own volume, and removes
them at the end. Sends only the committed synthetic golden image (ADR-017): no NEU file goes
into or near the container. Checks:
health check healthy, /health, /predict (= tests/golden/expected.json), 401 without key,
non-root user, no torch/mlflow inside, read-only filesystem, one logged row that survives a
restart, the key not in the logs, image labels.
"""

from __future__ import annotations

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
BASE = f"http://127.0.0.1:{PORT}"
GOLDEN = ROOT / "tests" / "golden"
ROW_COUNT = (
    "from src.db.store import PredictionLog; "
    "print(PredictionLog('sqlite:////data/predictions.db').count())"
)


class SmokeError(RuntimeError):
    pass


def compose(*args: str, env: dict[str, str], check: bool = True) -> str:
    cmd = ["docker", "compose", "-p", PROJECT, *args]
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if check and out.returncode != 0:
        raise SmokeError(f"`docker compose {' '.join(args)}` failed:\n{out.stderr[-2000:]}")
    return out.stdout + out.stderr


def exec_in(env: dict[str, str], *cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", "-p", PROJECT, "exec", "-T", "api", *cmd],
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


def run() -> list[str]:
    key = secrets.token_hex(16)  # throwaway: never printed
    env = {**os.environ, "API_KEY": key, "API_PORT": PORT}
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
    try:
        compose("up", "-d", "--no-build", env=env)
        out.append(f"  healthy after {wait_healthy(env):.1f} s (Docker HEALTHCHECK)")
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
    finally:
        compose("down", "-v", env=env, check=False)
        out.append("  stopped; smoke volume removed")
    return out


def main() -> None:
    try:
        lines = run()
    except SmokeError as exc:
        print(f"docker smoke FAILED: {exc}")
        sys.exit(1)
    print("docker smoke passed")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
