"""`make load-test` (TASK-043): warm up the Azure API, then a headless Locust run against it.

Only the committed synthetic golden image is sent (scripts/locustfile.py). Nothing is tuned:
the app runs as deployed (ADR-039: max 1 replica, 0.5 vCPU / 1 GiB). Writes
results/load/locust_report.json with RPS, p50/p95/p99 and the error rate. The API key comes
from the environment and is never written.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "load" / "locust_report.json"
DEPLOY = ROOT / "results" / "cloud" / "deploy_report.json"


def warm_up(url: str, key: str, predicts: int = 5, timeout: float = 300.0) -> float:
    """Wait for /health (scale from zero), then a few /predict calls; returns seconds taken."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + "/health", timeout=60) as r:
                if r.status == 200:
                    break
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(5)
    else:
        raise SystemExit("API not healthy within the warm-up timeout")
    png = (ROOT / "tests" / "golden" / "synthetic_golden.png").read_bytes()
    boundary = "loadtestboundary"
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="g.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode()
        + png
        + f"\r\n--{boundary}--\r\n".encode()
    )
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}", "X-API-Key": key}
    for _ in range(predicts):
        req = urllib.request.Request(url + "/predict", data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status != 200:
                raise SystemExit(f"warm-up /predict returned {r.status}")
    return round(time.perf_counter() - t0, 1)


def parse_stats(stats_csv: Path) -> dict:
    rows = {r["Name"]: r for r in csv.DictReader(stats_csv.open(encoding="utf-8"))}
    agg = rows["Aggregated"]
    n, fails = int(agg["Request Count"]), int(agg["Failure Count"])
    return {
        "requests": n,
        "failures": fails,
        "error_rate": round(fails / n, 4) if n else None,
        "rps": round(float(agg["Requests/s"]), 2),
        "p50_ms": float(agg["50%"]),
        "p95_ms": float(agg["95%"]),
        "p99_ms": float(agg["99%"]),
        "mean_ms": round(float(agg["Average Response Time"]), 1),
        "max_ms": float(agg["Max Response Time"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--spawn-rate", type=float, default=5)
    parser.add_argument("--run-time", default="2m")
    args = parser.parse_args()
    key = os.environ.get("API_KEY", "").strip()
    if not key:
        raise SystemExit("API_KEY is not set (.env)")
    url = json.loads(DEPLOY.read_text("utf-8"))["url"]

    print(f"warming up {url} ...", flush=True)
    warm_s = warm_up(url, key)
    with tempfile.TemporaryDirectory(prefix="locust-") as tmp:
        prefix = Path(tmp) / "run"
        cmd = [
            sys.executable, "-m", "locust", "-f", str(ROOT / "scripts" / "locustfile.py"),
            "--headless", "--host", url, "-u", str(args.users), "-r", str(args.spawn_rate),
            "-t", args.run_time, "--csv", str(prefix), "--only-summary", "--loglevel", "WARNING",
        ]  # fmt: skip
        print(f"locust: {args.users} users, spawn {args.spawn_rate}/s, {args.run_time} ...")
        done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        stats = parse_stats(Path(f"{prefix}_stats.csv"))
    report = {
        "task": "TASK-043",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "url": url,
        "endpoint": "POST /predict (synthetic golden image only)",
        "users": args.users,
        "spawn_rate_per_s": args.spawn_rate,
        "run_time": args.run_time,
        "wait_time_s": [0.5, 1.5],
        "warm_up_seconds": warm_s,
        "locust_exit_code": done.returncode,
        **stats,
        "deployment": "as deployed, nothing tuned: Azure Container Apps eastasia, max 1 replica, "
        "0.5 vCPU / 1 GiB, FP32 ONNX, every request logged to Azure Postgres (ADR-039)",
        "client": "the deployer's laptop over the public internet "
        "(latency includes the round trip)",
        "tool": "locust 2.47.0 (scripts/locustfile.py, scripts/load_test.py)",
    }
    text = json.dumps(report, indent=2) + "\n"
    if key in text:
        raise SystemExit("refusing to write a report that contains the API key")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_bytes(text.encode())
    print({k: report[k] for k in ("requests", "rps", "p50_ms", "p95_ms", "p99_ms", "error_rate")})
    print(f"wrote {REPORT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
