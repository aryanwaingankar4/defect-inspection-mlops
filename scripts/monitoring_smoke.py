"""`make monitoring-smoke` (TASK-032): start api + Prometheus + Grafana with docker compose and
check the monitoring path end to end, then remove everything.

Separate compose project, ports, throwaway API key and Grafana password, own volumes. Sends only
the committed synthetic golden image (ADR-017), a few generated non-steel images
(src/data/synthetic_ood.py) and one junk upload. Checks: Prometheus scrapes the API (target up);
every labelled series exists at 0 before any prediction and the routed-share panel shows no data
(never infinity); after N predictions the dashboard's own panel queries give exactly the counts
from the responses (by class summing to N, routed share = routed / N); request, rejection and
review-queue metrics; every dashboard query runs; Grafana is healthy, the dashboard is
provisioned, its datasource answers, and anonymous access is refused.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # for src.data.synthetic_ood (numpy + Pillow only)

from src.data.synthetic_ood import generate  # noqa: E402

PROJECT = "defect-inspection-monitoring-smoke"
PORTS = {"API_PORT": "18001", "PROMETHEUS_PORT": "19090", "GRAFANA_PORT": "13000"}
API = f"http://127.0.0.1:{PORTS['API_PORT']}"
PROM = f"http://127.0.0.1:{PORTS['PROMETHEUS_PORT']}"
GRAFANA = f"http://127.0.0.1:{PORTS['GRAFANA_PORT']}"
DASHBOARD = ROOT / "monitoring" / "grafana" / "dashboards" / "defect-api.json"
GOLDEN = ROOT / "tests" / "golden" / "synthetic_golden.png"
N_GOLDEN = 3  # the golden image is routed (low_confidence)
SYNTHETIC = ["gradient", "stripes", "checkerboard", "shapes", "blobs"]  # mixed routing
N_CLASSES = 6


class SmokeError(RuntimeError):
    pass


def compose(*args: str, env: dict[str, str], check: bool = True) -> str:
    out = subprocess.run(
        ["docker", "compose", "-p", PROJECT, *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if check and out.returncode != 0:
        raise SmokeError(f"`docker compose {' '.join(args)}` failed:\n{out.stderr[-2000:]}")
    return out.stdout + out.stderr


def http(url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except ValueError:
            return e.code, None
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return 0, None


def wait_for(label: str, fn, timeout: float = 120.0):
    start = time.monotonic()
    last = None
    while time.monotonic() - start < timeout:
        ok, last = fn()
        if ok:
            return last, time.monotonic() - start
        time.sleep(2)
    raise SmokeError(f"timed out waiting for {label} (last: {last})")


def prom_query(expr: str):
    url = f"{PROM}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    return http(url)


def prom_value(expr: str) -> float | None:
    status, body = prom_query(expr)
    if status != 200 or body["status"] != "success" or not body["data"]["result"]:
        return None
    return float(body["data"]["result"][0]["value"][1])


def predict_body(png: bytes, key: str | None) -> tuple[int, dict | None]:
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="g.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode()
        + png
        + f"\r\n--{boundary}--\r\n".encode()
    )
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if key:
        headers["X-API-Key"] = key
    return http(f"{API}/predict", data=body, headers=headers)


def predict(png: bytes, key: str | None) -> int:
    return predict_body(png, key)[0]


def check(ok: bool, label: str, out: list[str]) -> None:
    out.append(f"  [{'ok' if ok else 'FAIL'}] {label}")
    if not ok:
        print("\n".join(out))
        raise SmokeError(label)


def panel_expr(title_start: str) -> str:
    d = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    return next(p["targets"][0]["expr"] for p in d["panels"] if p["title"].startswith(title_start))


def dashboard_queries() -> list[tuple[str, str]]:
    d = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    return [(p["title"], t["expr"]) for p in d["panels"] for t in p["targets"]]


def run() -> list[str]:
    key, password = secrets.token_hex(16), secrets.token_hex(16)  # throwaway: never printed
    env = {**os.environ, **PORTS, "API_KEY": key, "GRAFANA_ADMIN_PASSWORD": password}
    auth = {"Authorization": "Basic " + base64.b64encode(f"admin:{password}".encode()).decode()}
    out: list[str] = []
    try:
        compose("up", "-d", "--no-build", env=env)
        _, s = wait_for("API health", lambda: (http(f"{API}/health")[0] == 200, None))
        out.append(f"  api healthy after {s:.1f} s")

        def first_scrape():
            n_series = prom_value("count(defect_api_predictions_total)")
            return prom_value('up{job="api"}') == 1 and n_series == N_CLASSES, n_series

        n_series, s = wait_for("the first scrape", first_scrape)
        out.append(f"  first scrape after {s:.1f} s")
        check(
            n_series == N_CLASSES,
            f"all {N_CLASSES} class series exist at 0 before any request",
            out,
        )
        status, share0 = prom_query(panel_expr("Routed share"))
        empty = status == 200 and share0["data"]["result"] == []
        check(empty, "routed share before any prediction: no data (not infinite)", out)

        images = [GOLDEN.read_bytes()] * N_GOLDEN + [png for _, png in generate(SYNTHETIC, 1, 23)]
        bodies = []
        for img in images:
            status, body = predict_body(img, key)
            check(status == 200, f"prediction {len(bodies) + 1} -> 200", out)
            bodies.append(body)
        out[-len(images) :] = [
            f"  [ok] {len(images)} predictions sent (golden x{N_GOLDEN} + "
            f"{len(SYNTHETIC)} synthetic)"
        ]
        codes = [predict(b"not an image", key), predict(images[0], None)]
        check(codes == [422, 401], f"bad requests: {codes}", out)
        n = len(bodies)
        by_class = Counter(b["prediction"] for b in bodies)
        n_routed = sum(b["routed_to_review"] for b in bodies)
        by_reason = Counter(r for b in bodies for r in b["reasons"])

        def scraped():
            up = prom_value('up{job="api"}')
            queue = prom_value("defect_api_review_queue_size")
            ok_200 = prom_value('sum(defect_api_requests_total{endpoint="/predict",status="200"})')
            return (up == 1 and queue == n_routed and ok_200 == n), (up, queue, ok_200)

        (up, queue, ok_200), s = wait_for("Prometheus to scrape the new values", scraped)
        out.append(f"  Prometheus scraped the new values after {s:.1f} s")
        check(up == 1, "Prometheus target api is up", out)
        check(ok_200 == n, f"defect_api_requests_total /predict 200 = {ok_200:.0f}", out)
        check(queue == n_routed, f"defect_api_review_queue_size = {queue:.0f} (= routed)", out)

        status, res = prom_query(panel_expr("Predictions by class"))
        panel = {r["metric"]["prediction"]: float(r["value"][1]) for r in res["data"]["result"]}
        same = all(panel.get(c, 0.0) == by_class.get(c, 0) for c in set(panel) | set(by_class))
        check(
            len(panel) == N_CLASSES and same and sum(panel.values()) == n,
            f"panel 'Predictions by class' = responses {dict(by_class)}, sums to N = {n}",
            out,
        )
        share = prom_value(panel_expr("Routed share"))
        exact = share is not None and abs(share - n_routed / n) < 1e-12
        check(exact, f"panel 'Routed share' = {share} = routed/N = {n_routed}/{n}", out)
        for reason in ("low_confidence", "ood", "audit"):
            got = prom_value(f'defect_api_routed_reason_total{{reason="{reason}"}}')
            check(got == by_reason.get(reason, 0), f"routing reason {reason} = {got:.0f}", out)
        rej = prom_value('defect_api_rejections_total{reason="undecodable"}')
        no_key = prom_value('defect_api_rejections_total{reason="missing_api_key"}')
        check(rej == 1 and no_key == 1, "rejections counted: undecodable 1, missing_api_key 1", out)

        bad = []
        for title, expr in dashboard_queries():
            status, body = prom_query(expr)
            if status != 200 or body.get("status") != "success":
                bad.append(title)
        n_queries = len(dashboard_queries())
        check(not bad, f"all {n_queries} dashboard queries run in Prometheus {bad or ''}", out)

        _, s = wait_for(
            "Grafana health",
            lambda: (http(f"{GRAFANA}/api/health")[0] == 200, None),
        )
        status, health = http(f"{GRAFANA}/api/health")
        check(
            health.get("database") == "ok",
            f"Grafana healthy (version {health.get('version')})",
            out,
        )
        status, dash = http(f"{GRAFANA}/api/dashboards/uid/defect-api", headers=auth)
        title = (dash or {}).get("dashboard", {}).get("title")
        n_panels = len((dash or {}).get("dashboard", {}).get("panels", []))
        check(
            status == 200 and n_panels > 0,
            f"dashboard provisioned: {title!r}, {n_panels} panels",
            out,
        )
        status, ds = http(f"{GRAFANA}/api/datasources/uid/prometheus/health", headers=auth)
        check(
            status == 200 and (ds or {}).get("status") == "OK",
            "Grafana -> Prometheus datasource OK",
            out,
        )
        status, _ = http(f"{GRAFANA}/api/dashboards/uid/defect-api")
        check(status == 401, "anonymous access refused (401)", out)

        logs = compose("logs", env=env)
        check(
            key not in logs and password not in logs,
            "API key and Grafana password not in logs",
            out,
        )
    finally:
        compose("down", "-v", env=env, check=False)
        out.append("  stopped; smoke volumes removed")
    return out


def main() -> None:
    try:
        lines = run()
    except SmokeError as exc:
        print(f"monitoring smoke FAILED: {exc}")
        sys.exit(1)
    print("monitoring smoke passed")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
