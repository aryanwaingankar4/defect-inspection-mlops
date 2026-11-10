"""TASK-032: Prometheus + Grafana config. Fast static checks; the live check is
`make monitoring-smoke` (manual, needs Docker)."""

import json
import re
from pathlib import Path

import pytest
import yaml

pytest.importorskip("prometheus_client")
pytest.importorskip("fastapi")

from src.serving.app import Metrics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MON = REPO_ROOT / "monitoring"
COMPOSE = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
SERVICES = COMPOSE["services"]
DASHBOARD = json.loads((MON / "grafana" / "dashboards" / "defect-api.json").read_text("utf-8"))
SUFFIXES = ("_bucket", "_sum", "_count", "_created")


def exposed_metric_names() -> set[str]:
    """Every sample name the app's registry can expose (counters end in _total)."""
    names = set()
    for family in Metrics().registry.collect():
        base = family.name
        names |= {base, f"{base}_total", *(base + s for s in SUFFIXES)}
    return names


def queries() -> list[str]:
    return [t["expr"] for p in DASHBOARD["panels"] for t in p["targets"]]


def test_images_are_pinned_and_ports_local():
    for name in ("prometheus", "grafana"):
        image = SERVICES[name]["image"]
        tag = image.split(":")[-1]
        assert ":" in image and tag not in ("latest", "main") and re.search(r"\d+\.\d+\.\d+", tag)
        assert all(p.startswith("127.0.0.1:") for p in SERVICES[name]["ports"]), name
        assert "no-new-privileges:true" in SERVICES[name]["security_opt"]
        assert "env_file" not in SERVICES[name]


def test_grafana_needs_a_password_and_refuses_anonymous_users():
    env = SERVICES["grafana"]["environment"]
    assert env["GF_SECURITY_ADMIN_PASSWORD"].startswith("${GRAFANA_ADMIN_PASSWORD:?")
    assert env["GF_AUTH_ANONYMOUS_ENABLED"] == "false"
    assert env["GF_USERS_ALLOW_SIGN_UP"] == "false"
    secrets_in_env = {k for k, v in env.items() if "API_KEY" in str(v) or "MLFLOW" in str(v)}
    assert not secrets_in_env


def test_prometheus_scrapes_the_api():
    cfg = yaml.safe_load((MON / "prometheus" / "prometheus.yml").read_text(encoding="utf-8"))
    jobs = {j["job_name"]: j for j in cfg["scrape_configs"]}
    assert jobs["api"]["metrics_path"] == "/metrics"
    assert jobs["api"]["static_configs"][0]["targets"] == ["api:8000"]


def test_grafana_provisioning_matches_the_dashboard():
    ds = yaml.safe_load(
        (MON / "grafana" / "provisioning" / "datasources" / "prometheus.yml").read_text("utf-8")
    )["datasources"][0]
    assert ds["uid"] == "prometheus" and ds["url"] == "http://prometheus:9090"
    prov = yaml.safe_load(
        (MON / "grafana" / "provisioning" / "dashboards" / "dashboards.yml").read_text("utf-8")
    )["providers"][0]
    mount = "./monitoring/grafana/dashboards:" + prov["options"]["path"] + ":ro"
    assert mount in SERVICES["grafana"]["volumes"]
    for panel in DASHBOARD["panels"]:
        assert panel["datasource"]["uid"] == ds["uid"], panel["title"]


def test_dashboard_has_the_required_panels():
    titles = " | ".join(p["title"].lower() for p in DASHBOARD["panels"])
    for needed in ("requests", "latency", "confidence", "routing reasons", "queue"):
        assert needed in titles, needed
    assert DASHBOARD["uid"] == "defect-api"


def test_every_dashboard_query_uses_metrics_the_app_exposes():
    exposed = exposed_metric_names()
    for expr in queries():
        used = set(re.findall(r"\b(defect_api_[a-z_]+)", expr))
        assert used, expr
        assert used <= exposed, (expr, used - exposed)


def test_metric_names_are_stable():
    """Renaming a metric breaks dashboards silently; change both on purpose."""
    names = {f.name for f in Metrics().registry.collect()}
    assert names == {
        "defect_api_requests",
        "defect_api_request_latency_seconds",
        "defect_api_inference_latency_seconds",
        "defect_api_rejections",
        "defect_api_predictions",
        "defect_api_routed_to_review",
        "defect_api_routed_reason",
        "defect_api_calibrated_confidence",
        "defect_api_energy",
        "defect_api_review_queue_size",
        "defect_api_db_write_failures",
        "defect_api_model_info",
    }
