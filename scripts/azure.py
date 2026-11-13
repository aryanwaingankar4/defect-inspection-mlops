"""Azure deployment of the API (ADR-039; TASK-036/038/039). Run through make:

    make azure-up       # resource group + infra/main.bicep (no app) + budget
    make azure-deploy   # build + push image, migrate Azure Postgres, deploy the app, smoke
    make azure-smoke    # smoke the running app again -> results/cloud/deploy_report.json
    make pg-stop / pg-start
    make azure-down     # delete the resource group (asks you to type its name)

Secrets (API_KEY, AUDIT_SALT, AZURE_PG_ADMIN_PASSWORD) come only from the environment, which
make loads from .env. They go to Azure through a temporary parameters file that is deleted
after each deployment. They are never printed, never passed on a command line and never
written to results/. Only the committed synthetic golden image is sent to the cloud (ADR-017).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "infra"
GOLDEN = ROOT / "tests" / "golden"
REPORT = ROOT / "results" / "cloud" / "deploy_report.json"
RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP", "rg-steelsentry")
LOCATION = os.environ.get("AZURE_LOCATION", "uaenorth")
DEPLOYMENT = "steelsentry-main"
REPOSITORY = "defect-inspection-api"
LOCAL_IMAGE = "defect-inspection-api:local"
FIREWALL_RULE = "deployer-temp-migrations"
LATENCY_REQUESTS = 20
SECRET_ENVS = ("API_KEY", "AUDIT_SALT", "AZURE_PG_ADMIN_PASSWORD")
AZ_WINDOWS = Path(r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd")


class DeployError(RuntimeError):
    pass


# ---------------------------------------------------------------- pure helpers (tested)


def database_url(user: str, password: str, host: str, database: str) -> str:
    """Postgres URL for the API, TLS required. User and password are percent-encoded."""
    return (
        f"postgresql+psycopg://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:5432/{quote(database, safe='')}?sslmode=require"
    )


def golden_matches(pred: dict, want: dict, tol: float = 1e-4) -> bool:
    """Same prediction, confidence within tol, same reasons apart from the random audit."""
    reasons = [r for r in pred.get("reasons", []) if r != "audit"]
    return (
        pred.get("prediction") == want["prediction"]
        and reasons == want["reasons"]
        and abs(float(pred.get("calibrated_confidence", -1.0)) - want["calibrated_confidence"])
        < tol
    )


def latency_summary(ms: list[float]) -> dict[str, float | int]:
    return {
        "n": len(ms),
        "p50_ms": round(statistics.median(ms), 1),
        "min_ms": round(min(ms), 1),
        "max_ms": round(max(ms), 1),
    }


def assert_no_secrets(text: str, secrets: list[str]) -> None:
    """Refuse to write a report that contains any secret value."""
    for s in secrets:
        if s and s in text:
            raise DeployError("refusing to write the report: it would contain a secret value")


# ---------------------------------------------------------------- az / docker wrappers


def az_exe() -> str:
    found = shutil.which("az")
    if found:
        return found
    if AZ_WINDOWS.exists():
        return str(AZ_WINDOWS)
    raise DeployError("Azure CLI not found (install it and run `az login`)")


def run(cmd: list[str], *, env: dict[str, str] | None = None, capture: bool = True) -> str:
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=capture, text=True)
    if out.returncode != 0:
        tail = (out.stderr or "")[-1500:] if capture else ""
        raise DeployError(f"`{' '.join(cmd[:4])} ...` failed (exit {out.returncode})\n{tail}")
    return out.stdout if capture else ""


def az(*args: str) -> str:
    return run([az_exe(), *args, "--only-show-errors"]).strip()


def az_json(*args: str):
    return json.loads(az(*args, "-o", "json") or "null")


def need_env(*names: str) -> dict[str, str]:
    missing = [n for n in names if not os.environ.get(n, "").strip()]
    if missing:
        raise DeployError(f"missing in the environment (.env): {', '.join(missing)}")
    return {n: os.environ[n] for n in names}


def deploy_bicep(params: dict[str, object]) -> dict:
    """Deploy infra/main.bicep with non-secret params from the parameters file and the
    given params (secrets included) from a temporary file that is always deleted."""
    deploy_template(INFRA / "main.bicep", DEPLOYMENT, params, base=INFRA / "main.parameters.json")
    return outputs()


def deploy_template(
    template: Path, name: str, params: dict[str, object], base: Path | None = None
) -> None:
    with tempfile.TemporaryDirectory(prefix="steelsentry-") as tmp:
        pfile = Path(tmp) / "secure.parameters.json"
        pfile.write_text(
            json.dumps(
                {
                    "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
                    "contentVersion": "1.0.0.0",
                    "parameters": {k: {"value": v} for k, v in params.items()},
                }
            ),
            encoding="utf-8",
        )
        base_args = ["-p", f"@{base}"] if base else []
        az(
            "deployment", "group", "create",
            "-g", RESOURCE_GROUP,
            "-n", name,
            "-f", str(template),
            *base_args,
            "-p", f"@{pfile}",
            "-o", "none",
        )  # fmt: skip


def outputs() -> dict[str, str]:
    raw = az_json(
        "deployment", "group", "show", "-g", RESOURCE_GROUP, "-n", DEPLOYMENT,
        "--query", "properties.outputs",
    )  # fmt: skip
    if not raw:
        raise DeployError("no infrastructure deployment found: run `make azure-up` first")
    return {k: v["value"] for k, v in raw.items()}


def client_ip() -> str:
    ip = os.environ.get("AZURE_CLIENT_IP", "").strip()
    if ip:
        return ip
    with urllib.request.urlopen("https://api.ipify.org", timeout=15) as r:
        return r.read().decode().strip()


# ---------------------------------------------------------------- commands


def cmd_up() -> None:
    env = need_env("AZURE_PG_ADMIN_PASSWORD")
    print(f"resource group {RESOURCE_GROUP} in {LOCATION}")
    az("group", "create", "-n", RESOURCE_GROUP, "-l", LOCATION, "-o", "none")
    print("deploying infra/main.bicep (infrastructure only; Postgres takes ~5-10 min) ...")
    out = deploy_bicep({"pgAdminPassword": env["AZURE_PG_ADMIN_PASSWORD"], "deployApp": False})
    for k in ("acrLoginServer", "pgServerName", "envName"):
        print(f"  {k}: {out[k]}")
    start = datetime.now(UTC).strftime("%Y-%m-01")
    try:
        az(
            "deployment", "group", "create", "-g", RESOURCE_GROUP, "-n", "steelsentry-budget",
            "-f", str(INFRA / "budget.bicep"), "-p", f"startDate={start}", "-o", "none",
        )  # fmt: skip
        print("  budget: $10/month on the resource group (alerts the Owner role at 50/80/100%)")
    except DeployError as exc:
        print(f"  WARNING: budget not created (the offer may not support budgets): {exc}")


def push_image(out: dict[str, str]) -> tuple[str, str]:
    """Build the API image locally, push it to ACR, return (image@digest, git commit)."""
    commit = run(["git", "rev-parse", "--short", "HEAD"]).strip()
    print(f"building the image at {commit} (make docker-build) ...")
    run([shutil.which("make") or "make", "docker-build"], capture=False)
    target = f"{out['acrLoginServer']}/{REPOSITORY}:{commit}"
    run(["docker", "tag", LOCAL_IMAGE, target])
    az("acr", "login", "-n", out["acrName"])
    print(f"pushing {target} ...")
    run(["docker", "push", target])
    digests = json.loads(run(["docker", "inspect", "--format", "{{json .RepoDigests}}", target]))
    ref = next((d for d in digests if d.startswith(out["acrLoginServer"] + "/")), None)
    if not ref:
        raise DeployError("pushed image has no registry digest")
    return ref, commit


def migrate(out: dict[str, str], password: str) -> None:
    """Upgrade the Azure database to head through a temporary firewall rule for this machine."""
    url = database_url(_param("pgAdminLogin"), password, out["pgHost"], _param("pgDatabaseName"))
    ip = client_ip()
    print("adding a temporary Postgres firewall rule for this machine ...")
    az(
        "postgres", "flexible-server", "firewall-rule", "create", "-g", RESOURCE_GROUP,
        "-s", out["pgServerName"], "-n", FIREWALL_RULE,
        "--start-ip-address", ip, "--end-ip-address", ip, "-o", "none",
    )  # fmt: skip
    try:
        env = {**os.environ, "DATABASE_URL": url}
        print(
            run(
                ["uv", "run", "--locked", "--group", "serve", "python", "-m", "src.db.migrate",
                 "upgrade"],
                env=env,
            ).strip()
        )  # fmt: skip
    finally:
        az(
            "postgres", "flexible-server", "firewall-rule", "delete", "-g", RESOURCE_GROUP,
            "-s", out["pgServerName"], "-n", FIREWALL_RULE, "--yes",
        )  # fmt: skip
        print("removed the temporary firewall rule")


def _param(name: str) -> str:
    data = json.loads((INFRA / "main.parameters.json").read_text("utf-8"))
    return data["parameters"][name]["value"]


def cmd_deploy() -> None:
    env = need_env(*SECRET_ENVS)
    out = outputs()
    image, commit = push_image(out)
    migrate(out, env["AZURE_PG_ADMIN_PASSWORD"])
    print("deploying the Container App ...")
    deploy_bicep(
        {
            "pgAdminPassword": env["AZURE_PG_ADMIN_PASSWORD"],
            "deployApp": True,
            "image": image,
            "apiKey": env["API_KEY"],
            "auditSalt": env["AUDIT_SALT"],
            "databaseUrl": database_url(
                _param("pgAdminLogin"),
                env["AZURE_PG_ADMIN_PASSWORD"],
                out["pgHost"],
                _param("pgDatabaseName"),
            ),
        }
    )
    cmd_smoke(git_commit=commit)


def http(url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except json.JSONDecodeError:
            return e.code, {}


def multipart(png: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="g.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode()
    return (
        head + png + f"\r\n--{boundary}--\r\n".encode(),
        f"multipart/form-data; boundary={boundary}",
    )


def cmd_smoke(git_commit: str | None = None) -> None:
    env = need_env("API_KEY")
    out = outputs()
    app = az_json("containerapp", "show", "-g", RESOURCE_GROUP, "-n", out["appName"])
    fqdn = app["properties"]["configuration"]["ingress"]["fqdn"]
    base = f"https://{fqdn}"
    image = app["properties"]["template"]["containers"][0]["image"]
    revision = app["properties"]["latestRevisionName"]
    checks: dict[str, bool] = {}

    t0 = time.perf_counter()
    status, health = 0, {}
    while time.perf_counter() - t0 < 300:
        status, health = http(base + "/health")
        if status == 200 and health.get("status") == "ok":
            break
        time.sleep(5)
    first_health_s = round(time.perf_counter() - t0, 1)
    checks["health_200_ok"] = status == 200 and health.get("status") == "ok"

    png = (GOLDEN / "synthetic_golden.png").read_bytes()
    want = next(
        e
        for e in json.loads((GOLDEN / "expected.json").read_text("utf-8"))["champion"]["images"]
        if e["id"] == "synthetic"
    )
    body, ctype = multipart(png)
    keyed = {"Content-Type": ctype, "X-API-Key": env["API_KEY"]}
    status, pred = http(base + "/predict", data=body, headers=keyed)
    checks["predict_golden_matches_expected"] = status == 200 and golden_matches(pred, want)
    status, err = http(base + "/predict", data=body, headers={"Content-Type": ctype})
    checks["predict_without_key_401"] = status == 401

    ms: list[float] = []
    for _ in range(LATENCY_REQUESTS):
        t = time.perf_counter()
        s, _ = http(base + "/predict", data=body, headers=keyed)
        ms.append((time.perf_counter() - t) * 1000)
        if s != 200:
            checks["latency_requests_all_200"] = False
    checks.setdefault("latency_requests_all_200", True)

    report = {
        "decision_record": "ADR-039",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "url": base,
        "region": app.get("location"),
        "postgres_region": out.get("pgLocation"),
        "resource_group_metadata_region": LOCATION,
        "resource_group": RESOURCE_GROUP,
        "container_app": out["appName"],
        "revision": revision,
        "image": image,
        "image_digest": image.split("@", 1)[1] if "@" in image else None,
        "git_commit": git_commit,
        "model": health.get("model"),
        "database": health.get("database"),
        "smoke": {
            "passed": all(checks.values()),
            "checks": checks,
            "golden_prediction": pred.get("prediction"),
            "golden_calibrated_confidence": pred.get("calibrated_confidence"),
            "golden_reasons": pred.get("reasons"),
            "expected": {k: want[k] for k in ("prediction", "calibrated_confidence", "reasons")},
        },
        "first_health_seconds": first_health_s,
        "first_health_note": "cold start only if the app had scaled to zero; right after a "
        "deploy the new replica is usually already running",
        "predict_latency_client_side": {
            **latency_summary(ms),
            "note": "sequential, warm replica, measured from the deployer's machine "
            f"(includes the internet round trip to {app.get('location')}); "
            "not a load test (TASK-043)",
        },
        "checked_by": "make azure-smoke (scripts/azure.py)",
    }
    text = json.dumps(report, indent=2) + "\n"
    assert_no_secrets(text, [os.environ.get(n, "") for n in SECRET_ENVS])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_bytes(text.encode())
    for k, v in checks.items():
        print(f"  [{'ok' if v else 'FAIL'}] {k}")
    print(f"  url {base}  revision {revision}")
    print(f"  p50 /predict {report['predict_latency_client_side']['p50_ms']} ms (client side)")
    print(f"  wrote {REPORT.relative_to(ROOT).as_posix()}")
    if not report["smoke"]["passed"]:
        raise DeployError("cloud smoke failed")


def cmd_pg(action: str) -> None:
    out = outputs()
    print(f"postgres {action} {out['pgServerName']} ...")
    az("postgres", "flexible-server", action, "-g", RESOURCE_GROUP, "-n", out["pgServerName"])
    print("done" + (" (Azure restarts a stopped server after 7 days)" if action == "stop" else ""))


def cmd_down() -> None:
    print(f"This DELETES resource group {RESOURCE_GROUP} and everything in it (database too).")
    typed = input(f"Type {RESOURCE_GROUP} to confirm: ").strip()
    if typed != RESOURCE_GROUP:
        print("not confirmed; nothing deleted")
        return
    az("group", "delete", "-n", RESOURCE_GROUP, "--yes", "--no-wait")
    print("deletion started (runs in Azure for a few minutes)")


PROBE_REGIONS = ("eastasia", "indiasouthcentral", "malaysiawest", "indonesiacentral")


def set_parameters(**values: str) -> None:
    """Write non-secret values into infra/main.parameters.json (kept in git)."""
    path = INFRA / "main.parameters.json"
    data = json.loads(path.read_text("utf-8"))
    for k, v in values.items():
        data["parameters"][k] = {"value": v}
    path.write_bytes((json.dumps(data, indent=2) + "\n").encode())


def cmd_pg_probe() -> None:
    """Create ONLY the Postgres server (B1ms, no zone, random name) region by region and keep
    the first that works; failed servers are deleted. Records region + name in the
    parameters file so main.bicep adopts that server (ADR-039)."""
    env = need_env("AZURE_PG_ADMIN_PASSWORD")
    regions = os.environ.get("AZURE_PG_PROBE_REGIONS", ",".join(PROBE_REGIONS)).split(",")
    for region in regions:
        name = f"steelsentry-pg-{uuid.uuid4().hex[:8]}"
        print(f"probe {region}: creating {name} (B1ms, no zone) ...", flush=True)
        t0 = time.perf_counter()
        try:
            deploy_template(
                INFRA / "postgres.bicep",
                f"steelsentry-pg-probe-{region}",
                {
                    "location": region,
                    "serverName": name,
                    "adminLogin": _param("pgAdminLogin"),
                    "adminPassword": env["AZURE_PG_ADMIN_PASSWORD"],
                    "databaseName": _param("pgDatabaseName"),
                },
            )
        except DeployError as exc:
            code = next((c for c in ("SkuNotAvailable", "RequestDisallowedByPolicy",
                                     "LocationIsOfferRestricted", "ServerNameAlreadyExists")
                         if c in str(exc)), "error")  # fmt: skip
            print(f"  failed after {time.perf_counter() - t0:.0f} s: {code}", flush=True)
            try:
                az("postgres", "flexible-server", "delete", "-g", RESOURCE_GROUP, "-n", name,
                   "--yes")  # fmt: skip
                print(f"  deleted {name}")
            except DeployError:
                print(f"  {name} not present (nothing to delete)")
            continue
        print(f"  OK in {time.perf_counter() - t0:.0f} s: {name} in {region}")
        set_parameters(pgLocation=region, pgServerName=name)
        print("  recorded pgLocation / pgServerName in infra/main.parameters.json")
        return
    raise DeployError(f"Postgres B1ms failed in every probed region: {', '.join(regions)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = ["up", "deploy", "smoke", "pg-probe", "pg-stop", "pg-start", "down"]
    parser.add_argument("command", choices=commands)
    args = parser.parse_args()
    try:
        {
            "up": cmd_up,
            "deploy": cmd_deploy,
            "smoke": cmd_smoke,
            "pg-probe": cmd_pg_probe,
            "pg-stop": lambda: cmd_pg("stop"),
            "pg-start": lambda: cmd_pg("start"),
            "down": cmd_down,
        }[args.command]()
    except DeployError as exc:
        print(f"azure {args.command} FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
