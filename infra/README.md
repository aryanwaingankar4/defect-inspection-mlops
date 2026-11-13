# Azure setup (ADR-039)

The API runs on Azure Container Apps, with its prediction log in Azure Database for PostgreSQL. Everything runs in **`eastasia`**, in the resource group `rg-steelsentry` (the group's metadata is in `uaenorth`).

Why eastasia: the subscription policy allows only a few regions. Container Apps is not in `indiasouthcentral`, and Postgres B1ms returned `SkuNotAvailable` in `uaenorth` in every zone. `make pg-probe` found the first region that offers B1ms (ADR-039).

No credential is stored in this folder or anywhere in the repo.

## What gets created (`infra/main.bicep`)

| Resource | Setting |
|---|---|
| Log Analytics workspace | 30-day retention; Container Apps logs go here |
| Container Registry | Basic, admin user off |
| User-assigned managed identity | `AcrPull` on the registry; the app pulls with it (no registry password) |
| Container Apps environment | consumption only; names end in `-eastasia` |
| Container App `steelsentry-api` | external HTTPS → 8000, min 0 / max 1 replicas, 0.5 vCPU / 1 GiB, probes on `/health`; secrets `API_KEY`, `AUDIT_SALT`, `DATABASE_URL` |
| Postgres Flexible Server | B1ms Burstable, 32 GB, PostgreSQL 16, TLS required, 7-day backups, no HA; database `defect` |
| Postgres firewall | "allow Azure services" only; the deployer's IP is added only while migrating |
| Budget (`infra/budget.bicep`) | $10/month on the resource group, alerts the Owner role at 50/80/100% |

## Steps

1. Install the Azure CLI, then run `az login`.
2. Register the providers: `Microsoft.App`, `Microsoft.OperationalInsights`, `Microsoft.DBforPostgreSQL`, `Microsoft.ContainerRegistry`, `Microsoft.ManagedIdentity`.
3. Fill in `.env` from `.env.example`: `API_KEY`, `AUDIT_SALT` and `AZURE_PG_ADMIN_PASSWORD`.
4. Optional: run `make pg-probe`. It creates only the Postgres server, trying regions in order, and records the region and name in `main.parameters.json`.
5. Run `make azure-up`. It creates the infrastructure and is safe to re-run.
6. Run `make serving-bundle`, then `make azure-deploy`. This:
   - builds the image locally;
   - pushes it to ACR;
   - migrates the Azure database (ADR-038);
   - deploys the app by image digest;
   - runs the cloud smoke and writes `results/cloud/deploy_report.json`.
7. Run `make azure-smoke` to re-check the running app.
8. Run `make pg-stop` / `make pg-start` to stop Postgres when idle. Azure restarts a stopped server after 7 days.
9. Run `make azure-down` to delete everything. You must type the resource group name to confirm.

## Trade-offs (ADR-039)

- **No VNet.** Postgres has a public endpoint behind a firewall that allows Azure services, plus TLS and a password. That rule admits traffic from any Azure tenant's resources. Private networking would fix this, but costs more and takes longer.
- **Admin login.** The API connects as the server admin. A least-privilege app role is a follow-up.
- **No read-only root filesystem.** Container Apps has no such option, unlike `docker-compose.yml`.
- **Cold start.** Scale to zero means the first request after idle waits for the replica to start.
- **Manual deploys.** Deploys run from the deployer's machine for now. A GitHub Actions deploy with OIDC is TASK-041.
- **Cost.** About $30.75/month in eastasia with Postgres running all the time, or about $9.9 with `make pg-stop` (retail prices, 2026-11-13).
