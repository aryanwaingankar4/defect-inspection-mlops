# Self-Improving Steel Defect Inspection System

[![CI](https://github.com/aryanwaingankar4/defect-inspection-mlops/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/aryanwaingankar4/defect-inspection-mlops/actions/workflows/ci.yml)

A steel-surface defect classifier that improves itself in production — and proves each update is safe before it ships.

Low-confidence or unfamiliar images go to a human review queue. Corrected labels are versioned, a pipeline retrains the model, and a new model is promoted only if it is statistically better than the current one on a frozen test set. Everything is served, monitored and deployed with standard MLOps tooling.

> **Status: in development.** No results yet. Every number that appears in this README will come from a committed file in `results/`.

## Why this project

Factories cannot label every image up front. Inspection models have to keep learning from inspectors' corrections without silently getting worse. This project focuses on making that loop trustworthy:

- **Calibrated confidence**: temperature scaling, with ECE reported, so the review threshold means something.
- **Unknown-input handling**: an energy-based OOD score routes unfamiliar defects and non-steel images to review.
- **Unbiased feedback**: a random audit of auto-accepted predictions estimates the errors that would otherwise escape.
- **Honest promotion gate**: a paired McNemar test on a frozen test set, with per-class regression limits and a logged decision for every candidate, including rejections.

## Architecture (planned)

```
image → FastAPI (ONNX, INT8) → prediction + calibrated confidence + OOD score
          │
          ├─ log every request → SQLite locally / PostgreSQL on Azure (hash, prediction, confidence, OOD, latency, image features)
          │
          ├─ low confidence OR OOD OR random audit → Streamlit review queue
          │                                              │ corrected labels
          │                                              ▼
          │                                   data/feedback/ (DVC)
          │                                              │ N new labels
          │                                              ▼
          │                         Prefect retrain → evaluate on test_frozen → promotion gate
          │                                              │ pass
          │                                              ▼
          │                                MLflow registry: move @champion alias
          │
          └─ monitoring: Prometheus + Grafana locally (latency, confidence, queue size, audit error)
                         Azure Monitor in the cloud
                         Evidently drift reports on per-image features

push to main → GitHub Actions: CI → build image → GitHub Container Registry → Azure Container Apps (API + review UI)
```

## Deployment (planned)

- **Azure Container Apps:** the API and the review UI run as two apps that scale to zero.
- **Azure Database for PostgreSQL:** stores the prediction log and feedback in the cloud. Locally the same code uses SQLite; `DATABASE_URL` picks the database.
- **GitHub Actions:** every push to `main` runs CI, builds the image, pushes it to GitHub Container Registry and deploys to Azure. Azure credentials live only in GitHub Secrets.
- **Cost safety:** Azure for Students subscription, a budget alert, scale to zero and the smallest Postgres tier.
- A Hugging Face Space demo is optional.

## Tech stack

PyTorch + timm (EfficientNet-B0) · MLflow (aliases) · DVC · Prefect · ONNX Runtime (static INT8) · FastAPI · Streamlit · SQLite / PostgreSQL · Prometheus + Grafana · Azure Monitor · Evidently · Docker · GitHub Actions · GitHub Container Registry · Azure Container Apps

## Promotion gate (non-inferiority)

A retrained model (`@challenger`) replaces the serving model (`@champion`) only if it is **not worse**: on the `val` split, macro-F1 may drop by at most 0.01, no class recall may drop by more than 0.05, an exact McNemar test must not show it significantly worse, and its ONNX export must match PyTorch. The gate does **not** require a significant improvement. With 180 val images and a champion that makes few errors, no challenger could ever show one. `test_frozen` is never used by the loop; it stays for final, report-only results. Thresholds are in `configs/params.yaml`; every decision is written to `results/promotion/` and `results/promotion_log.json` (ADR-033).

## Data

- **NEU-CLS**: 6 hot-rolled steel defect classes, 1,800 grayscale 200×200 images. This is the main dataset.
- **X-SDD**: 7 hot-rolled strip defect classes. It is used as a second production line, giving both real domain shift and unseen defect types.

Licences and a full data card will be in `docs/DATA_CARD.md`.

## What is simulated

This is a portfolio project, not a factory deployment.

- **The reviewer is simulated.** Corrections use the dataset's true labels, with a configurable error rate (0%, 5%, 15%).
- **Production traffic is simulated** by streaming a held-out split through the API.
- **Lighting and camera drift** is partly synthetic (brightness, blur, noise, contrast). Cross-line drift comes from real data (X-SDD).

## Repository layout

See `docs/ARCHITECTURE.md`.

## Running it

Setup instructions will be added as each phase lands. The target is `make setup && make demo`.

### Local review loop (simulated)

`make serve` runs the API and writes the prediction log to `data/predictions.db` (`DATABASE_URL` in `.env`). `make stream`, `make review-ui`, `make simulate-review` and `make feedback-export` all work on that local log.

The Docker stack (`make docker-up`) keeps its own log in the Docker volume `predictions`. The review UI does **not** read it, so traffic sent to the Docker API never shows up in the review queue. One SQLite file cannot be shared safely between Windows and the Docker Desktop VM, so run the loop with `make serve`. In the cloud, both sides will share PostgreSQL through `DATABASE_URL`.

`make db-reset` moves the local log to `data/archive/` (git-ignored) and starts an empty one, for example after testing the UI by hand.

### Database schema (Alembic, ADR-038)

- **Migrations:** Alembic owns the schema (`src/db/migrations/`). `make db-migrate` upgrades `DATABASE_URL` to the latest revision.
- **The API never migrates.** It refuses to start if the database is not at the latest revision, and the error says to run `make db-migrate`. The review UI, the simulated reviewer and the feedback export do the same.
- **Docker:** in the compose stack, a one-shot `migrate` service runs before the API.
- **Logs from before Alembic:** `make db-migrate` checks a pre-Alembic local log against the baseline schema and stamps it only if they match exactly.
- **Local Postgres for tests:** `make pg-up` starts it, then `make test-db-postgres` runs the migrations and the DB tests against it. `make docker-smoke-postgres` runs the API smoke against it. It listens on 127.0.0.1 only, with a fixed password meant only for these tests. CI runs the same migrations and tests on a Postgres service container.

`make retrain` runs one loop cycle (ADR-034). It uploads the feedback export to a private Kaggle dataset, trains on Kaggle's GPU from the ImageNet weights on `train_v1` + feedback, registers the result as `@challenger`, and runs the promotion gate in record mode. It resumes if interrupted, and it never moves `@champion`. Each cycle writes `results/retrain/cycle_<run_uid>.json` and `results/promotion/gate_<version>.json`.

### Loop cycles (simulated, ADR-036)

[`results/loop/README.md`](results/loop/README.md) has one row per cycle: labels, reviewer error, val macro-F1 of champion vs challenger, verdict and gate file. It also has the report-only `test_frozen` comparison of the original v2 with the final champion. The table is generated from `results/loop/loop_summary.json` by `make loop-summary`.

- **Cycle 1 (pilot):** every `production_stream` image went through v2, and the images it routed were labelled.
- **Cycles 2 and 3:** each streams a disjoint chunk of the images the pilot did not label (`data/splits/loop_chunks.csv`, `make stream ARGS="--chunk cycleN"`). These are **not brand-new traffic**: v2 already predicted them in the pilot, but none of them was labelled or trained on. Cycle 3's simulated reviewer gives a wrong label 5% of the time.
- **Fault-injection run (labelled as such):** a broken reviewer moves every label to the next class. It works on a copy (`make fault-export`) that never enters the feedback history. The model is tagged `fault_injection=true`, and the gate runs in record mode only; `gate --apply` refuses such a version. In the actual run the gate was never reached: the review-threshold risk check (ADR-023) stopped the model first. The table records that outcome as it happened.
- **Calibration (ADR-037):** if a retrained model's temperature fit lands on its range bound (for example with zero val errors), it inherits the champion's T and is tagged `calibration=inherited_T`. This rule was added after cycle 2 exposed the gap.

## Documentation

| File | Purpose |
|---|---|
| `docs/PRD.md` | What is being built, and what is out of scope |
| `docs/ARCHITECTURE.md` | How the system works; folder rules |
| `docs/DECISIONS.md` | Why each technical choice was made |
| `docs/TEST_PLAN.md` | What "working" means |
