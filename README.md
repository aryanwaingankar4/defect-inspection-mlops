# Self-Improving Steel Defect Inspection System

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

## Documentation

| File | Purpose |
|---|---|
| `docs/PRD.md` | What is being built, and what is out of scope |
| `docs/ARCHITECTURE.md` | How the system works; folder rules |
| `docs/DECISIONS.md` | Why each technical choice was made |
| `docs/TEST_PLAN.md` | What "working" means |
