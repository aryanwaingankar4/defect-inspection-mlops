# Architecture

## Data flow

1. **Request.** An image is sent to FastAPI, which serves the `@champion` ONNX model (static INT8, FP32 fallback).
2. **Pre-checks.** The API validates the file: size, type, decodability and near-blank frames. A failed check returns a clear 4xx error.
3. **Inference.** The response contains:
   - `prediction`;
   - `calibrated_confidence` (temperature-scaled);
   - `ood_score` (energy score from the logits);
   - `model_version` and `alias`;
   - `threshold`;
   - `routed_to_review`;
   - `request_id`.
4. **Logging.** Every request is written to the prediction database (SQLite locally, Azure PostgreSQL in the cloud, chosen by `DATABASE_URL`) with:
   - the image hash (never the image bytes);
   - the prediction, confidence and OOD score;
   - latency and model version;
   - per-image features: brightness, contrast, sharpness and embedding PCA components.
5. **Routing to review.** An image goes to the Streamlit queue if:
   - its confidence is below the threshold, OR
   - its OOD score flags it, OR
   - it is picked by the random audit sample of auto-accepted images.
6. **Feedback.** Corrected labels are saved to `data/feedback/` and versioned with DVC. The reviewer is simulated in batch runs.
7. **Retrain.** When N new labels exist, the Prefect `retrain_flow` runs on the laptop CPU:
   - it trains from the base checkpoint on `train_v1 + all feedback`;
   - it calibrates on `val`;
   - it evaluates on `test_frozen`.
8. **Promotion gate.** The candidate (`@challenger`) and `@champion` are compared on the same `test_frozen` images. See `DECISIONS.md` (ADR-005) for the rules. Every decision is written to `results/promotion_log.json`.
9. **Deploy.** If the candidate passes, `@champion` moves to it and the model is exported to ONNX. On push to `main`, GitHub Actions (`.github/workflows/deploy-azure.yml`) runs CI, builds the image, pushes it to GitHub Container Registry and deploys the API and review UI Container Apps on Azure.
10. **Monitor.**
    - Local: Prometheus + Grafana for latency, throughput, confidence histogram, review-queue size and audit error rate.
    - Cloud: Azure Monitor (Log Analytics logs from the structured JSON logs, Container Apps metrics).
    - Evidently: drift reports on the logged feature table.
11. **Rollback.** `scripts/rollback.py` moves `@champion` back to the previous version. In the cloud, rollback redeploys the previous image tag (Container Apps revision).

## Environments

| Where | What runs |
|---|---|
| Laptop (docker-compose) | API, review UI, MLflow, Prometheus, Grafana, Prefect retrain, SQLite. Training runs here on CPU too (ADR-013) |
| Kaggle GPU | Training sweeps and loop experiments only, run from repo code |
| Azure Container Apps | Main deployment: API app + review UI app, scale to zero, secrets in Container Apps secrets (ADR-012) |
| Azure Database for PostgreSQL | Flexible Server, smallest tier: prediction log and feedback in the cloud (ADR-003) |
| Azure Monitor | Cloud logs (Log Analytics) and Container Apps metrics |
| GitHub Actions + GHCR | CI, image build and push to GitHub Container Registry, deploy to Azure; Azure credentials only in GitHub Secrets |
| DVC remote (`../dvc-storage`) | Private local folder outside the repo: raw images and pixel outputs (ADR-018) |
| DagsHub (public repo) | Hosted MLflow: metrics, params, models only; never NEU images (ADR-018) |
| Kaggle private dataset | NEU images for GPU sweeps, attached to the launcher notebook (ADR-018) |
| Hugging Face Space (optional) | One container: Streamlit demo + ONNX model, port 7860 (ADR-009) |

## Folder structure

```
defect-inspection-mlops/
├── configs/params.yaml        # seeds, hyperparameters, thresholds, gate rules
├── data/                      # DVC-tracked: raw/, splits/, feedback/, xsdd/
├── docs/                      # PRD, architecture, decisions, test plan, memory, learning log, data + model cards
├── src/
│   ├── data/                  # download, split, pHash check, drift generators
│   ├── training/              # train, evaluate, calibrate, export_onnx, quantize
│   ├── pipelines/             # Prefect retrain_flow, promotion gate
│   ├── serving/               # FastAPI app, ONNX inference, preprocess.py (shared), logging
│   ├── review_ui/             # Streamlit queue
│   ├── monitoring/            # Prometheus metrics, Evidently reports
│   ├── tracking/              # MLflow helpers; artifact_policy.py guards against image artifacts
│   └── simulation/            # production stream, simulated reviewer
├── scripts/                   # rollback.py, register_from_kaggle.py
├── notebooks/                 # Kaggle launchers only (clone + run), no logic
├── tests/
├── docker/                    # Dockerfiles, compose, Grafana dashboards
├── infra/
│   └── azure/                 # Azure setup scripts/notes: resource group, budget alert, Postgres, Container Apps
├── .github/workflows/
│   └── deploy-azure.yml       # CI → build → push to GHCR → deploy to Azure Container Apps
├── results/                   # source of every reported number
├── dvc.yaml
├── Makefile
└── README.md
```

## Folder rules

- `src/serving/preprocess.py` is the only place that resizes, normalises or handles channels. Training, evaluation and serving all import it.
- No training or evaluation logic in notebooks.
- `test_frozen` is read only by the promotion gate and final reports.
- All configurable values come from `configs/params.yaml`.
- `results/` holds committed metric files (JSON/CSV/PNG). The README and resume quote only these.
- The serving image contains ONNX Runtime, not PyTorch.
- `infra/azure/` holds setup scripts and notes only, never credentials. Azure credentials live in GitHub Secrets; runtime secrets live in Container Apps secrets.
- MLflow artifacts are logged only through `src/tracking/artifact_policy.safe_log_artifact`; the public DagsHub server never receives images with NEU pixels (ADR-018).
- Outputs containing NEU pixels go to `results/private/` (git-ignored, DVC-cached only).
- Database access goes through one layer that reads `DATABASE_URL`. No SQLite- or Postgres-specific code outside it.
