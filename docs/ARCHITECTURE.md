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
4. **Logging.** Every request is written to SQLite with:
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
9. **Deploy.** If the candidate passes, `@champion` moves to it, the model is exported to ONNX, and GitHub Actions rebuilds the image.
10. **Monitor.**
    - Prometheus + Grafana: latency, throughput, confidence histogram, review-queue size, audit error rate.
    - Evidently: drift reports on the logged feature table.
11. **Rollback.** `scripts/rollback.py` moves `@champion` back to the previous version.

## Environments

| Where | What runs |
|---|---|
| Laptop (docker-compose) | API, review UI, MLflow, Prometheus, Grafana, Prefect retrain |
| Kaggle GPU | Training sweeps and loop experiments only, run from repo code |
| Hugging Face Space | One container: Streamlit demo + ONNX model, port 7860 |

## Folder structure

```
defect-inspection-mlops/
├── configs/params.yaml        # seeds, hyperparameters, thresholds, gate rules
├── data/                      # DVC-tracked: raw/, splits/, feedback/, xsdd/
├── docs/                      # PRD, architecture, decisions, test plan, memory, data + model cards
├── src/
│   ├── data/                  # download, split, pHash check, drift generators
│   ├── training/              # train, evaluate, calibrate, export_onnx, quantize
│   ├── pipelines/             # Prefect retrain_flow, promotion gate
│   ├── serving/               # FastAPI app, ONNX inference, preprocess.py (shared), logging
│   ├── review_ui/             # Streamlit queue
│   ├── monitoring/            # Prometheus metrics, Evidently reports
│   └── simulation/            # production stream, simulated reviewer
├── scripts/                   # rollback.py, register_from_kaggle.py
├── notebooks/                 # Kaggle launchers only (clone + run), no logic
├── tests/
├── docker/                    # Dockerfiles, compose, Grafana dashboards
├── .github/workflows/
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
