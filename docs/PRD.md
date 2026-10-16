# Product Requirements Document

## Product
Self-Improving Steel Defect Inspection System

## Problem
Steel plants inspect strip surfaces for defects with cameras. A model trained once degrades when lighting, cameras or defect types change. Labelling everything up front is impossible. The model must keep learning from inspectors' corrections without silently getting worse.

## Users
- **Line inspector (reviewer):** reviews images the model is unsure about and corrects labels.
- **ML engineer:** owns training, promotion, rollback and monitoring.
- **Portfolio reader / interviewer:** must be able to run the project and verify every claim.

## Goal
Show a complete, trustworthy production loop for a vision model. The loop is: serve, flag uncertain or unknown inputs, collect corrections, retrain, then promote only with statistical evidence, while monitoring and rolling back when needed.

## MVP
1. Baseline classifier on NEU-CLS with calibrated confidence and a threshold chosen from a risk–coverage curve.
2. FastAPI service (ONNX, static INT8) that returns:
   - the prediction;
   - the calibrated confidence;
   - the OOD score;
   - the model version.
3. Prediction log (SQLite) with per-image features.
4. Streamlit review queue fed by three routes: low confidence, OOD, and a random audit sample.
5. Versioned feedback (DVC). A Prefect retrain flow is triggered after N labels.
6. A promotion gate using McNemar or bootstrap on `test_frozen`, with per-class limits. It logs every decision, and promotion moves the MLflow alias.
7. A rollback script.
8. Monitoring: Prometheus/Grafana plus Evidently drift on image features.
9. CI (ruff, pytest, Trivy) and a public Hugging Face Space demo.

## Out of scope (v1)
- Postgres (SQLite is enough)
- GCP Cloud Run deployment
- Hosted Grafana
- Object detection / localisation (YOLOv8n): stretch only
- Active learning: stretch only
- Real factory data or real reviewers
- Multi-camera / edge deployment (described in the README only)

## Success criteria
- `dvc repro` rebuilds the v1 model from a clean clone.
- The API runs in Docker, rejects bad inputs, and returns version and confidence fields.
- The simulated loop produces versions v1 → v3 with at least one gate rejection, and every decision is logged in `results/`.
- Rollback restores the previous `@champion`.
- Drift is detected on the shifted streams. The false-alarm rate on the clean stream is reported.
- A stranger can run the demo from the README.
- Every number in the README and resume comes from a file in `results/`.

## Constraints
- Free tools only.
- A 16 GB CPU laptop, plus Kaggle GPU (about 30 h/week) for training sweeps.
- 8–12 h/week. Target: 5 weeks.
