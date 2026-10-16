# Architecture Decisions

Each entry records a choice and why it was made, so it is not reverted later. Change an entry only with a new ADR that supersedes it.

## ADR-001 — MLflow aliases, not registry stages
- **Decision:** Use `@champion` (serving) and `@challenger` (candidate) aliases. Promotion moves the alias; rollback moves it back.
- **Reason:** Registry stages are deprecated since MLflow 2.9 and planned for removal ([mlflow#10336](https://github.com/mlflow/mlflow/issues/10336)).

## ADR-002 — Static INT8 quantisation
- **Decision:** ONNX Runtime static QDQ quantisation, with calibration images from `train_v1` and `quant_pre_process` run first. FP32 is kept as a fallback.
- **Reason:** ONNX Runtime recommends static quantisation for CNNs and dynamic for RNNs/transformers ([ORT docs](https://www.onnxruntime.ai/docs/performance/model-optimizations/quantization.html)). Per-class F1 is compared for FP32 vs INT8.

## ADR-003 — SQLite for the prediction log
- **Decision:** SQLite. No Postgres in v1.
- **Reason:** One service and low write volume. The README notes it would move to Postgres with concurrent writers.

## ADR-004 — Retrain from the base checkpoint
- **Decision:** Every retrain starts from the pretrained base on `train_v1 + all feedback`. It never fine-tunes the previous model.
- **Reason:** Stops errors and feedback bias from compounding across cycles, and keeps each version reproducible.

## ADR-005 — Statistical promotion gate
- **Decision:** Champion and challenger are evaluated on the same `test_frozen` images. Promote only if all of these hold:
  - McNemar's test is significant, or the gain is above a minimum margin (alternative: paired bootstrap CI on macro-F1);
  - no class recall drops by more than the tolerance in `params.yaml`;
  - no ECE or latency regression beyond limits.

  Every decision is logged, rejections included.
- **Reason:** The frozen test set is small (one image ≈ 0.28 pp), so a raw accuracy comparison would promote noise.

## ADR-006 — Calibrated confidence and OOD routing
- **Decision:**
  - Temperature scaling fitted on `val`, with `T` stored with the model.
  - The review threshold is chosen from a risk–coverage curve.
  - An energy OOD score is added; route to review if confidence is low OR the OOD score flags the image.
- **Reason:** Raw softmax is often overconfident ([Guo et al. 2017](https://arxiv.org/abs/1706.04599)). Energy scores separate unknown inputs better than softmax ([Liu et al. 2020](https://arxiv.org/abs/2010.03759)).

## ADR-007 — Random audit sample
- **Decision:** A fixed 2–5% of auto-accepted predictions also go to review.
- **Reason:** Reviewing only low-confidence images never measures the errors that ship. The audit gives a live estimate of escaped defects.

## ADR-008 — Evidently on image features
- **Decision:** Use Evidently ≥ 0.7 (new `Report` API, pinned). It runs on per-image features (brightness, contrast, sharpness, embedding PCA) plus prediction and confidence.
- **Reason:** Evidently works on tabular columns, not raw pixels. 0.7 made the new API the default ([migration](https://docs.evidentlyai.com/faq/migration.md)).

## ADR-009 — Single-container Hugging Face Space
- **Decision:** The public demo is one Docker Space (Streamlit + ONNX, port 7860). The full stack runs locally with docker-compose.
- **Reason:** A Docker Space runs one container on one port and cannot run docker-compose.

## ADR-010 — X-SDD as a second production line
- **Decision:** X-SDD classes that overlap NEU give a real-shift stream. X-SDD classes NEU never saw give an unknown-defect stream.
- **Reason:** NEU is near-solved (published 99%+). Real cross-line shift is a stronger test of the loop than synthetic filters alone.

## ADR-011 — Kaggle ↔ MLflow bridge (OPEN)
- **Options:**
  - **A.** Hosted MLflow on DagsHub. Kaggle logs with a token in Kaggle Secrets.
  - **B.** Offline: Kaggle writes artifacts, and `scripts/register_from_kaggle.py` logs them locally.
- **Status:** Not decided. Decide before the first Kaggle run in Week 1.
