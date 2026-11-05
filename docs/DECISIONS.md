# Architecture Decisions

Each entry records a choice and why it was made, so it is not reverted later. Change an entry only with a new ADR that supersedes it.

## ADR-001 — MLflow aliases, not registry stages
- **Decision:** Use `@champion` (serving) and `@challenger` (candidate) aliases. Promotion moves the alias; rollback moves it back.
- **Reason:** Registry stages are deprecated since MLflow 2.9 and planned for removal ([mlflow#10336](https://github.com/mlflow/mlflow/issues/10336)).

## ADR-002 — Static INT8 quantisation
- **Decision:** ONNX Runtime static QDQ quantisation, with calibration images from `train_v1` and `quant_pre_process` run first. FP32 is kept as a fallback.
- **Reason:** ONNX Runtime recommends static quantisation for CNNs and dynamic for RNNs/transformers ([ORT docs](https://www.onnxruntime.ai/docs/performance/model-optimizations/quantization.html)). Per-class F1 is compared for FP32 vs INT8.

## ADR-003 — SQLite locally, Postgres in the cloud (amended 2026-11-06)
- **Decision:** One database layer, chosen by the `DATABASE_URL` env var:
  - SQLite on the laptop and in tests;
  - Azure Database for PostgreSQL Flexible Server (smallest tier) on Azure.

  The application code is the same for both. Only `DATABASE_URL` changes.
- **Reason:** Locally there is one service and low write volume, so SQLite is enough. On Azure, container filesystems are lost on restart and on scale to zero, and two apps (API and review UI) share the data, so a SQLite file there would lose data.
- **History:** Originally "SQLite only, no Postgres in v1". Amended when deployment moved to Azure (ADR-012).

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

## ADR-009 — Single-container Hugging Face Space (SUPERSEDED by ADR-012)
- **Status:** Superseded. Azure Container Apps is now the main deployment. A Hugging Face Space is optional; if one is built, the rule below still applies.
- **Decision:** The public demo is one Docker Space (Streamlit + ONNX, port 7860). The full stack runs locally with docker-compose.
- **Reason:** A Docker Space runs one container on one port and cannot run docker-compose.

## ADR-010 — X-SDD as a second production line
- **Decision:** X-SDD classes that overlap NEU give a real-shift stream. X-SDD classes NEU never saw give an unknown-defect stream.
- **Reason:** NEU is near-solved (published 99%+). Real cross-line shift is a stronger test of the loop than synthetic filters alone.

## ADR-011 — Kaggle ↔ MLflow bridge (OPEN)
- **Options:**
  - **A.** Hosted MLflow on DagsHub. Kaggle logs with a token in Kaggle Secrets.
  - **B.** Offline: Kaggle writes artifacts, and `scripts/register_from_kaggle.py` logs them locally.
- **Status:** Not decided. Leaning A: DagsHub for MLflow and as the DVC remote. Decide at TASK-005 (DVC remote setup), before the first Kaggle run.

## ADR-012 — Azure Container Apps as the main deployment (supersedes ADR-009)
- **Decision:**
  - The API (FastAPI + ONNX) and the Streamlit review UI run as two Azure Container Apps in one Container Apps environment, with scale to zero (min replicas 0).
  - Images are pushed to GitHub Container Registry (GHCR).
  - GitHub Actions runs CI, builds the image, pushes it to GHCR and deploys to Azure on every push to `main`.
  - Azure deploy credentials live only in GitHub Secrets. Runtime secrets (API key, `DATABASE_URL`, GHCR pull credentials if the package is private) live only in Container Apps secrets.
  - Prediction log and feedback go to Azure Database for PostgreSQL Flexible Server (see ADR-003).
  - Cloud monitoring uses Azure Monitor (Log Analytics logs + Container Apps metrics). Prometheus + Grafana stay in the local docker-compose stack.
  - Cost safety: Azure for Students subscription, a budget alert at about $10, scale to zero, smallest Postgres tier. Postgres does not scale to zero, so it is the main standing cost; stop it when idle.
  - Rollback in the cloud means redeploying the previous image tag (Container Apps revision).
  - A Hugging Face Space is optional (ADR-009 rules apply if it is built).
- **Reason:** Two separate apps with HTTPS ingress, managed secrets, a managed database and an automated deploy from `main` are closer to a real production setup than a single-container demo Space. Scale to zero and a budget alert keep it within student credit.

## ADR-013 — Training runs on the laptop CPU too
- **Decision:** `python -m src.training.train --config configs/params.yaml` must run end to end on a CPU-only laptop. The device is selected automatically (or from config), with no CUDA-only code paths. Kaggle GPU is used only for sweeps and loop experiments.
- **Reason:** `dvc repro` from a clean clone and the Prefect retrain flow both run on the laptop.

## ADR-014 — SQLAlchemy for the database layer from TASK-019
- **Decision:** The prediction log is built with SQLAlchemy at TASK-019 and reads `DATABASE_URL` from the start (SQLite URL locally and in tests). The dependency is approved for TASK-019. A Postgres driver is added only at TASK-037, and needs approval then.
- **Reason:** Postgres is needed on Azure (ADR-003).

## ADR-015 — NEU data source and licence
- **Decision:**
  - Use the Kaggle copy `kaustubhdikshit/neu-surface-defect-database`, pinned to version 1. It is the authors' NEU-DET release: the same 1,800 images as NEU-CLS (6 classes × 300, 200×200), stored as `.jpg` instead of NEU-CLS's `.bmp`. The JPEG files have 3 channels (checked from the file headers), although the source images are grayscale; whether the 3 channels are identical is checked in TASK-006/007 and feeds the grayscale-handling decision. The class comes from the folder name. The uploader's train/validation split and the XML boxes are ignored; TASK-003 makes our own split.
  - The dataset has **no formal licence.** The authors (Kechen Song, Yunhui Yan, Northeastern University) only ask for citation. We use it for non-commercial academic/portfolio work only, cite the 3 papers listed in `docs/DATA_CARD.md`, and **never redistribute the images**.
  - Consequences: images are never committed to git; the DVC remote must be private; no dataset images in a public demo, Docker image or README without a deliberate decision.
- **Reason:** The authors' own links (Google Drive / Baidu Pan) cannot be scripted reliably. The Kaggle copy can be downloaded with one pinned command, has a complete class-per-folder layout, and is the most-used NEU copy on Kaggle. The zip's SHA-256 is recorded in the data card to detect changes. JPEG instead of BMP may differ slightly at the pixel level from NEU-CLS, so published NEU-CLS numbers are not directly comparable.

## ADR-016 — Grayscale handling: load as L, replicate to 3 channels
- **Decision:** Every image is opened as grayscale (PIL mode `L`), then replicated to 3 identical channels before normalisation. This lives only in `src/serving/preprocess.py` (TASK-007) and is used by training, evaluation and serving. The model keeps its standard 3-channel ImageNet input (`in_chans=3`).
- **Reason:** The NEU source images are grayscale, but the Kaggle JPEGs store 3 channels (ADR-015), and uploads at serving time may be RGB. Forcing `L` first removes any colour or JPEG chroma noise, so every path sees the same input. Replicating to 3 channels keeps the pretrained ImageNet first-layer weights as they are, instead of changing them for `in_chans=1`.

## ADR-017 — Golden-image tests without redistributing NEU
- **Decision:** At TASK-020 there are two golden tests:
  - a committed **synthetic** image, generated by a script in the repo, with its expected class/output stored next to it. It runs in CI;
  - a **real NEU** golden test, marked `needs_data`, which reads a fixed image from `data/raw/` and never commits it.
- **Reason:** The NEU dataset has no licence to redistribute (ADR-015), so no NEU image may be committed. The synthetic image still catches changes in preprocessing or the model in CI; the real-image test checks behaviour on true data where the data is available. Building the SQLite log on SQLAlchemy now means TASK-037 is a small switch-over (driver, URL, Postgres test run), not a rewrite of the logging code. Reproducing v1 must not depend on Kaggle quota or a GPU, so a reviewer without a GPU can rebuild the model.
