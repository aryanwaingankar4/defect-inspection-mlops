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

## ADR-011 — Kaggle ↔ MLflow bridge (DECIDED by ADR-018)
- **Options:**
  - **A.** Hosted MLflow on DagsHub. Kaggle logs with a token in Kaggle Secrets.
  - **B.** Offline: Kaggle writes artifacts, and `scripts/register_from_kaggle.py` logs them locally.
- **Status:** Decided in ADR-018: option A (DagsHub) for MLflow only. The DVC remote is NOT DagsHub; it is a private local folder.

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

## ADR-013 — Training runs on the laptop CPU too (SUPERSEDED by ADR-019)
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
- **Reason:** The NEU dataset has no licence to redistribute (ADR-015), so no NEU image may be committed. The synthetic image still catches changes in preprocessing or the model in CI; the real-image test checks behaviour on true data where the data is available. Building the SQLite log on SQLAlchemy now means TASK-037 is a small switch-over (driver, URL, Postgres test run), not a rewrite of the logging code. Reproducing v1 must not depend on Kaggle quota or a GPU, so a reviewer without a GPU can rebuild the model. *(Superseded by ADR-019: v1 is trained on Kaggle GPU.)*

## ADR-018 — DVC remote, MLflow on DagsHub, Kaggle data (closes ADR-011)
- **Decision:**
  - **DVC remote:** a private local folder outside the repo, `../dvc-storage` (stored in `.dvc/config` relative to `.dvc/`). `data/raw` is tracked with `dvc add`; `dvc.yaml` has the `split` and `near_dups` stages. Outputs that git tracks (split CSV, near-duplicate CSV and summary) are `cache: false`; outputs with NEU pixels (`results/private/`) live only in the DVC cache and remote.
  - **MLflow:** hosted on DagsHub in a **public** repository (free tier: unlimited runs for public repos; private repos are capped at 100 tracked experiment runs). It holds metrics, params and models only — **never images with NEU pixels**.
    - Every MLflow artifact call goes through `src/tracking/artifact_policy.safe_log_artifact`. Image-like artifacts (`.png/.jpg/.jpeg/.bmp/.gif/.tif/.webp/.svg/.pdf`) are allowed only as `plots/<name>.png` with `<name>` in an explicit allowlist of metric plots (confusion matrix, reliability diagram, ...). Grad-CAM overlays, sample predictions or duplicate grids are rejected.
    - A test scans `src/` and fails on any direct `log_artifact(s)`, `log_image`, `log_figure` or `log_table` call outside `src/tracking/`.
  - **Kaggle (TASK-010):** upload the NEU images as a **private** Kaggle dataset owned by this project's account, and attach it to the launcher notebook. Kaggle notebooks never use the DVC remote.
- **Reason:** A local folder is private by construction and costs nothing; the data is small and can be rebuilt from the pinned Kaggle source (ADR-015, SHA-256 in the data card). DagsHub gives a hosted MLflow that both Kaggle and the laptop can reach. Making it public avoids the private-repo run cap, which sweeps and loop cycles could exceed, and is safe because the code is already public on GitHub and the artifact policy keeps dataset pixels out. A private Kaggle dataset lets GPU sweeps read the images without redistributing them.
- **Implemented (TASK-009):** `src/tracking/mlflow_tracking.py` is the only module that talks to MLflow (a test fails on `import mlflow` anywhere else in `src/`). The server comes from `MLFLOW_TRACKING_URI` (credentials only from env vars; loaded from `.env` by `uv run --env-file` in `make train*`, from Kaggle Secrets on Kaggle); without it, runs go to a local SQLite store in `./mlruns/`, and tests always use a per-test local store. Files go through `safe_log_artifact`; the model is saved and policy-checked locally before upload, logged with a shape-only signature and **no input example**, and registered as `steel-defect-classifier`. `--smoke` runs use the experiment `neu-smoke` and the model `steel-defect-classifier-smoke`, so the real registry only holds real models.
- **Risks:** The local remote is a single copy on one laptop; losing it means re-downloading with `make data` (checksum-verified). A public MLflow server shows model weights trained on NEU; weights are not images, but this is noted in the model card (TASK-044).

## ADR-019 — Full training on Kaggle GPU only (supersedes ADR-013)
- **Decision:**
  - Every full training run (baseline, sweeps, Phase 3 retrains) runs on **Kaggle GPU**, from repo code through the launcher notebook (TASK-010).
  - The laptop runs only: unit tests, `make train-smoke` (a tiny end-to-end check on CPU, outputs in `models/smoke/`), and later ONNX export, quantisation and serving.
  - `src/training/train.py` enforces this: without a GPU, a full run prints a warning and stops unless `--allow-cpu` is passed. `--smoke` is always allowed.
  - All data and output paths (`--raw-dir`, `--split-file`, `--out-checkpoint`, `--out-dir`) and `--num-workers` can be overridden on the CLI, so the same command runs with `/kaggle/input/...` and `/kaggle/working/...`.
  - Code stays device-agnostic (no CUDA-only paths), because tests and smoke runs use the CPU.
  - **Reproducibility** = pinned git commit + data version + `train.seed`. Every run records the git commit (and whether the tree was dirty), the DVC hash of `data/raw` from the committed `data/raw.dvc`, and a SHA-256 fingerprint of the images actually read. The fingerprint can be compared between the laptop's DVC copy and the Kaggle dataset.
  - **Trained models come back through DagsHub MLflow** (TASK-009): the Kaggle run logs the model; the laptop downloads it by alias for evaluation, ONNX export and serving.
  - **Phase 3:** the Prefect `retrain_flow` on the laptop submits a Kaggle job with `kaggle kernels push`, polls `kaggle kernels status`, and picks up the candidate from MLflow.
- **Reason:** Full training runs out of memory on the 16 GB laptop. Kaggle gives a free GPU, and running every real training there keeps one consistent environment. (Quota: on 2026-11-06 the Kaggle API reported 6 h of GPU per period for this account, refreshing Saturday 00:00 UTC; not the "about 30 h/week" first assumed. `make kaggle-push` prints the live quota.)
- **Consequences / risks:**
  - "`dvc repro` rebuilds v1 from a clean clone" no longer holds on a CPU-only machine. v1 is rebuilt by rerunning the Kaggle launcher at the recorded commit, data version and seed.
  - GPU runs are seeded and use deterministic cuDNN settings, but are not guaranteed bit-for-bit identical across GPU types; CPU runs on one machine are.
  - Kaggle quota, availability and internet access become external dependencies. The Kaggle API token (`kaggle.json`) and the DagsHub token stay out of git (Kaggle Secrets / local user config).
- **Implemented (TASK-010):**
  - `make kaggle-dataset` uploads `data/raw` as the private dataset `neu-det-private` and verifies privacy (`src/data/kaggle_dataset.py`).
  - `make kaggle-push` renders `notebooks/kaggle_train_launcher.ipynb` pinned to `HEAD` (refuses a dirty tree or an unpushed commit) and pushes it as the private notebook `neu-train-launcher` on a T4. P100s cannot run the CUDA 13 torch build; `train.py` stops on a GPU without kernels.
  - MLflow credentials come from three Kaggle Secrets (`MLFLOW_TRACKING_URI`, `_USERNAME`, `_PASSWORD`), attached in the notebook editor and never printed.
  - **Finding:** a run started by `kaggle kernels push` could not read the attached secrets (HTTP 400 from the secrets service), while a run started from the website (**Save Version → Save & Run All**) could. The attachment survives a CLI push. Current workflow: `make kaggle-push ARGS=--no-run`, then start the run on kaggle.com, then `make kaggle-status` / `make kaggle-output`.
  - **Open for Phase 3:** fully automatic submission from the Prefect `retrain_flow` needs another way to give the job its MLflow credentials. To be decided before the Phase 3 `retrain_flow` task.
  - Kaggle runs the launcher under Jupyter, which exports an inline `MPLBACKEND`; `train.py` forces `Agg` before plotting.
  - Baseline v1 (efficientnet_b0): results in `results/train/`, model `steel-defect-classifier` version 1. Two T4 runs at different commits with no training-code change gave identical best-epoch validation metrics.
