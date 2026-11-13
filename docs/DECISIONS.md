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
> **Superseded in part by ADR-033 (2026-11-12):** the gate runs on `val` (not `test_frozen`), as a non-inferiority test, without ECE/latency checks.

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

- **Implemented (TASK-012):** `make calibrate` (`src/training/calibrate.py`) fits one temperature per model on `val` only (mean NLL; declared range `calibrate.t_min`..`t_max`; a fit on a bound writes nothing and stops). `T` is stored in `calibration.json` next to each model's results (`results/train/`, `results/compare/mobilenetv3_small_100/`), with what it was fitted on and the checkpoint hash; TASK-015 applies it in ONNX / serving. Val reports ECE, NLL and Brier before/after in-sample and cross-fitted (5 class-stratified folds, near-copy groups kept together), with grouped bootstrap intervals for each score, the change, and `T` itself. `make evaluate` applies `T` on `test_frozen` (never fits it) and reports the same scores. With only a few misclassified images in each set, the intervals of the changes include 0 for both models on test; see the JSON files for the numbers. `T` is kept as fitted: nothing is chosen from these results.
- **Owner decision (2026-11-08):** MobileNet's `T` rests on a single misclassified `val` image and is poorly determined (its `calibration.json` summary says how often the fit sat on a range bound). It is still kept as fitted rather than reset to `T = 1`, so both models follow the same rule.

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
- **Implementation note (TASK-020, owner decision 2026-11-10):**
  - The synthetic golden image is committed as `tests/golden/synthetic_golden.png`, made by a formula in `src/serving/golden.py` (no random generator, no dataset pixels). `tests/test_download.py` allows only `tests/golden/synthetic_*.png` outside `results/`, and a test regenerates the image and requires the committed file to decode to exactly the same pixels, which shows it holds no NEU pixels.
  - CI part: the image's preprocessed tensor must match committed summary values, and the API on a small hand-built test model must give the output computed independently from those values.
  - Real-model part (`needs_model`, and `needs_data` for NEU): `tests/golden/expected.json` holds the `@champion` outputs (class, calibrated confidence, energy, routing reasons, logits) for the synthetic image and two fixed `val` images (the first `val` image in the split file and the `val` image with the lowest calibrated confidence), together with the model version and ONNX SHA-256 they belong to. `make golden` writes it and refuses unless the two NEU results agree with `results/train/val_predictions.csv` (PyTorch). A new champion makes the test fail until `make golden` is re-run on purpose. Never `test_frozen`.
  - Tolerance as ADR-025: logits within `1e-4 + 1e-4 · |logit|`; class and routing exact.

## ADR-018 — DVC remote, MLflow on DagsHub, Kaggle data (closes ADR-011)
- **Decision:**
  - **DVC remote:** a private local folder outside the repo, `../dvc-storage` (stored in `.dvc/config` relative to `.dvc/`). `data/raw` is tracked with `dvc add`; `dvc.yaml` has the `split` and `near_dups` stages. Outputs that git tracks (split CSV, near-duplicate CSV and summary) are `cache: false`; outputs with NEU pixels (`results/private/`) live only in the DVC cache and remote.
  - **MLflow:** hosted on DagsHub in a **public** repository (free tier: unlimited runs for public repos; private repos are capped at 100 tracked experiment runs). It holds metrics, params and models only — **never images with NEU pixels**.
    - Every MLflow artifact call goes through `src/tracking/artifact_policy.safe_log_artifact`. Image-like artifacts (`.png/.jpg/.jpeg/.bmp/.gif/.tif/.webp/.svg/.pdf`) are allowed only as `plots/<name>.png` with `<name>` in an explicit allowlist of metric plots (confusion matrix, reliability diagram, ...). Grad-CAM overlays, sample predictions or duplicate grids are rejected.
    - A test scans `src/` and fails on any direct `log_artifact(s)`, `log_image`, `log_figure` or `log_table` call outside `src/tracking/`.
  - **Kaggle (TASK-010):** upload the NEU images as a **private** Kaggle dataset owned by this project's account, and attach it to the launcher notebook. Kaggle notebooks never use the DVC remote.
- **Reason:** A local folder is private by construction and costs nothing; the data is small and can be rebuilt from the pinned Kaggle source (ADR-015, SHA-256 in the data card). DagsHub gives a hosted MLflow that both Kaggle and the laptop can reach. Making it public avoids the private-repo run cap, which sweeps and loop cycles could exceed, and is safe because the code is already public on GitHub and the artifact policy keeps dataset pixels out. A private Kaggle dataset lets GPU sweeps read the images without redistributing them.
- **Implemented (TASK-009):** `src/tracking/mlflow_tracking.py` is the only module that talks to MLflow (a test fails on `import mlflow` anywhere else in `src/`). The server comes from `MLFLOW_TRACKING_URI` (credentials only from env vars; loaded from `.env` by `uv run --env-file` in `make train*`, from Kaggle Secrets on Kaggle, superseded by ADR-020: Kaggle jobs no longer log to MLflow); without it, runs go to a local SQLite store in `./mlruns/`, and tests always use a per-test local store. Files go through `safe_log_artifact`; the model is saved and policy-checked locally before upload, logged with a shape-only signature and **no input example**, and registered as `steel-defect-classifier`. `--smoke` runs use the experiment `neu-smoke` and the model `steel-defect-classifier-smoke`, so the real registry only holds real models.
- **Risks:** The local remote is a single copy on one laptop; losing it means re-downloading with `make data` (checksum-verified). A public MLflow server shows model weights trained on NEU; weights are not images, but this is noted in the model card (TASK-044).

## ADR-019 — Full training on Kaggle GPU only (supersedes ADR-013; credentials and model return superseded by ADR-020)
- **Decision:**
  - Every full training run (baseline, sweeps, Phase 3 retrains) runs on **Kaggle GPU**, from repo code through the launcher notebook (TASK-010).
  - The laptop runs only: unit tests, `make train-smoke` (a tiny end-to-end check on CPU, outputs in `models/smoke/`), and later ONNX export, quantisation and serving.
  - `src/training/train.py` enforces this: without a GPU, a full run prints a warning and stops unless `--allow-cpu` is passed. `--smoke` is always allowed.
  - All data and output paths (`--raw-dir`, `--split-file`, `--out-checkpoint`, `--out-dir`) and `--num-workers` can be overridden on the CLI, so the same command runs with `/kaggle/input/...` and `/kaggle/working/...`.
  - Code stays device-agnostic (no CUDA-only paths), because tests and smoke runs use the CPU.
  - **Reproducibility** = pinned git commit + data version + `train.seed`. Every run records the git commit (and whether the tree was dirty), the DVC hash of `data/raw` from the committed `data/raw.dvc`, and a SHA-256 fingerprint of the images actually read. The fingerprint can be compared between the laptop's DVC copy and the Kaggle dataset.
  - **Trained models come back through DagsHub MLflow** (TASK-009): the Kaggle run logs the model; the laptop downloads it by alias for evaluation, ONNX export and serving. *(Superseded by ADR-020: the Kaggle job logs nothing; the laptop publishes the run and model.)*
  - **Phase 3:** the Prefect `retrain_flow` on the laptop submits a Kaggle job with `kaggle kernels push`, polls `kaggle kernels status`, and picks up the candidate from MLflow.
- **Reason:** Full training runs out of memory on the 16 GB laptop. Kaggle gives a free GPU, and running every real training there keeps one consistent environment. (Quota: on 2026-11-06 the Kaggle API reported 6 h of GPU per period for this account, refreshing Saturday 00:00 UTC; not the "about 30 h/week" first assumed. `make kaggle-push` prints the live quota.)
- **Consequences / risks:**
  - "`dvc repro` rebuilds v1 from a clean clone" no longer holds on a CPU-only machine. v1 is rebuilt by rerunning the Kaggle launcher at the recorded commit, data version and seed.
  - GPU runs are seeded and use deterministic cuDNN settings, but are not guaranteed bit-for-bit identical across GPU types; CPU runs on one machine are.
  - Kaggle quota, availability and internet access become external dependencies. The Kaggle API token (`kaggle.json`) and the DagsHub token stay out of git (local user config only; no token on Kaggle since ADR-020).
- **Implemented (TASK-010):**
  - `make kaggle-dataset` uploads `data/raw` as the private dataset `neu-det-private` and verifies privacy (`src/data/kaggle_dataset.py`).
  - `make kaggle-push` renders `notebooks/kaggle_train_launcher.ipynb` pinned to `HEAD` (refuses a dirty tree or an unpushed commit) and pushes it as the private notebook `neu-train-launcher` on a T4. P100s cannot run the CUDA 13 torch build; `train.py` stops on a GPU without kernels.
  - *(Superseded by ADR-020.)* The first version read MLflow credentials from three Kaggle Secrets. **Finding:** a run started by `kaggle kernels push` could not read them (HTTP 400 from the secrets service), while a run started from the website (Save Version → Save & Run All) could; the attachment itself survived a CLI push. Baseline v1 was trained that way.
  - Kaggle runs the launcher under Jupyter, which exports an inline `MPLBACKEND`; `train.py` forces `Agg` before plotting.
  - Baseline v1 (efficientnet_b0): results in `results/train/` (now `results/archive/split_v1/train/`, ADR-021), model `steel-defect-classifier` version 1. Two T4 runs at different commits with no training-code change gave identical best-epoch validation metrics.

## ADR-020 — Credential-free Kaggle jobs; the laptop publishes runs to MLflow (supersedes the credential and model-return parts of ADR-019)
- **Decision:**
  - The Kaggle job holds **no credentials** and does not talk to MLflow. `src/training/train.py` only writes files to `/kaggle/working/`: the best checkpoint, `val_metrics.json` (best-epoch metrics, provenance: git commit, dirty flag, DVC hash of `data/raw`, image fingerprint, device, runtime; model name, parameter count, a random `run_uid`, training start/end times, checkpoint SHA-256), `training_curves.csv` (per-epoch), the curve plot and the effective `params.yaml`. A test fails if `train.py` imports MLflow or `src.tracking`; the launcher template must not contain Kaggle Secrets or `MLFLOW` names.
  - On the laptop: `make kaggle-output` downloads the files; `make register` (`src/tracking/publish.py`) publishes them as ONE MLflow run: params, per-epoch metrics (step = epoch), best metrics, provenance tags, the allowlisted files through `safe_log_artifact` (ADR-018), and the model rebuilt from the checkpoint (no pretrained download). The run keeps the real training start/end times.
  - **Registration only on request:** `ARGS=--register`. `--role comparison` adds the tag `role=comparison` and can never be registered. A `run_uid` is published at most once; `--register` on an already published run registers its existing model (once). The checkpoint must match the recorded SHA-256.
  - `make kaggle-push [ARGS="--model NAME"]` starts the run directly (no website step again). Extra training arguments are rendered into the notebook as plain tokens only (no quotes or spaces).
  - Smoke runs use the same path: `make train-smoke` trains, then publishes to `neu-smoke` and the smoke model name.
- **Reason:** API-started Kaggle runs could not read Kaggle Secrets (ADR-019 finding), which blocked automatic runs and the Phase 3 `retrain_flow`. Not sending the DagsHub token to Kaggle at all also removes a secret from a third-party machine, and one publishing path (shared by Kaggle and smoke runs) is tested once instead of twice.
- **Consequences / risks:**
  - A run reaches MLflow only after someone (or the Phase 3 flow) runs `make kaggle-output` + `make register`; until then it exists only as Kaggle output files. Kaggle keeps the output of each notebook version.
  - The MLflow run's start/end times are the training times, but the upload happens later; the tag `published_from=laptop` says so.
  - Baseline v1 (`steel-defect-classifier` version 1) was logged directly from Kaggle under ADR-019 and stays as it is; its `results/train/val_metrics.json` has the older layout (no `run_uid`, timing or parameter count).
  - The DagsHub token that was stored in Kaggle Secrets should be deleted there and rotated.
- **Implemented (TASK-010):** `src/tracking/publish.py` (`make register`), commit `a681102`. First run through the new flow: MobileNetV3-small (`mobilenetv3_small_100.lamb_in1k`) started by `make kaggle-push` alone, published with `--role comparison` and not registered; results in `results/compare/mobilenetv3_small_100/` next to the baseline in `results/train/`.

## ADR-021 — Group-aware split v2: overlapping NEU crops stay in one split (supersedes the split rule of TASK-003 / ADR-015)
- **Finding (TASK-008b):** NEU images are crops of longer steel strips, and many crops overlap. pHash (TASK-004) missed this because it is not shift-tolerant. A shift-tolerant check (`src/data/near_duplicates_cnn.py`) found confirmed near-copies across every pair of splits of `neu_split_v1.csv`, including `train_v1` / `val` and `test_frozen` / all other splits. Counts are in `results/near_copies_summary.json`, `results/archive/split_v1/near_duplicates_cnn_summary.json` and `docs/DATA_CARD.md`.
- **Decision:**
  - **Near-copy rule.** Every pair of images (any split) is scored. A pretrained CNN (ImageNet weights, head removed, forward passes only; not our model) shortlists pairs with cosine >= `near_duplicates_cnn.candidate_cosine`. Each shortlisted pair gets a pixel check: the best Pearson correlation of the high-passed texture (local mean removed) over shifts up to `near_duplicates_cnn.max_shift` px. Confirmed if >= `near_duplicates_cnn.confirm_pixel_corr` (0.3).
  - **Why two signals:** cosine alone cannot separate copies from different surfaces of the same class (random same-class pairs reach the cosine of real copies; the TASK-004 suspect pair ranked in the thousands by cosine). Raw pixel correlation is fooled by shared lighting gradients; removing the local mean fixes that. 0.3 sits well above the random same-class baseline (p95 in the summary) and matches a visual check of borderline pairs: real overlaps were seen down to about 0.3.
  - **Group-aware split v2** (`data/splits/neu_split_v2.csv`, now `data.split_file`): confirmed near-copies are joined into groups (union-find, so chains join too); each group stays in one split. Per class, groups and single images are shuffled with `split.seed` and placed first-fit in `split.per_class` order; groups larger than `split.groups.max_size_outside_last` (3) always go to `production_stream`. Counts per class are unchanged. With no groups the rule gives exactly the v1 split. v1 stays committed (`data.split_file_v1`) for history.
  - **Pipeline:** `near_copies` (all pairs, writes the committed `data/splits/near_copy_pairs.csv`) -> `split` (v1 + v2) -> `near_dups` and `near_dups_cnn` on v2. `near_dups_cnn` runs with `--require-clean` and fails if a confirmed near-copy crosses splits; `make kaggle-push` refuses to train unless the committed report for `data.split_file` shows 0. Training provenance now records `split_file` and `split_sha256`.
  - **Retrain:** train_v1 and val changed, so the baseline is retrained on v2 on Kaggle and registered as `steel-defect-classifier` version 2. Version 1 is kept, tagged as trained on the leaky split `neu_split_v1`, with a description. v1-era results move to `results/archive/split_v1/`; new results go to `results/train/` and `results/compare/`.
- **Reason:** `test_frozen` must not share surfaces with anything that trains, selects or feeds back into training (`production_stream` labels join retraining in Phase 3), and `val` must not share surfaces with `train_v1` or validation metrics are optimistic. Grouping fixes the cause instead of removing images, so class counts stay balanced.
- **Consequences / risks:**
  - Recall is not perfect: a copy shifted by more than `max_shift` px, or with cosine below `candidate_cosine`, can be missed; pairs just below 0.3 are not grouped (they are kept in the pairs file for review).
  - Long chains (mostly `scratches`) all sit in `production_stream`, so the other splits hold fewer chain images than a plain random split.
  - The byte-identical pair `patches_101` / `patches_105` is one group and lands in `test_frozen`, which counts that image twice (documented in the data card).
  - v1 metrics (archived) are not comparable 1:1 with v2 metrics: both `train_v1` and `val` changed, not only the leakage.
  - `make near-copies` takes about 20 minutes on the laptop CPU (forward passes + FFT pixel checks); it only reruns when the data or its parameters change.
- **Implemented (TASK-008b):** commit `9c9845f` (check, split v2, gate). Retrained on v2 on Kaggle: baseline efficientnet_b0 at `9c9845f`, registered as `steel-defect-classifier` version 2, results in `results/train/`; MobileNetV3-small comparison at `695b306` (not registered), results in `results/compare/mobilenetv3_small_100/`. Both report the same data fingerprint. Version 1 and the v1 comparison run are tagged `split_file=neu_split_v1.csv`, `leaky_split=true` in MLflow; their results are in `results/archive/split_v1/`.

## ADR-022 — Main model choice is fixed on `val` before `test_frozen` is read
- **Context:** On split v2, MobileNetV3-small has a higher val macro-F1 than the baseline efficientnet_b0 (`results/compare/mobilenetv3_small_100/val_metrics.json` vs `results/train/val_metrics.json`). The gap is a few of the 180 val images. efficientnet_b0 is the pre-declared main model (`train.model`); MobileNetV3-small was trained only as a comparison (TASK-010).
- **Decision (written and committed before the test below was run, and before any model reads `test_frozen`):**
  - **Test:** exact two-sided McNemar test on `val` of split v2: both best checkpoints predict the same 180 images (CPU inference only); the p-value is the exact binomial test on the discordant images (one model right, the other wrong). Significance level `compare_val.alpha` (0.05) in `params.yaml`. Output: `results/compare/val_mcnemar.json`.
  - **If p >= 0.05:** efficientnet_b0 stays the main model (`steel-defect-classifier`, the candidate for `@champion` in TASK-014). MobileNetV3-small is reported as the lighter alternative, with its own results, and is not registered.
  - **If p < 0.05:** no automatic switch. Work stops and the project owner decides.
  - **Sanity check first:** the CPU predictions must reproduce each model's recorded val macro-F1 and accuracy; if not, stop before testing.
  - **`test_frozen` never changes this choice.** TASK-011 evaluates both models on `test_frozen` and reports the numbers, but the main model stays the one chosen here.
- **Reason:** Choosing between models on `test_frozen` would turn the test set into a selection set and make its numbers optimistic. Fixing the rule before seeing the result stops us from picking whichever rule favours the result.
- **Consequences / risks:**
  - `val` was also used for early stopping of both models, so both val scores are slightly optimistic; the paired test still compares them on equal terms.
  - With 180 images and few discordant pairs, the test has low power: "not significant" means "no evidence of a difference", not "the models are equal".
- **Result (`make compare-val`, `src/training/compare_val.py`):** both checkpoints reproduced their recorded val metrics on CPU. Of 180 val images, 5 are discordant: 1 only efficientnet_b0 got right, 4 only MobileNetV3-small got right; exact p = 0.375 (`results/compare/val_mcnemar.json`). Not significant, so efficientnet_b0 stays the main model and MobileNetV3-small is the lighter alternative.
- **Implemented (TASK-011):** `make evaluate` (`src/training/evaluate.py`, commit `cd6147c`) reads `test_frozen` only after these checks pass: the split is clean, the comparison above is not significant for the current split, and the checkpoints are the ones compared. It evaluates efficientnet_b0 (main) first, then MobileNetV3-small (lighter alternative), on CPU, with no head-to-head test. Intervals use a cluster bootstrap stratified by class: the confirmed near-copy groups inside `test_frozen` are resampled as whole units. Results are in `results/eval/<model>/`; `--log-mlflow` added the test metrics and allowlisted plots to each model's existing MLflow run. The model choice is unchanged by these results.

## ADR-023 — Review threshold from a fixed review budget on `val`, with a risk bound as a stop check
- **Context:** ADR-006 says the review threshold comes from a risk–coverage curve. *Coverage* is the share of images auto-accepted (calibrated confidence at or above the threshold); *risk* is the error rate among those accepted images. The main model makes only a handful of errors on the 180 `val` images (`results/train/calibration.json`), so no threshold can be shown to keep risk at 1%: even with zero errors among about 171 accepted images, the one-sided 95% upper bound on risk is about 1.7%, and about 299 accepted images with zero errors would be needed to show 1%.
- **Decision (written and committed before the threshold is computed and before any val error is looked up by confidence):**
  - **Model:** the main model only (efficientnet_b0, ADR-022). MobileNetV3-small is not served and gets no threshold.
  - **Confidence:** the top-label calibrated confidence, `softmax(logits / T)` with `T` from `calibration.json` (ADR-006); the run refuses a calibration for another checkpoint or split.
  - **Rule (review budget):** route to review the `ceil(threshold.review_budget × n)` `val` images with the lowest calibrated confidence (`review_budget` = 0.05 in `params.yaml`: 9 of 180). The threshold is the midpoint between the highest routed and the lowest accepted confidence; an image is routed if its confidence is below the threshold. Labels play no part in choosing it.
  - **Stop check:** among the accepted `val` images, compute the one-sided Clopper–Pearson upper bound on risk at `threshold.ucb_level` (0.95). If it is above `threshold.max_risk_ucb` (0.05), nothing is written, the run stops, and the project owner decides.
  - **Evidence stored** in `results/train/threshold.json`, tied to the hashes of the split file, the checkpoint and `calibration.json`: the threshold, routed and accepted counts, routed and accepted errors, risk and its bound, the full risk–coverage curve with grouped bootstrap bands (near-copy groups resampled together), and a statement of what this does and does not show. Also reported, not used by the stop check: the same bound counting each near-copy group as one unit.
  - **`test_frozen` is report only:** `make evaluate` applies the stored threshold and reports routed share, risk among accepted images (bound and bootstrap interval) and how many test errors review would catch. Test results never change the threshold or the rule.
- **Reason:** With so few errors, choosing the threshold by a risk target would be fitting to three or four images. A review budget is decided by the cost of human review, does not use labels, and so does not bias the risk estimate at the chosen point; the bound then says honestly how much risk the data can rule out.
- **Consequences / risks:**
  - The bound treats images as independent; near-copy groups in `val` make it slightly optimistic (hence the group-level figure).
  - `T` and the threshold are both set on `val`, which was also used for early stopping.
  - This does not measure errors on new defect types or under drift (OOD score, TASK-017) or the errors that actually ship (random audit sample, ADR-007).
- **Result (`make threshold`, `src/training/threshold.py`; `results/train/threshold.json`):** the risk check passed. 9 of 180 `val` images are routed; 3 of the 4 `val` errors are among them and 1 is auto-accepted, so the observed risk is 1 of 171 and its one-sided 95% upper bound is 2.7% (3.0% counting each near-copy group once), below `max_risk_ucb`. The file states what this does not show, including that a 1% risk cannot be shown with this many images.
- **Test, report only (`results/eval/efficientnet_b0/test_metrics.json`):** at the stored threshold, 8 of 360 `test_frozen` images are routed (2.2%, below the 5% budget set on `val`), 1 of the 3 test errors is caught, and the upper bound on risk among the 352 accepted images is 1.8%. The routed share on new images can differ from the budget; the threshold is not changed because of this.

## ADR-024 — The first `@champion` is set without a champion/challenger test
- **Context:** ADR-005 promotes a challenger only if it beats the current `@champion` on the same `test_frozen` images. When the first model is registered there is no champion to compare against. The model choice was made on `val` (ADR-022), and calibration (ADR-006), the review threshold (ADR-023) and the test report (TASK-011) exist for it.
- **Decision (written and committed before the registry is changed):**
  - `make champion` (`src/tracking/promote.py`) sets `@champion` only when no version holds it yet. If the same version already holds it, nothing changes. If another version holds it, it refuses: moving `@champion` later is the promotion gate's job (ADR-005, Phase 3).
  - **Checks before the registry is touched:** the ADR-022 choice is settled for the current split with the main model first; `calibration.json` and `threshold.json` match the checkpoint and the split; the test report used the same checkpoint, temperature and threshold on a clean tree; the registry version is found by the run id in `mlflow_run.json` (never by a typed version number), is READY, and its training run matches `val_metrics.json`.
  - **Evidence linked on the version (text only, no images):** tags with the split, checkpoint hash, `temperature`, `review_threshold`, the SHA-256 of `calibration.json`, `threshold.json` and `test_metrics.json`, the results commit, the decision records, and the artifact paths of those files in the run; a description with links to the MLflow run and to the results files on GitHub at that commit. Serving (TASK-015/018) reads `temperature` and `review_threshold` from the `@champion` version's tags and checks them against those files.
  - Test results are attached as evidence, never as a condition: nothing on `test_frozen` decides whether this version becomes champion.
  - Dry run by default; `--apply` writes. Each applied decision is appended to `results/promotion_log.json`, which the promotion gate extends later.
- **Reason:** The first champion is a starting point, not a promotion; comparing it with nothing would be a fake gate. Tying the alias to hashed evidence makes it clear later what exactly was put into service.
- **Consequences:** Version 1 (leaky split, ADR-021) never holds an alias. The first real ADR-005 decision is the first retrained challenger in Phase 3.
- **Result (`make champion ARGS=--apply`, 2026-11-08):** all checks passed; `@champion` now points to `steel-defect-classifier` version 2 (MLflow run `6895c4fd…`), whose tags hold `temperature`, `review_threshold` and the SHA-256 of the three evidence files, with links pinned to the results commit. The decision is the first entry of `results/promotion_log.json`. Version 1 holds no alias.

## ADR-025 — ONNX outputs raw logits; temperature and review threshold are applied in serving
- **Context:** TASK-015 exports the `@champion` model to ONNX for serving with ONNX Runtime (no PyTorch in the serving image). The earlier plan said "ONNX export with temperature applied". Since ADR-024 the `@champion` version carries `temperature` and `review_threshold` as tags, tied by hash to the files they came from.
- **Decision:**
  - The ONNX graph has one input `input` `[batch, 3, H, W]` (preprocessed by the shared `src/serving/preprocess.py`, dynamic batch axis) and one output `logits` `[batch, n_classes]`: raw logits, no softmax, no temperature.
  - Serving computes `softmax(logits / T)` and routes to review if the top calibrated confidence is below the threshold, reading `T` and the threshold from the tags of the `@champion` version it loads (ADR-024).
  - The model is found through the registry alias; the local checkpoint is used only if its SHA-256 equals the version's `checkpoint_sha256` tag. The export uses a synthetic example input (no NEU pixels).
  - Parity on `val` (never `test_frozen`) must hold before the file is kept: every logit within `|onnx − torch| ≤ atol + rtol · |torch|` (`export.parity.atol`, `export.parity.rtol`, like `numpy.allclose`), identical predicted classes, and identical review routing at the champion's `T` and threshold, also for several batch sizes. Evidence: `results/onnx/export_report.json`.
  - **Tolerance change (owner decision, 2026-11-08, after the first run):** the rule was first an absolute limit of 1e-4 on every logit. The first export failed it with a largest gap of 1.30e-4 while classes and review routing matched on all 180 `val` images, and nothing was kept. Cause: the trained model's logits are large (about −21 to +35), and ONNX Runtime sums the same float32 numbers in a different order than PyTorch over about 80 layers, which gives relative errors of about 1e-5 (turning off ONNX Runtime graph optimisations did not reduce it). An absolute limit therefore asks for more precision as logits grow. The limit is now `atol + rtol · |logit|` with `atol = rtol = 1e-4`; predicted classes and review routing must still match exactly, so the decisions serving makes are held to the same strict standard.
  - The ONNX file carries metadata (registry version, checkpoint hash, class order, preprocessing) and is logged to the champion's MLflow run; the version gets `onnx_sha256` and `onnx_artifact` tags so serving can verify the file it downloads.
- **Reason:** One source of truth for `T` and the threshold (the registry tags) instead of a second copy baked into the graph; a new calibration or threshold does not require a new ONNX file, and INT8 quantisation (TASK-016) works on the plain classifier graph.
- **Consequences:** Serving must never use the ONNX output as probabilities directly; a test in TASK-018 must check that `T` is applied. Supersedes the "with temperature applied" wording of TASK-015.
- **Result (`make export-onnx ARGS=--log-mlflow`, `results/onnx/export_report.json`):** version 2 (`@champion`) exported to one 16.5 MB opset-18 file. On the 180 `val` images, the largest logit gap is 1.30e-4 and the worst ratio to the tolerance is 0.59; predicted classes and review routing match on 180 of 180; batch sizes 1, 7 and 32 pass. The file and the report are in the champion's MLflow run under `onnx/`, and the version carries `onnx_sha256` and `onnx_artifact`.

## ADR-026 — INT8 serves only if it passes a rule fixed on `val` beforehand; FP32 is always the fallback
- **Context:** ADR-002 chose static INT8 QDQ quantisation with ONNX Runtime (calibration images from `train_v1`, `quant_pre_process` first), keeping FP32 as a fallback. TASK-015 produced the FP32 ONNX of `@champion` (ADR-025). EfficientNet is known to be sensitive to post-training INT8 (SiLU activations, depthwise convolutions, squeeze-excite).
- **Decision (written and committed before quantising):**
  - **Input:** the FP32 file of TASK-015, used only if its SHA-256 matches `results/onnx/export_report.json` and the `@champion` version's `onnx_sha256` tag.
  - **One fixed setting, no search:** `quant_pre_process`, then `quantize_static` in QDQ format with per-channel INT8 weights, INT8 activations, MinMax calibration and no `reduce_range`; calibration on all `train_v1` images through the shared preprocessing without augmentation (never `val` or `test_frozen`). The settings are in `params.yaml` → `quantize`.
  - **Rule, on `val` only (never `test_frozen`), INT8 against FP32 ONNX:** INT8 becomes the serving default only if every class F1 drops by at most `quantize.rule.max_class_f1_drop` (0.02), macro-F1 drops by at most `quantize.rule.max_macro_f1_drop` (0.01), and INT8 still passes the ADR-023 risk check at the stored review threshold with the champion's `T` (one-sided upper bound on risk among accepted images ≤ `threshold.max_risk_ucb`).
  - **If the rule fails:** FP32 stays the serving model, the INT8 file and the report are still logged as evidence, the result is recorded under this ADR, and work continues. No other setting is tried automatically.
  - **Outputs:** `models/onnx/<model>-v<version>.int8.onnx` (git-ignored, metadata copied from FP32 plus precision and calibration set) and `results/onnx/quantization_report.json` (per-class precision/recall/F1 for both, class and routing agreement, logit changes, ECE with `T`, routing and risk at the threshold, size and a rough CPU latency). With `--log-mlflow`, both go to the champion's run under `onnx/`, and the version gets `onnx_int8_sha256`, `onnx_int8_artifact` and `serving_precision` (`int8` or `fp32`). Serving (TASK-018) loads the file named by `serving_precision` and falls back to FP32.
- **Reason:** A rule fixed in advance stops us from accepting INT8 because the numbers look close enough after the fact. With 180 `val` images it can only catch large damage, so passing means "no visible loss on val", not "equal to FP32".
- **Consequences:** The latency figure here is one laptop CPU and is rough; the load benchmark is TASK-022. `T` is not refitted for INT8: the champion has one `T` (ADR-006, ADR-024).
- **Result (`make quantize ARGS=--log-mlflow`, `results/onnx/quantization_report.json`): INT8 fails the rule; FP32 serves.** On the 180 `val` images, macro-F1 drops from 0.978 (FP32) to 0.426 (INT8); the worst class F1 drop is 1.0 (`rolled-in_scale` is never predicted); INT8 agrees with FP32 on 86 of 180 classes, and its risk upper bound at the review threshold is 0.496. All three checks fail, so `serving_precision` is `fp32`. The INT8 file (5.2 MB vs 16.5 MB) and the report are logged in the champion's run as evidence. It is also slower on this CPU (median 108.6 ms vs 37.3 ms at batch 1). A diagnosis without new settings ruled out a pipeline error: `quant_pre_process` alone leaves the FP32 logits unchanged on `val`, so the loss comes from quantisation itself. The quantised graph keeps 65 Sigmoid·Mul (SiLU) pairs and the squeeze-excite blocks in float between hundreds of quantise/dequantise steps; the error builds up through the network (logit range shrinks from about [−21, 35] to [−12, 15]) and the extra conversions cost time. This matches the known weakness of post-training INT8 on EfficientNet. Per this ADR no other setting was tried; options for later (owner's call) are excluding sensitive layers, another calibration method, or quantisation-aware training.

## ADR-027 — Energy OOD threshold from a fixed flag budget on `val`; route on low confidence OR OOD
- **Context:** ADR-006 adds an energy OOD score and routes to review if confidence is low OR the OOD score flags the image. The classifier always picks one of its classes, even for an input that is not steel. The serving model is the FP32 ONNX file named by the `@champion` version's `serving_precision` tag (ADR-026).
- **Decision (written and committed before any energy is computed):**
  - **Score:** `energy = −T · logsumexp(logits / T)` on the serving model's raw logits, with `T` from the `@champion` `temperature` tag (ADR-024, ADR-025). Higher energy means less familiar. One shared, torch-free implementation, `src/serving/scoring.py`, is used offline and in serving, together with the calibrated confidence and the routing decision (reasons `low_confidence`, `ood`).
  - **Threshold rule (no labels):** flag the `ceil(ood.id_flag_budget × n)` `val` images with the highest energy (`id_flag_budget` = 0.02: 4 of 180); the threshold is the midpoint between the lowest flagged and the highest unflagged energy; an image is flagged if its energy is above it. Never `test_frozen`.
  - **Routing:** review if calibrated confidence < `review_threshold` (ADR-023) OR energy > the OOD threshold. On `val` the report gives routed counts for each reason, their overlap, the combined routed share, and the ADR-023 risk bound recomputed for the images still accepted.
  - **Evaluation on synthetic non-steel images:** procedurally generated from a seed in `params.yaml` (no download, no NEU pixels), at NEU's format, through the shared preprocessing; per family and pooled: share flagged with a Clopper–Pearson lower bound, AUROC of energy against `val`, false-positive rate at 95% detection. The AUROC of softmax confidence and of energy at `T = 1` are reported for reference only; they choose nothing.
  - **No stop condition:** whatever the synthetic results are, they are recorded under this ADR and work continues.
  - **Outputs:** `results/train/ood.json` (threshold, `T`, rule, hashes of the split file, the serving ONNX file and `calibration.json`, `val` and synthetic results, what can and cannot be claimed) and `ood_score_histogram.png`; with `--log-mlflow`, both go to the champion's run and the version gets the `ood_energy_threshold` tag that serving reads. `make evaluate` reports the OOD flag share and the combined routing on `test_frozen` (report only; nothing is chosen there).
- **Reason:** A flag budget fixes the extra review load and uses no labels; the threshold cannot be tuned to the synthetic set, which is only used to evaluate.
- **Consequences / limits:** Synthetic images are far from steel and easy to detect; doing well on them says nothing about unseen steel defect types (near-OOD), which X-SDD will test later (ADR-010). With 180 `val` images, a 2% budget is set by a handful of extreme images, so the threshold is noisy.
- **Result (`make ood ARGS=--log-mlflow`, `results/train/ood.json`):** energy threshold −6.1446 on the FP32 serving model (95% grouped-bootstrap interval of the 98% energy quantile about −7.51 to −5.30: the threshold is noisy, as expected). On `val`, 4 of 180 images are flagged (by construction), 9 are routed for low confidence, 1 for both reasons, so 12 (6.7%) go to review; none of the 4 `val` errors is flagged as OOD, and the risk upper bound among the 168 images still accepted is 2.8% (2.7% with low confidence alone). On 400 synthetic non-steel images (8 families × 50), 322 are flagged as OOD (at least 77.0% with 95% confidence) and 355 go to review for either reason; AUROC of energy against `val` is 0.972 (softmax confidence 0.966, energy at `T = 1` 0.973, reference only). The weakest family is blurred blobs: 12 of 50 flagged, 35 routed for either reason, and the model calls most of them `inclusion`. A few Gaussian-noise images get very low energy (the model is very confident on them). `test_frozen` (report only, `results/eval/efficientnet_b0/test_metrics.json`): 3 of 360 in-distribution images flagged (false alarms; one-sided 95% upper bound 2.1%), 10 routed for either reason (2.8%), risk upper bound among accepted 1.8%. These synthetic results say nothing about unseen steel defect types; X-SDD tests that later.

## ADR-028 — The API serves a verified bundle built from `@champion` at deploy time; no registry access at runtime
- **Context:** TASK-018 adds the FastAPI service. It needs the ONNX file named by the `@champion` version's `serving_precision` tag (ADR-026) and the values serving applies: `temperature`, `review_threshold` (ADR-024, ADR-025) and `ood_energy_threshold` (ADR-027). Reading them live from the registry at startup would put DagsHub credentials and `mlflow` into the API container and make every start depend on DagsHub being up.
- **Decision (written and committed before the code):**
  - **Build step, `make serving-bundle`** (`src/tracking/serving_bundle.py`, laptop, read-only on the registry): resolve `@champion` by alias; require the tags `temperature`, `review_threshold`, `ood_energy_threshold`, `serving_precision` and the hash and artifact tags of the file it names (`onnx_sha256`/`onnx_artifact` for `fp32`, `onnx_int8_sha256`/`onnx_int8_artifact` for `int8`); download that ONNX file from the champion's run and check its SHA-256 against the tag; write `models/serving/` (git-ignored): the ONNX file and `manifest.json` with the registered model, version, alias, run id, precision, the three values, the class order, the preprocessing settings, the ONNX SHA-256 and the time it was built. Nothing in the registry is changed.
  - **API startup** (`src/serving/bundle.py`): read the manifest, check the ONNX file's SHA-256, and check that the class order, preprocessing settings and registry version in the ONNX metadata match the manifest and `params.yaml` → `preprocess`; refuse to start on any mismatch or if `API_KEY` is unset. `/health` returns 503 until the model is loaded.
  - **Serving uses only shared code:** `preprocess.load_image` → `to_model_input`, `OnnxClassifier`, and `scoring.score` with the manifest's `T` and thresholds. The response gives the prediction, calibrated confidence `softmax(logits / T)`, energy, `routed_to_review` with reasons (`low_confidence`, `ood`), the model version and a `request_id`.
  - **Security and input checks** (limits in `params.yaml` → `serving`): `X-API-Key` on `/predict` and `/feedback`, compared in constant time with env `API_KEY` (401 missing, 403 wrong); 413 above `max_upload_bytes`; 415 if the decoded format is not in `allowed_formats` (the bytes decide, not the client's content type); 422 if it does not decode, is smaller than `min_side` or larger than `max_pixels`, or is near-blank (greyscale pixel standard deviation below `near_blank_max_std`, set well below every NEU `train_v1`/`val` image and checked by a data test). Errors are JSON with `error`, `detail` and `request_id`. `/health` and `/metrics` (Prometheus format) are open; access to them is limited at the ingress later.
  - **Logs:** one JSON line per request with `request_id`, the image SHA-256 and size, the result and latency; never the image bytes, the file name or the key. `/feedback` is a stub (validates, stores nothing) until TASK-019/026.
- **Reason:** The API image needs no credentials and no network to start, the files it serves are tied by hash to the registry tags, and rollback is redeploying the previous image (ADR-012).
- **Consequences:** The values the API applies are a snapshot of the tags when the bundle was built: moving `@champion` or changing a tag needs a new bundle and a new image, which is the deploy step anyway. The manifest records the version so a stale bundle is visible in `/health` and in every response.
- **Result (`make serving-bundle`, `make serve`, 2026-11-08):** the bundle holds version 2 (`@champion`, FP32); the downloaded ONNX file's SHA-256 matched the `onnx_sha256` tag and the file passed the same checks the API runs at startup. The API started from it with no registry access, and a real `/predict` call on one `val` image returned the same class and calibrated confidence as the committed `results/train/val_predictions.csv`. A `needs_data`/`needs_model` test sends all `val` images through the API and checks every predicted class, every calibrated confidence (to the CSV's 4 decimals) and the routing counts in `results/train/ood.json`. The near-blank limit is 2.0; a data test checks that no NEU `train_v1` or `val` image is within a factor of 2 of it.

## ADR-029 — Prediction log: one row per served prediction, no image bytes, fail closed
- **Context:** TASK-019 builds the prediction log on SQLAlchemy reading `DATABASE_URL` (ADR-003, ADR-014): SQLite on the laptop and in tests, Postgres on Azure later (TASK-037). The review queue (TASK-024), feedback (TASK-026) and drift reports (TASK-033/034) read from it. The serving ONNX model outputs only logits (ADR-025).
- **Decision (written and committed before the code):**
  - **One row per successful `/predict`** in table `predictions`: `request_id` (unique), `created_at` (UTC), `image_sha256`; `prediction`, `calibrated_confidence`, `energy`, `routed_to_review`, `reason_low_confidence`, `reason_ood`; `model_name`, `model_version`, `model_precision`; `latency_ms`, `inference_ms`; `image_bytes` (size), `image_width`, `image_height`, `image_format`; drift features `brightness` (mean grey level), `contrast` (grey standard deviation), `sharpness` (variance of the 3×3 Laplacian), and `logits` (the raw logits as a JSON list).
  - **Never stored:** image bytes, pixels, file names or the API key. A test checks the schema has no binary column.
  - **Features** are computed by one numpy-only function, `src/serving/features.py`, on the decoded greyscale image at its original size (width and height are stored with them); the drift generators and Evidently reports reuse it.
  - **No embedding yet:** the raw logits stand in as a small model-side view. A true embedding needs a second ONNX output (new file, parity re-run, new tags) and is decided before TASK-034.
  - **Rejected requests** (4xx) are not stored; metrics and JSON logs count them.
  - **One database layer, `src/db/`:** the engine from `DATABASE_URL`, the table definitions and the insert; no backend-specific code outside it. Tables are created at startup if missing; Alembic migrations come at TASK-037, before Postgres holds real data.
  - **`DATABASE_URL` is required:** the API refuses to start without it, as with `API_KEY`.
  - **Fail closed:** if the row cannot be written, `/predict` returns 503 `log_unavailable` (no prediction is returned) and a metric counts the failure.
- **Reason:** A prediction that is served but not logged breaks the loop: a routed image would never reach the review queue and could not be audited. Storing hashes and summary features keeps the log useful for drift without holding NEU pixels (ADR-015).
- **Consequences:** The API depends on the database being up. Sharpness depends on resolution, so it is only comparable between images of the same size. Any secret or personal data must stay out of this table.
- **Result (`make serve` with `DATABASE_URL` set to a local SQLite file, 2026-11-08):** six `val` images sent through `/predict` wrote six rows; every row's class and calibrated confidence matches `results/train/val_predictions.csv`, and the two least confident images were stored as routed (one for both reasons, one for low confidence). The table has no binary column (SQLite types: boolean, datetime, float, integer, JSON, short strings); the database file is git-ignored. A `needs_data`/`needs_model` test sends all `val` images through the API and checks one row per prediction.
- **Docker (TASK-021, `make docker-smoke`):** the API image (`Dockerfile`) installs only the serve group from the lockfile, copies only `src/` serving and database code, `configs/params.yaml` and the bundle (the build context is an allowlist, so no NEU data, `.env` or results can enter), runs as a non-root user, and labels the image with the model version and ONNX SHA-256. `docker-compose.yml` passes only `API_KEY` (required) and a fixed `DATABASE_URL` on a named volume, binds to 127.0.0.1, and runs read-only with all capabilities dropped. The smoke check sends only the synthetic golden image: health check healthy, `/predict` equals `tests/golden/expected.json`, 401 without key, non-root, no torch or mlflow inside, read-only filesystem, the logged row survives a restart, the key never appears in the logs.

## ADR-030 — Random audit: a salted hash of the request id picks 2% of auto-accepted predictions
- **Context:** ADR-007 sends a fixed share of auto-accepted predictions to review, so the errors that actually ship can be measured. Routing so far uses low confidence (ADR-023) and OOD (ADR-027). The review queue (TASK-024) and the promotion loop read the routing from the prediction log (ADR-029).
- **Decision (owner decisions of 2026-11-11, written and committed before the code and any results):**
  - **Rate:** `serving.audit_rate` = 0.02 in `params.yaml`.
  - **Only auto-accepted predictions can be audited:** a prediction already routed for `low_confidence` or `ood` is never also marked `audit`, so audit counts measure only what would otherwise ship.
  - **Selection:** `u = first 8 bytes of SHA-256(request_id + AUDIT_SALT) / 2^64`; the prediction is audited if `u < audit_rate`. The request id is a random UUID made by the server, so selection is deterministic for a given request and independent of the image, the confidence and the energy; with the salt secret, a client cannot predict or steer it. One function (`src/serving/routing.py`) makes the decision.
  - **Salt:** from env `AUDIT_SALT`, never logged, never committed. If `audit_rate > 0` and `AUDIT_SALT` is missing, the API refuses to start. `.env.example` has an empty `AUDIT_SALT=`; compose passes it to the API container. Tests and smoke checks use a fixed, non-secret test salt.
  - **Recording:** the response and the log row carry reason `audit` (`routed_to_review` true, new column `reason_audit`); audited rows count in the review-queue gauge and the `routed_reason` metric. Existing local databases get the new column added with value false (no earlier prediction was audited); any other schema difference stops the API (Alembic migrations stay at TASK-037).
- **Reason:** A salted hash gives a fair, reproducible sample (the same request id always gets the same decision, which tests and audits can check) without storing random state, and the salt stops anyone from gaming which images get audited.
- **Consequences:** With 2% of auto-accepted traffic, few audits accumulate on small streams, so audit error-rate estimates will have wide intervals; the loop reports them with exact binomial bounds. Changing the salt or the rate changes which future requests are audited, never past rows.

## ADR-031 — Review by image hash, a feedback table, blind audit cards and a seeded simulated reviewer
- **Context:** TASK-024 builds the review UI and TASK-025 the simulated reviewer. The prediction log stores only the image hash, never the bytes (ADR-029, CLAUDE.md), so the review card cannot get the image from the database. NEU images must never be redistributed (ADR-015). The review queue is fed by low confidence, OOD and the random audit (ADR-023, ADR-027, ADR-030). The reviewer and the stream are simulated in this project.
- **Decision (owner decisions of 2026-11-11, written and committed before the code):**
  - **Images by hash lookup:** the review side looks the image up in the operator's own image archive by SHA-256. Locally the archive is `data/raw/`, indexed in memory from the split file (hash → path, class, split; a hash shared by files of different classes is treated as unknown). No image is copied, stored in the database, logged, or put in a Docker image. A card whose hash is not in the archive shows the stored features instead. The deployed review UI would therefore have no NEU images; TASK-040 will likely deploy only the API and keep the review UI local (decided there).
  - **Feedback table** (same database layer, ADR-003/029): one final label per prediction: `request_id` (unique), `label` (one of the model's classes or `unknown`), `reviewer_kind` (`human` or `simulated`), `reviewer_id`, `created_at`; for simulated labels also `simulated_error_rate` and `simulated_flipped`, kept only to measure reviewer noise and never read by retraining or the promotion gate. The UI and the simulated reviewer write through `src/db`; the API's `/feedback` stays a stub until TASK-026. `unknown` labels are counted for audit and OOD evaluation and excluded from retraining (TASK-026).
  - **Queue:** routed predictions without a feedback row, oldest first, filterable by reason. The `defect_api_review_queue_size` gauge becomes "routed and not yet labelled".
  - **Review UI** (Streamlit, local, `127.0.0.1`): needs `REVIEWER_PASSWORD` from env (constant-time compare). Each card shows the image, the routing reasons and label buttons (the classes plus `unknown`); low-confidence and OOD cards also show the prediction, the calibrated confidence and the energy with its threshold. **Blind audit:** on an audit card the prediction, the confidence and the energy are hidden until the reviewer has submitted a label, so the label is not anchored on the model's answer; they are shown after submitting. A banner states that the stream and any automated reviewer are simulated.
  - **Simulated reviewer:** for each queued prediction (oldest first, up to a limit) it takes the true NEU label of the archive file with that hash and, with probability `review.simulated.error_rate`, replaces it with a class drawn uniformly from the other classes. The draw uses a random generator seeded with `review.simulated.seed` and the request id, so a rerun gives the same labels in any order. It never labels an image whose hash is in `test_frozen` (skipped and counted: a test image labelled in the loop would leak into retraining; TASK-026's export repeats this guard) and skips images without a unique ground truth. Its output says it is simulated.
  - **Stream:** `make stream` sends `production_stream` images (and no other split) to the API in a seeded order, so the queue fills the way the loop (TASK-028/030) will use it.
- **Reason:** Looking images up by hash keeps "hash only, never bytes" and matches a plant, where images stay in the camera archive. Blind audits keep the audit an unbiased check of the model. A seeded, order-independent simulator makes the loop experiments (TASK-030, 0/5/15% reviewer error) reproducible.
- **Consequences:** The review UI only works where the archive is; images not in it can be labelled from features only (or skipped). Human labels entered in the UI are real labels on simulated traffic.

## ADR-032 — Feedback export: production_stream labels only, as a path/label CSV versioned with DVC
- **Context:** TASK-026. Retraining (TASK-028) uses `train_v1 + all feedback`. Feedback lives in the prediction log's feedback table (ADR-031), keyed by request id; the image itself is found by hash in the local archive (`data/raw/`). The reviewer and the stream are simulated. NEU images must never be redistributed (ADR-015), and `test_frozen` must never reach training. The local log (`data/predictions.db`) holds label rows from manual UI tests and from TASK-019 traffic on `val` images.
- **Decision (owner decisions of 2026-11-11, written and committed before the code):**
  - **What is exported:** one row per labelled image whose hash is in the archive under split `production_stream`. Excluded, each counted in the report: `test_frozen`, `val` and any other split; hashes not in the archive or without a unique label (ADR-031); label `unknown`. Predictions without a feedback row are not read. A feedback row is the final label (one per request id, ADR-031).
  - **Repeated images:** if the same image was labelled in several requests, it is exported once when all labels agree (the latest request id is kept) and excluded (counted as conflicting) when they disagree. Training never sees two labels for one image.
  - **Columns read:** request id, label and reviewer kind from the feedback table, and the image hash from the prediction. `simulated_flipped` and `simulated_error_rate` are never selected: they exist only to measure reviewer noise (ADR-031). A test checks the SQL the export runs.
  - **Output:** `data/feedback/feedback.csv` with columns `path` (relative to `data/raw`, POSIX), `label`, `reviewer_kind`, `request_id`, sorted by path. No image bytes are copied: retraining reads the image from `data/raw/` by path.
  - **Versioning:** `make feedback-export` writes the CSV, then `dvc add data/feedback`. Each export overwrites the CSV, so each committed `data/feedback.dvc` is one version; older versions come back with `git checkout <commit> data/feedback.dvc && dvc checkout`. `make push` copies it to the private DVC remote. The CSV is not committed to git.
  - **Report:** `results/feedback/export_report.json` (committed): rows, per-class counts (all classes, zeros included), per reviewer kind, every exclusion count, the CSV's MD5 (the hash DVC records) and the split file used. No paths, labels per image or pixels. It states that the reviewer and the stream are simulated.
  - **The simulated reviewer** labels only images whose hash is in `production_stream` (ADR-031 already skipped `test_frozen`); images from other splits are skipped and counted. The export repeats every check, so a label written another way (the UI) still cannot leak.
  - **Local log reset:** `make db-reset` moves the local SQLite log (and its `-wal`/`-shm`/`-journal` files) to `data/archive/predictions-<UTC time>.db` (git-ignored) and creates an empty schema. It refuses any non-SQLite `DATABASE_URL`. The UI-test labels of 2026-11-11 are archived this way and never reach an export.
  - **Docker log vs local log:** the review UI, the simulated reviewer and the export read the SQLite file from `DATABASE_URL` (locally `data/predictions.db`, written by `make serve`). The Docker API writes to the named volume `predictions`, which they do not read. Bind-mounting one SQLite file into both the Docker Desktop VM and Windows is not safe (file locks do not cross that boundary), so the local loop runs against `make serve`. In the cloud both sides share Postgres through `DATABASE_URL` (TASK-037/038). Documented in the Makefile help and the README.
  - **Out of scope:** the API's `/feedback` endpoint stays a validated stub; labels are written by the UI and the simulated reviewer through `src/db`.
- **Reason:** A path list keeps the export small, free of pixels and reviewable, and the archive stays the only copy of the images. Re-checking the split at export time makes the export the last guard before training, whatever wrote the label.
- **Consequences:** An export is only as good as the local archive: on a machine without `data/raw/` every row is excluded as "not in archive". The export is cumulative (all feedback so far), so TASK-028 always trains on the latest version.

## ADR-033 — Promotion gate: non-inferiority on `val`, four checks, decision files and alias moves
- **Context:** TASK-027. ADR-005 planned a superiority gate on `test_frozen` (McNemar significant or a minimum gain, plus recall, ECE and latency limits). The loop retrains from the base checkpoint on `train_v1 + feedback` (ADR-019, ADR-032) and registers the result as `@challenger`. The champion's val accuracy is 0.9778 on 180 images (`results/train/val_metrics.json`), i.e. 4 errors. A challenger can therefore be right on at most 4 val images where the champion is wrong. The exact two-sided McNemar test needs at least 6 discordant images, all in one direction, to reach p < 0.05. So a significant improvement cannot be shown on this set, whatever the challenger does. `test_frozen` stays untouched by the loop, so that the final reports stay unbiased across many loop cycles.
- **Decision (owner decisions of 2026-11-12, written and committed before the code; supersedes ADR-005's gate set and rule):**
  - **Gate set: `val` only.** Never `test_frozen`, never feedback rows: the gate refuses a split named `test_frozen`, and refuses to run if any gate image path is in the current feedback export. Champion and challenger are both scored from their checkpoints (verified against each version's `checkpoint_sha256` tag) on the same images in the same order.
  - **Non-inferiority, not superiority.** Promote only if all four checks pass (thresholds in `params.yaml` → `promotion.gate`):
    - a) challenger macro-F1 ≥ champion macro-F1 − `max_macro_f1_drop` (0.01);
    - b) no class recall drops by more than `max_class_recall_drop` (0.05). With 30 val images per class one image is 1/30 of recall, so this allows at most one more miss per class;
    - c) exact McNemar on the paired table: **block** if p < `mcnemar_alpha` (0.05) **and** the champion is right on more images where the challenger is wrong than the other way round (significantly worse). A significant difference in the challenger's favour never blocks;
    - d) the challenger's FP32 ONNX export passes the ADR-025 parity check on `val` (logits within tolerance, identical classes and routing at the challenger's own T and threshold, every batch size). This reuses `src/training/export_onnx.py`, generalised from `@champion` to any alias.
  - ECE and latency are not gate checks: every model is recalibrated on `val` (ADR-006), and the architecture and serving path stay the same. Both are reported elsewhere (calibration report, docker smoke).
  - **Decision files.** `make gate` is a dry run: it scores, prints and writes nothing. `make gate ARGS=--apply` writes `results/promotion/gate_<challenger version>.json`, then changes the registry, then appends an entry to `results/promotion_log.json`. The gate file holds both versions, each check's value, limit, pass/fail and reason, both models' metrics, the paired table and p-value, the split hash and the git commit. The log entry holds the decision, the versions, the previous champion, and the gate file with its SHA-256. Rejections are recorded too.
  - **Promote:** tag the challenger version (`gate_decision`, `gate_file_sha256`, `previous_champion_version`), move `@champion` to it, and remove `@challenger` (it now serves). The previous champion version is in the tag and the log, for the rollback script (TASK-029). The ONNX file and its tags are logged to the challenger's run during the gate (`--apply`), so `make serving-bundle` can build the new champion's bundle.
  - **Reject:** `@champion` does not move and `@challenger` stays on the rejected version; the version is tagged `gate_decision=reject`.
  - The gate refuses (an error, not a decision) if either alias is missing, if both are on the same version, if a version lacks the tags `model`, `checkpoint_sha256`, `temperature`, `review_threshold`, or if a checkpoint does not match its tag.
- **Reason:** With a near-perfect champion and 180 val images, a superiority rule would never promote, and the loop could never ship a retrained model. Non-inferiority with per-class limits asks the useful question: did retraining on feedback break anything? Keeping `test_frozen` out of every loop decision means it is still an unbiased final test after many cycles.
- **Consequences:** `val` is not untouched data. The challenger used it for early stopping, calibration and its threshold, and the champion's model choice used it too (ADR-022). Gate metrics are therefore optimistic for both models, possibly more for the challenger. The margins are small and the final comparison on `test_frozen` is report-only (TASK-030). A challenger that is merely "as good" can replace the champion; this is intended, because it has seen more production data. The README states that the gate is non-inferiority.

## ADR-034 — Retrain flow: plain Python, feedback as a private Kaggle dataset version, checkpoints in DVC
- **Context:** TASK-028 connects the loop: feedback export (ADR-032) → training on Kaggle GPU (ADR-019, credential-free per ADR-020) → registration as `@challenger` → promotion gate (ADR-033). The feedback CSV is DVC-tracked and git-ignored, so a Kaggle job that clones the repo cannot see it. After the first promotion, the champion's checkpoint is no longer `models/baseline/best.pt`, and the gate needs both checkpoints.
- **Decision (owner decisions of 2026-11-12, written and committed before the code):**
  - **Plain Python flow, no Prefect:** `make retrain` (`src/pipelines/retrain.py`) runs fixed steps in order and saves progress in `build/retrain/state.json` (git-ignored). A rerun continues from the first unfinished step; `--fresh` starts over. The steps:
    - preflight → feedback dataset → Kaggle push → wait → download and verify → register;
    - calibrate + threshold → tag + `@challenger` → gate record → cycle record.
  - **Preflight:** the tree is clean and the commit is pushed (Kaggle clones it). The near-copy report is clean (ADR-021). `@champion` exists. The committed export report matches `data/feedback/feedback.csv` (MD5). The export has at least `retrain.min_feedback_rows` rows; the value is set from the first real cycle's yield.
  - **Feedback to Kaggle:** a separate PRIVATE dataset (`kaggle.feedback_dataset_slug`), with a new version per cycle. It holds only `feedback.csv` (paths and labels, no pixels). It gets the same checks as the image dataset: private per the API, anonymous access blocked, file list exactly `feedback.csv`. The kernel attaches both datasets. The job still holds no credentials: it receives the repo at a pinned commit, the two private datasets, and the feedback MD5 as a plain training argument.
  - **Training** (`src.training.train --feedback CSV --feedback-md5 MD5`): `train_v1` plus the feedback rows, from the ImageNet base weights (never the previous model, CLAUDE.md). The CSV must have the given MD5. Every feedback path must be a `production_stream` image of the split file, with a label among the classes and no duplicates; otherwise training stops. `val_metrics.json` records the feedback MD5, row count and per-class counts. The published run carries the tag `feedback_md5`.
  - **After download:** the run must come from the pushed commit, use the expected feedback MD5, and its checkpoint must match the recorded SHA-256. Otherwise the flow stops, so an older Kaggle output is never picked up.
  - **Registration:** `publish --register` (role `candidate`), with results in `results/retrain/<run_uid>/` (committed: no pixels). Calibration (ADR-006) and the review threshold (ADR-023) are fitted on `val` for the challenger with the same rules and logged to its run. Tags: `model`, `checkpoint_sha256`, `temperature`, `review_threshold`, `feedback_md5`, `feedback_rows`, `decision_record=ADR-034`. Then `@challenger` moves to the new version.
  - **Checkpoints:** `models/versions/v<version>/best.pt`, tracked with `dvc add models/versions` (`models/versions.dvc` committed; `.gitignore` gets an exception) and pushed to the private DVC remote. On the first cycle, the champion's checkpoint is copied there from `compare_val.models[0].checkpoint`, after checking it against the `@champion` tag. The gate's default champion checkpoint becomes `models/versions/v<champion version>/best.pt`.
  - **Gate record mode (amends ADR-033):** `gate --record` writes `results/promotion/gate_<challenger version>.json` with `applied: false`, and does not change the registry or the promotion log. The flow always runs the gate in record mode: `@champion` does not move until the rollback script exists (TASK-029); promoting is then a separate `make gate ARGS=--apply`.
  - **OOD threshold:** `src/training/ood.py` scores the `@champion` ONNX bundle and is not yet generalised. Challengers therefore get no `ood_energy_threshold` / ONNX tags, and `gate --apply` refuses to **promote** a version without them (it would not be servable). Generalising OOD for challengers is follow-up work before the first real promotion.
  - **Cycle record:** `results/retrain/cycle_<run_uid>.json`:
    - feedback rows, per class and MD5;
    - commit, Kaggle wall time (push → complete) and training time;
    - challenger version;
    - both models' val macro-F1 from the gate, the gate decision and the gate file.
    - It states that the reviewer and the stream are simulated.
- **Reason:** Plain Python keeps the flow testable with fakes and avoids a new dependency during the sprint. A separate small dataset keeps the image dataset (and its checks) unchanged and makes each cycle's labels an immutable Kaggle version, matched by MD5. DVC keeps every model version's weights recoverable for the gate and for rollback without putting weights on public MLflow artifacts.
- **Consequences:** The laptop must stay on while the flow waits for Kaggle; a rerun resumes. Each cycle adds one DVC version of `models/versions` (about 16 MB per checkpoint). Feedback comes only from routed images (low confidence, OOD, audit), so it is a biased, hard-example sample, not a random one. The challenger uses `val` for early stopping, calibration and the threshold, and the gate also scores on `val` (ADR-033 consequences).

## ADR-035 — Rollback, challenger serving readiness, and decision files that are never overwritten
- **Context:** TASK-029. A promotion (ADR-033) moves `@champion` and records the previous version (`previous_champion_version` tag, promotion log). The serving bundle (ADR-028) needs, on the champion version, `serving_precision`, the ONNX file tags of that precision, `temperature`, `review_threshold` and `ood_energy_threshold`. Challengers so far get only T and the review threshold (ADR-034); the OOD threshold (ADR-027) was computed for `@champion` only, so `gate --apply` refuses to promote them. The gate file name `gate_<version>.json` (ADR-033) collides when one version is gated more than once.
- **Decision (owner decisions of 2026-11-12, written and committed before the code):**
  - **Challenger serving readiness** (`make challenger-serving`, and a `serving` step of `make retrain` after `challenger`): for the version holding `@challenger`, with its kept checkpoint (`models/versions/v<v>/best.pt`) and results folder (`results/retrain/<train_run_uid>/`):
    - export the FP32 ONNX and run the ADR-025 parity check on `val` (the report goes into the results folder; the file and its `onnx_sha256` / `onnx_artifact` tags go to the version only if parity passes);
    - tag `serving_precision=fp32`. Challengers serve FP32 like v2: ADR-026 chose FP32, and INT8 is not re-evaluated per cycle;
    - compute the energy OOD threshold on `val` with exactly the champion's method and budget (ADR-027), on that ONNX file, at the challenger's own T. Write `ood.json` and the histogram into the results folder, and tag `ood_energy_threshold`.
    - `src/training/ood.py` and `export_onnx.py` take an alias and explicit files instead of being tied to `@champion`.
  - **Rollback** (`make rollback [VERSION=v]`, `src/tracking/rollback.py`): moves `@champion` to `VERSION`, or else to the current champion's `previous_champion_version` tag. It refuses, without changing anything:
    - if there is no target, or the target already holds `@champion`;
    - if the target's checkpoint is missing or does not match its `checkpoint_sha256` tag;
    - if the target lacks the ONNX parity result: the ONNX file tags of its `serving_precision`, which are only written after parity passed;
    - if it lacks `temperature`, `review_threshold` or `ood_energy_threshold`.
  - Rollback writes `results/promotion/rollback_<UTC timestamp>.json`: from/to versions, the checks, and the reason (`previous_champion_version` tag or explicit). Then it tags the target (`rolled_back_to_at`), moves the alias, and appends a `rollback` entry to `results/promotion_log.json`. It prints the next step (`make serving-bundle`, then `make docker-smoke` / deploy) and does not redeploy by itself. `@challenger` is not touched.
  - **Re-gating a version** (`make challenger VERSION=v`): points `@challenger` at a registered version that passes the same readiness checks and is not the champion. A version that was rolled back can then go through the gate again. It never moves `@champion`.
  - **Decision files are never overwritten:** if `gate_<v>.json` exists, the next gate file for that version is `gate_<v>.<k>.json` (k = 2, 3, …). Rollback files carry a timestamp.
  - **Gate apply and the ONNX file:** if the challenger already has its ONNX tags (readiness ran), `gate --apply` checks parity again but does not log a second ONNX file to the run.
  - **Golden outputs after a promotion:** `make golden ARGS="--val-predictions <results dir>/val_predictions.csv"` regenerates `tests/golden/expected.json` for the new champion. It refuses a predictions file whose `calibration.json` belongs to another checkpoint than the bundle's. `make docker-smoke` then checks the image against the new champion.
- **Reason:** A rollback is only safe if the target can actually be served and reproduced. Checking the same tags the serving bundle needs, plus the kept checkpoint, before moving the alias means a rollback can never leave `@champion` on a version that `make serving-bundle` would reject. Immutable decision files keep every promotion, rejection and rollback auditable.
- **Consequences:** Every challenger now costs one ONNX export and one OOD pass on CPU (minutes) before it can be promoted. A promoted challenger's OOD threshold comes from `val` like the champion's, with the same small-sample limits (ADR-027). The committed golden file follows the champion, so it changes with every promotion; the diff is reviewed in the PR.

## ADR-036 — Loop cycles: pilot plus disjoint stream chunks, 0% and 5% reviewer error, one fault-injection run, one report-only test
- **Context:** TASK-030. Cycle 1 (ADR-034) streamed all 1,110 `production_stream` images through v2; the simulated reviewer labelled the 67 that were routed, and v3 was trained on them and promoted (ADR-035 demo). No unused stream images are left, so later cycles cannot get brand-new traffic. The loop must also show that the gate (ADR-033) rejects a challenger trained on bad labels.
- **Decision (owner decisions of 2026-11-12, written and committed before the code):**
  - **Cycle 1 is the pilot** and is not re-run. It keeps its files (`results/retrain/cycle_e7f2….json`, `results/promotion/gate_3.json`).
  - **Disjoint chunks** (`make loop-chunks` → committed `data/splits/loop_chunks.csv`, paths only): every `production_stream` image in the seeded stream order (`review.stream.seed`). The 67 images labelled in the pilot are `pilot`; they are read from the pilot export, whose MD5 must match the cycle-1 record. The remaining 1,043, in stream order, are cut into `cycle2` (first 520) and `cycle3` (next 523); the sizes are in `loop.chunks`. **These are not brand-new traffic:** v2 already predicted every one of them in the pilot. None of them was labelled or trained on. Every write-up says so.
  - **Cycles 2 and 3 run the real flow:**
    1. rebuild the serving bundle for the current `@champion` and restart `make serve`, so the current champion routes the chunk;
    2. `make stream ARGS="--chunk cycleN --out …"` sends exactly that chunk (the pilot chunk is refused);
    3. `make simulate-review` labels the whole queue, which holds only this chunk's routed images. Cycle 2 uses error rate 0.0 and cycle 3 uses 0.05. The rate applies to the new labels only; the earlier labels stay as they are;
    4. `make feedback-export` writes all feedback so far (pilot plus chunks), commit;
    5. `make retrain ARGS=--fresh` trains from the base checkpoint on `train_v1` plus all feedback (gate in record mode);
    6. `make gate ARGS="--challenger-checkpoint … --apply"`.
  - A rejected challenger is reported as rejected. No threshold, seed, label or parameter is changed to make one pass.
  - **Fault injection, clearly labelled** (one run, after cycle 3):
    - A *broken reviewer* (`make fault-export`) moves **every** label of the latest real export (pilot plus both chunks) to the next class in sorted class order (cyclic). This is the kind of bug a reordered button row would cause.
    - It works on a copy: `data/feedback_fault/feedback.csv` (git-ignored, not DVC-tracked) and `results/loop/fault_injection/export_report.json` (source MD5, rule, counts).
    - `make retrain ARGS="--fresh --fault-injection"` runs the same flow. It uses its own state folder and its own private Kaggle dataset (`neu-feedback-fault-private`). It tags the version `fault_injection=true` and runs the gate in record mode only.
    - The gate refuses `--apply` for any version tagged `fault_injection=true`. When the run ends, `@challenger` goes back to its previous holder (or is removed).
    - The shifted labels never enter the real feedback history: the prediction log, the `data/feedback` DVC versions and `neu-feedback-private`.
    - The expected verdict is REJECT. The verdict and the failed checks are reported whatever they are; a pass would be a finding about the gate.
  - **Loop records:** the stream and reviewer summaries of each cycle go to `results/loop/cycle_<n>/`. The reviewer summary may count `simulated_flipped` for reporting; the export still never reads it (ADR-032). `make loop-record` then adds a copy of that cycle's export report and `cycle.json` (pointers to the retrain record and the gate files). For the pilot, these facts are read from the prediction log before cycle 2 starts.
  - **Final test, report only** (`make loop-final`, once, after the last gate): on `test_frozen`, the original v2 (the `initial_champion` log entry) against the final `@champion`. It reports macro-F1 with the grouped cluster bootstrap interval of ADR-022 (`evaluate.bootstrap`), the paired difference on the same resamples, per-class recall and error counts. It goes to `results/loop/final_test_frozen.json`. It writes nothing to the registry, and no decision, threshold or later cycle may use it.
  - **Summary:** `make loop-summary` builds `results/loop/loop_summary.json` and the table in `results/loop/README.md` (cycle, labels, reviewer error, val macro-F1 champion vs challenger, verdict, gate file) from those files only. A test checks that the table matches the JSON.
  - The final `@champion` gets a rebuilt serving bundle, golden outputs and a passing `make docker-smoke`.
- **Reason:** Reusing the pilot's unlabelled images is the only honest way to run more cycles on NEU, and saying so keeps the claim honest. Disjoint chunks make sure that no image is labelled twice and that each cycle's new labels are known. A fault-injection run checks the gate itself: a gate that never rejects proves nothing.
- **Consequences:**
  - Each cycle adds only about as many labels as its chunk routes (about 6% in the pilot). Val has 180 images, so differences between cycles are mostly within noise; the summary says so.
  - The fault-injection model is registered and kept like any version, but it can never be promoted.
  - Three Kaggle runs: cycle 2, cycle 3 and the fault run.
- **Amendment (2026-11-12, after the fault run):** the fault-injection model (v6) never reached the gate. The challenger's review threshold (ADR-023) refused it, because the risk upper bound among accepted val images was above `threshold.max_risk_ucb`, so it never got the tags the gate needs. For a fault-injection run only, such a refusal is now **recorded as the run's outcome**:
  - the retrain cycle record says `stopped before the gate` and gives the check and its message;
  - the challenger, serving and gate steps are skipped; the checkpoint is still kept (`versions`);
  - `@challenger` is never moved.
  - The loop table shows this outcome as it is. The gate's own checks are not evaluated, and nothing is relaxed so that the gate could run.
  - In a real cycle the same refusal still stops the flow for an owner decision.

## ADR-037 — Retrain calibration: inherit the champion's T when the val fit hits its bound
- **Context:** **This rule was added after cycle 2 of TASK-030 exposed a gap.** ADR-006 says a temperature fit on a range bound writes nothing and stops. The cycle-2 challenger (v4) classified all 180 val images correctly. With no errors, mean NLL keeps falling as T → 0, so the fit always lands on `calibrate.t_min`. `make retrain` stopped at `calibrate`, and the challenger could never reach the gate. The gate (ADR-033) does not use T, but a promoted version must be servable, which needs T, the review threshold and the OOD threshold. So any challenger with a perfect val score, and more generally any challenger whose fit hits a bound, was locked out.
- **Decision (owner decision of 2026-11-12, written and committed before the code):**
  - **General rule for every retrain** (`make retrain`, including the fault-injection run of ADR-036), not a v4 special case. If the challenger's val fit sits on either bound of `[t_min, t_max]`, the challenger **inherits the current champion's validated T**: the `temperature` tag of the `@champion` version recorded at preflight. Otherwise the fitted T is used, as before.
  - The challenger's `calibration.json` records:
    - `calibration: inherited_T` (or `fitted`);
    - the bound fit itself (`fit.temperature`, `fit.at_bound: true`);
    - where T came from (champion version and T);
    - the val reports with the inherited T, plus the cross-fitted report.
  - `at_bound` describes the T in use, so it is `false` for an inherited T.
  - The version is tagged `calibration=inherited_T` (fitted versions are tagged `calibration=fitted`). The retrain cycle record and the loop summary table show it.
  - Everything after calibration is unchanged and still computed on the challenger's own val logits with the T in use: the review threshold (ADR-023, including its risk check), the ONNX parity check, the OOD threshold (ADR-027) and the gate.
  - `make calibrate` outside a retrain keeps the ADR-006 behaviour: a bound fit stops the run.
- **Reason:** A bound fit says that val cannot determine T for this model. It does not say the model is unusable. The champion's T is the only temperature this system has validated (fitted on val off its bound), and the challenger shares its architecture, training recipe and data (ADR-034). Inheriting it is the least arbitrary choice, and it is recorded rather than hidden. Resetting to T = 1 would replace a validated value with an assumption.
- **Consequences:**
  - An inherited T is **not validated for the challenger**. Its confidences may be miscalibrated by an unknown amount, and the val ECE reported with the inherited T rests on zero errors.
  - The review threshold still routes the least confident 5% of val (ADR-023), so routing stays bounded even if T is off.
  - A chain of inherited Ts is possible if several champions in a row inherit; each version's tags and `calibration.json` show where its T came from.
  - v4 resumes from `calibrate` under this rule. Nothing is retrained.

## ADR-038 — Alembic migrations; the API checks the schema revision but never migrates; Postgres tested locally and in CI
- **Context:** TASK-037. The prediction log (ADR-014, ADR-029) has so far been created with `create_all`, and one later column (`reason_audit`, ADR-030) was added by ad-hoc "additive only" code in `src/db/store.py`. Azure will use PostgreSQL (ADR-003/012). Schema changes need real, ordered, reviewable migrations, and an app must never change a shared database's schema on its own.
- **Decision (owner decisions of 2026-11-12, written and committed before the code):**
  - **Alembic** owns the schema. The migrations live in `src/db/migrations/` and are configured from code (no `alembic.ini`). The **baseline revision** is today's schema: `predictions` (including `reason_audit`) and `feedback`, written out by hand in the revision. It does not import the models, so it stays frozen. A test checks that a database at head matches the models exactly (Alembic's autogenerate comparison finds no difference), so a model change without a migration fails the tests.
  - `PredictionLog.create_tables` / `upgrade_schema` (the additive-schema code of ADR-030) are **removed**.
  - **Who migrates:** only `make db-migrate` (`python -m src.db.migrate upgrade`: `alembic upgrade head` on `DATABASE_URL`) and `make db-reset` (a fresh local database, created at head).
    - In docker compose, a one-shot `migrate` service (same image, same `DATABASE_URL`) runs to completion before the API starts.
  - **The API never migrates.** At startup it reads the database's revision and **refuses to start** unless it is at head. The error says which revision it found, which one it expected, and to run `make db-migrate`.
    - The review UI, the simulated reviewer and the feedback export apply the same check.
  - **Existing pre-Alembic databases** (`data/predictions.db`, `data/archive/*.db`):
    - `db-migrate` finds no `alembic_version` table but finds the app tables. It compares their schema with a fresh baseline database: tables, columns, nullability, types, indexes, unique constraints and foreign keys.
    - Only if they are identical does it **stamp** the baseline, then upgrade to head. Otherwise it refuses and changes nothing.
    - A non-SQLite database with app tables but no revision is always refused; there are no such databases.
  - **Driver:** a `postgresql://` URL (the form Azure gives) is read as `postgresql+psycopg://`.
  - **Dependencies (owner-approved, serve group, each released at least two weeks before 2026-11-12):**
    - `alembic==1.20.1` (2026-10-22; already in the lock through mlflow);
    - `psycopg[binary]==3.3.6` (2026-09-18).
  - **Local Postgres:** a compose profile `postgres` with `postgres:16.14` (2026-08-05), pinned by digest.
    - Bound to `127.0.0.1:${POSTGRES_PORT:-5433}`.
    - User and database `defect`, with a **fixed, non-secret password used only for local tests** (`defect-local-test`). It is never used for any deployed database.
    - The compose API reads `API_DATABASE_URL`, which defaults to its SQLite volume.
  - **DB tests on Postgres:** `TEST_DATABASE_URL` points the DB and migration tests at Postgres; without it they use a temporary SQLite file. These tests drop and recreate the schema, so they refuse any host other than `localhost` / `127.0.0.1`.
  - **Smoke:** `make docker-smoke` keeps using SQLite and writes `results/docker/image_report.json`, including the new image size. `make docker-smoke-postgres` runs the same API checks with the API on the compose Postgres and writes `results/docker/image_report_postgres.json`.
  - **CI:** a new job runs the same pinned Postgres image as a service container, then `make db-migrate` and the DB/migration tests against it.
- **Reason:**
  - Migrations make every schema change explicit, ordered and reviewable.
  - Refusing to start on the wrong revision turns a silent mismatch into a clear deploy-time error.
  - Keeping migration out of the app means several replicas or apps (API, review UI) never race to change a shared schema.
  - Stamping only an identical legacy schema keeps old local logs usable without guessing.
- **Consequences:**
  - Every schema change from now on needs an Alembic revision.
  - After pulling this change, existing local logs need `make db-migrate` once.
  - Azure will need a migration step before each deploy (a Container Apps job; TASK-039/041).
  - The API image grows by Alembic and psycopg; the size is recorded in `results/docker/image_report.json`.

## ADR-039 — Azure deployment: one Bicep file, image pushed to ACR, managed-identity pull, migrate before start, public Postgres with a firewall
- **Context:** TASK-036, 038 and 039. The API (ADR-012, ADR-028) and its Postgres log (ADR-014, ADR-038) go to Azure for Students. The subscription policy allows only a few regions, and Container Apps is not offered in `indiasouthcentral`. The deadline is 14 Nov 2026, so the setup must be small, repeatable and cheap.
- **Decision (owner decisions of 2026-11-13, written before the code):**
  - **Infrastructure as code:** `infra/main.bicep` plus `infra/main.parameters.json` (non-secret values only). It is deployed with `az deployment group create` (incremental, idempotent) to resource group `rg-steelsentry` in **`uaenorth`**. The backup region is `eastasia`.
  - **Resources:**
    - Log Analytics workspace (30-day retention).
    - Azure Container Registry, **Basic**, admin user **off**.
    - A user-assigned managed identity with **AcrPull** on the registry. A user-assigned identity, not a system-assigned one, so the pull permission exists before the app is created.
    - A Container Apps environment (consumption only) that sends logs to Log Analytics.
    - Postgres Flexible Server **B1ms (Burstable)**, 32 GB, **PostgreSQL 16**, `require_secure_transport=on`, 7-day backups (the minimum), no geo-redundant backup, no HA. Database `defect`.
  - **Postgres network:** public access with firewall rules only, and **no VNet**.
    - The rule "allow Azure services" (`0.0.0.0`) stays, so the Container App can connect.
    - The deployer's IP is added only for the migration step and removed right after, including when the migration fails.
    - **Trade-off:** "allow Azure services" admits traffic from any Azure tenant's resources, not only ours. Access still needs the admin password and TLS. A VNet-integrated environment with private access would close this, but it costs more, takes longer to build, and needs a workload-profiles environment. Out of scope for the sprint.
  - **Image:** built locally from the commit being deployed (`make docker-build`, serving bundle inside, ADR-028), then pushed with `docker push` to ACR. ACR Tasks are not used. The Container App references the image **by digest**, not by tag.
  - **Container App:** external HTTPS ingress to port 8000, **min replicas 0, max 1**, 0.5 vCPU / 1 GiB.
    - Probes: liveness and readiness on `/health`.
    - Container App secrets: `API_KEY`, `AUDIT_SALT` and `DATABASE_URL` (`postgresql+psycopg://…?sslmode=require`), exposed to the app as env vars.
    - Their values come only from the deployer's environment (`.env` loaded by `uv run --env-file`). They are written to a temporary parameters file that is deleted after the deployment. They are never printed or committed, and never written to `results/`.
  - **Migrations before start:** `azure-deploy` runs `python -m src.db.migrate upgrade` against the Azure database (the IP window above) **before** the app revision is created. The API itself never migrates (ADR-038); it refuses to start on a database that is not at head.
  - **Cloud smoke:** `/health` 200; `/predict` with the committed **synthetic** golden image (no NEU pixels go to the cloud, ADR-015/017) matches `tests/golden/expected.json` (prediction and confidence within 1e-4); 401 without the key; client-side p50 latency over 20 requests after a warm-up.
    - The random audit (ADR-030) may add `audit` to the reasons, so the reasons are compared without `audit`.
    - Written to `results/cloud/deploy_report.json`: URL, revision, image digest, smoke results, latencies. No secrets, no IP address.
  - **Make targets:** `azure-up` (infrastructure), `azure-deploy` (build, push, migrate, app, smoke), `azure-smoke`, `pg-stop` / `pg-start`, and `azure-down` (deletes the resource group after typing its name).
  - **Budget:** a $10/month budget on the resource group (separate `infra/budget.bicep`), alerting the subscription Owner role at 50/80/100%, so no email address is in the repo. If the offer does not support budgets, `azure-up` warns and goes on; the Azure for Students credit is the hard cap.
- **Reason:**
  - One Bicep file makes the environment reviewable and repeatable.
  - Managed-identity pull removes registry passwords.
  - Digest references make the deployed bits exact.
  - Migrating first keeps the API's "never migrate" rule (ADR-038).
  - Scale to zero and a stoppable B1ms keep the cost near the estimate.
- **Consequences:**
  - Estimated cost ≈ $24/month with Postgres running all the time (retail prices for uaenorth, 2026-11-13), or ≈ $9.5 with `make pg-stop` when idle. A stopped server is restarted by Azure after 7 days.
  - The first request after scale to zero is a cold start (TASK-043 measures it).
  - Container Apps has no read-only root filesystem option, unlike `docker-compose.yml`.
  - The Container Apps environment was created as an **express** environment. It rejects custom revision suffixes (`ExpressRevisionSuffixNotSupported`), so the single revision is always `steelsentry-api--latest` and each deploy updates it in place. The deployed bits are identified by the **image digest** in `deploy_report.json`. Rollback (TASK-042) must redeploy the previous digest; it cannot switch revisions.
  - The API connects as the server admin. A least-privilege app role is a follow-up.
  - CI/CD deploy from `main` (TASK-041) is still open; it will reuse `infra/main.bicep`.
  - **Region change (owner, 2026-11-13; amends the decision above):**
    - Azure refused Postgres `Standard_B1ms` in `uaenorth` for this subscription with `SkuNotAvailable`. It was tried with no zone, then zones 1, 2 and 3, although `list-skus` lists B1ms in zones 1-3. Each failed server was dropped by Azure.
    - The owner first chose to stop by mistake, then asked for a probe instead. `make pg-probe` creates **only** the Postgres server (B1ms, no zone, random name) in the order eastasia, indiasouthcentral, malaysiawest, indonesiacentral, and keeps the first success. eastasia succeeded on the first try (`steelsentry-pg-ce1ab5ca`, 458 s).
    - Rule given in advance: if eastasia works, everything moves there. All resources now run in **`eastasia`**; the resource group's metadata stays in `uaenorth`.
    - Resource names carry the region (`-eastasia`), so the move could run side by side. The uaenorth Log Analytics workspace, ACR, identity and environment were deleted once the eastasia ones were up.
    - Postgres moved into its own module, `infra/postgres.bicep`. The probe deploys only that module; `main.bicep` uses the same module, so it adopts the probed server without changing it.
    - **Cost in eastasia (retail prices, 2026-11-13):** B1ms $0.0286/h ≈ $20.88, storage 32 GB × $0.15 ≈ $4.80, ACR Basic ≈ $5.07. That is **≈ $30.75/month** with Postgres running all the time, above the $25 line set before creation. With `make pg-stop` when idle it is ≈ $9.9 (storage + ACR). Container Apps (scale to zero) and Log Analytics stay within their free grants. Flagged to the owner.

## ADR-040 — Deploy workflow: GitHub OIDC to a managed identity, bundle from @champion, Trivy on the image, automatic rollback to the previous digest
- **Context:** TASK-041, plus a light TASK-042. ADR-039 deploys from the laptop. Deploys should run from `main` with no long-lived Azure secret. The express Container Apps environment keeps a single revision (`--latest`), so a rollback means redeploying the previous image.
- **Decision (owner decisions of 2026-11-13, written before the code):**
  - **Auth: GitHub OIDC, no client secret.**
    - A user-assigned managed identity, `steelsentry-gh-deploy`, has a **federated credential** for `repo:aryanwaingankar4/defect-inspection-mlops:ref:refs/heads/main`. Everything is in `infra/github-deploy.bicep`.
    - Least privilege, scoped to `rg-steelsentry` only:
      - **AcrPush** on the registry;
      - **Container Apps Contributor** on the resource group, to update the app;
      - **Managed Identity Operator** on the pull identity only, because updating an app that uses that identity needs its `assign` action.
    - No subscription-level role.
    - `AZURE_CLIENT_ID`, `AZURE_TENANT_ID` and `AZURE_SUBSCRIPTION_ID` are GitHub **repo variables**. They are identifiers, not secrets. The resource group, registry and app names are variables too.
  - **Triggers:** `workflow_dispatch`, and `workflow_run` after **CI** completes successfully on `main`. The workflow checks out the commit CI tested.
  - **Model source:** the workflow builds the serving bundle from **`@champion`** in the DagsHub MLflow registry (`make serving-bundle`; read-only, SHA-256 checked, ADR-028).
    - No NEU images are needed. It installs the `train` group because the tracking module imports `mlflow.pytorch`.
    - `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` are GitHub **secrets**, set with `gh secret set` from the deployer's environment and never printed.
    - **The token is the owner's DagsHub token, which can write.** A read-only token would be better (follow-up).
  - **Steps:**
    1. Build the bundle, then the image (`make docker-build`).
    2. **Trivy** on the built image, with the CI policy (TASK-035): report HIGH and CRITICAL, fail on CRITICAL with a fix available.
    3. Push to ACR tagged with the **git SHA**.
    4. Deploy **by digest** (`az containerapp update --image`).
    5. Cloud smoke: `/health`, the golden `/predict` against `tests/golden/expected.json`, and 401 without the key. The smoke needs `API_KEY`, a GitHub secret.
    - The app's other secrets (`AUDIT_SALT`, `DATABASE_URL`) stay as Container App secrets and are not in GitHub. Infrastructure and migrations stay on `make azure-up` / `make azure-deploy` (ADR-039). The workflow only changes the image.
  - **Rollback (TASK-042 light):**
    - Before deploying, the workflow records the image digest that is currently running.
    - If the update or the smoke fails, it redeploys that digest, re-runs the smoke, writes `results/cloud/rollback_*.json` and **fails** the workflow.
    - The golden check also catches a model change: if `@champion` moves without `make golden`, the smoke fails and the previous image comes back.
  - **Rollback drill:** run once with a deliberately broken image (the good image with a start command that exits), recorded in `results/cloud/rollback_drill.json`.
  - **Proving runs (what actually happened, 2026-11-13):**
    - The identity was created with a federated credential for `main` **only**, as decided.
    - A temporary credential for the feature branch would have allowed a real run before merge. Setting it up was blocked by the agent's safety check, and it was not part of the owner's decisions, so it was not created.
    - `workflow_dispatch` only works for workflows on the default branch, so **the first real workflow run happens after the PR is merged**: the `workflow_run` trigger after CI on `main`, or a manual dispatch. Its run URL goes into `results/cloud/deploy_report.json` then.
    - The **rollback drill** was run from the deployer's machine with the same code path as the workflow (`scripts/azure.py ci-deploy --drill`), against the live app, recorded in `results/cloud/rollback_drill.json` (`workflow_run_url: null`):
      - the broken image (the good image with a start command that exits) failed `/health`;
      - the previous digest was redeployed;
      - the smoke then passed (model v5);
      - the command exited non-zero.
    - A drill through the workflow (`rollback_drill: true`) is possible after merge.
- **Reason:**
  - OIDC removes the long-lived secret.
  - Scoped roles limit what a compromised workflow could do.
  - Building from `@champion` makes the registry the source of truth for what ships.
  - Rollback by digest is exact and fits the express environment.
- **Consequences:**
  - Each deploy installs torch through the `train` group, which is slow (see the post-sprint CPU-torch task).
  - Each deploy runs 22 synthetic `/predict` calls, which are logged in the Azure database like any other request.
  - A failed smoke always ends with a red workflow, even when the rollback succeeds, so a person looks at it.
