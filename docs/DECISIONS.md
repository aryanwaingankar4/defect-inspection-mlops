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

- **Implemented (TASK-012):** `make calibrate` (`src/training/calibrate.py`) fits one temperature per model on `val` only (mean NLL; declared range `calibrate.t_min`..`t_max`; a fit on a bound writes nothing and stops). `T` is stored in `calibration.json` next to each model's results (`results/train/`, `results/compare/mobilenetv3_small_100/`), with what it was fitted on and the checkpoint hash; TASK-015 applies it in ONNX / serving. Val reports ECE, NLL and Brier before/after in-sample and cross-fitted (5 class-stratified folds, near-copy groups kept together), with grouped bootstrap intervals for each score, the change, and `T` itself. `make evaluate` applies `T` on `test_frozen` (never fits it) and reports the same scores. With only a few misclassified images in each set, the intervals of the changes include 0 for both models on test; see the JSON files for the numbers. `T` is kept as fitted: nothing is chosen from these results.

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
