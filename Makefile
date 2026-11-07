# Works with GNU Make in Git Bash (Windows) and on Linux. Needs uv on PATH.
.DEFAULT_GOAL := help

RUN := uv run --locked
# Training deps (torch, timm, matplotlib) live in the `train` group.
RUN_TRAIN := uv run --locked --group train
# Publishing to MLflow loads secrets (DagsHub) from .env if it exists. Training and tests never do.
ENV_FILE := $(if $(wildcard .env),--env-file .env,)

# Tests that need data, a model, a human, or a long time are skipped by `make test`.
FAST_TESTS := not slow and not manual and not needs_data and not needs_model

# Pinned Kaggle copy of NEU (NEU-DET release, .jpg). See ADR-015 and docs/DATA_CARD.md.
NEU_DATASET := kaustubhdikshit/neu-surface-defect-database/1
RAW_DIR := data/raw
PARAMS := configs/params.yaml

.PHONY: help setup lock lint format test test-all check data split near-dups near-copies near-dups-cnn repro push pull train train-smoke \
	compare-val calibrate evaluate register kaggle-dataset kaggle-push kaggle-status kaggle-output

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-10s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create .venv from the lockfile (fails if uv.lock is out of date)
	uv sync --locked

lock: ## Re-resolve dependencies and update uv.lock
	uv lock

lint: ## Ruff lint + format check (no changes)
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format: ## Auto-fix lint issues and format code
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

test: ## Fast tests (skips slow, manual, needs_data, needs_model)
	$(RUN_TRAIN) pytest -m "$(FAST_TESTS)"

test-all: ## All tests, including slow / manual / needs_data / needs_model
	$(RUN_TRAIN) pytest

check: lint test ## Lint + fast tests (what CI runs)

data: ## Download NEU into data/raw (skips if already complete; needs ~/.kaggle/kaggle.json)
	$(RUN) python -m src.data.download --dataset $(NEU_DATASET) --out $(RAW_DIR)

split: ## Make the split CSVs: v1 (history) and group-aware v2 (needs the near-copy pairs file)
	$(RUN) python -m src.data.split --config $(PARAMS)

near-dups: ## pHash near-duplicate report across splits -> results/ (grid in results/private/)
	$(RUN) python -m src.data.near_duplicates --config $(PARAMS)

near-copies: ## Score all image pairs for overlaps (pretrained CNN on CPU + pixel check; ~15 min)
	$(RUN_TRAIN) python -m src.data.near_duplicates_cnn pairs --config $(PARAMS)

near-dups-cnn: ## Overlaps across the current split -> results/; fails if any confirmed one crosses
	$(RUN_TRAIN) python -m src.data.near_duplicates_cnn report --config $(PARAMS) --require-clean

repro: ## Re-run the DVC pipeline (dvc.yaml) where inputs changed (near_dups_cnn needs torch)
	$(RUN_TRAIN) dvc repro
	@echo ""
	@echo "Reminder: DVC autostage is off. Stage the results by hand:"
	@echo "  git add dvc.lock   (plus any changed cache: false outputs; see git status)"

push: ## Upload DVC-tracked data to the private local remote (../dvc-storage)
	$(RUN) dvc push

pull: ## Restore DVC-tracked data from the private local remote
	$(RUN) dvc pull

train: ## Full training, files only (Kaggle GPU; on a CPU laptop it stops unless ARGS=--allow-cpu, ADR-019)
	$(RUN_TRAIN) python -m src.training.train --config $(PARAMS) $(ARGS)

train-smoke: ## Tiny end-to-end check on real data (CPU OK): train, then publish to the smoke experiment
	$(RUN_TRAIN) python -m src.training.train --config $(PARAMS) --smoke $(ARGS)
	$(RUN_TRAIN) $(ENV_FILE) python -m src.tracking.publish --config $(PARAMS) --smoke --register

compare-val: ## Paired McNemar test of the two v2 models on val (CPU inference; ADR-022)
	$(RUN_TRAIN) python -m src.training.compare_val --config $(PARAMS) $(ARGS)

calibrate: ## Temperature scaling on val for both models (CPU; ADR-006). ARGS: --log-mlflow
	$(RUN_TRAIN) $(ENV_FILE) python -m src.training.calibrate --config $(PARAMS) $(ARGS)

evaluate: ## Both models on test_frozen -> results/eval/ (CPU; after compare-val, ADR-022). ARGS: --log-mlflow
	$(RUN_TRAIN) $(ENV_FILE) python -m src.training.evaluate --config $(PARAMS) $(ARGS)

register: ## Publish the downloaded Kaggle run to MLflow. ARGS: --register, --role comparison, --results-dir DIR (ADR-020)
	$(RUN_TRAIN) $(ENV_FILE) python -m src.tracking.publish --config $(PARAMS) $(ARGS)

kaggle-dataset: ## Upload data/raw to Kaggle as a PRIVATE dataset and verify it stays private
	$(RUN) python -m src.data.kaggle_dataset --config $(PARAMS)

kaggle-push: ## Push + run the launcher pinned to HEAD (committed + pushed). ARGS: --model NAME, --no-run (ADR-020)
	$(RUN) python -m src.pipelines.kaggle_job --config $(PARAMS) push $(ARGS)

kaggle-status: ## Status of the latest Kaggle training run
	$(RUN) python -m src.pipelines.kaggle_job --config $(PARAMS) status

kaggle-output: ## Download the latest Kaggle run's outputs to build/kaggle-output/
	$(RUN) python -m src.pipelines.kaggle_job --config $(PARAMS) output
