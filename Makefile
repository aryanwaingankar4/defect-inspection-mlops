# Works with GNU Make in Git Bash (Windows) and on Linux. Needs uv on PATH.
.DEFAULT_GOAL := help

RUN := uv run --locked
# Training deps (torch, timm, matplotlib) live in the `train` group.
RUN_TRAIN := uv run --locked --group train
# Training commands load secrets (MLflow / DagsHub) from .env if it exists. Tests never do.
ENV_FILE := $(if $(wildcard .env),--env-file .env,)

# Tests that need data, a model, a human, or a long time are skipped by `make test`.
FAST_TESTS := not slow and not manual and not needs_data and not needs_model

# Pinned Kaggle copy of NEU (NEU-DET release, .jpg). See ADR-015 and docs/DATA_CARD.md.
NEU_DATASET := kaustubhdikshit/neu-surface-defect-database/1
RAW_DIR := data/raw
PARAMS := configs/params.yaml

.PHONY: help setup lock lint format test test-all check data split near-dups repro push pull train train-smoke \
	kaggle-dataset kaggle-push kaggle-status kaggle-output

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

split: ## Make the deterministic train_v1/val/test_frozen/production_stream split CSV
	$(RUN) python -m src.data.split --config $(PARAMS)

near-dups: ## pHash near-duplicate report across splits -> results/ (grid in results/private/)
	$(RUN) python -m src.data.near_duplicates --config $(PARAMS)

repro: ## Re-run the DVC pipeline (dvc.yaml) where inputs changed
	$(RUN) dvc repro
	@echo ""
	@echo "Reminder: DVC autostage is off. Stage the results by hand:"
	@echo "  git add dvc.lock   (plus any changed cache: false outputs; see git status)"

push: ## Upload DVC-tracked data to the private local remote (../dvc-storage)
	$(RUN) dvc push

pull: ## Restore DVC-tracked data from the private local remote
	$(RUN) dvc pull

train: ## Full training (Kaggle GPU; on a CPU laptop it stops unless ARGS=--allow-cpu, ADR-019)
	$(RUN_TRAIN) $(ENV_FILE) python -m src.training.train --config $(PARAMS) $(ARGS)

train-smoke: ## Tiny end-to-end training check on real data (CPU OK; outputs in models/smoke/)
	$(RUN_TRAIN) $(ENV_FILE) python -m src.training.train --config $(PARAMS) --smoke $(ARGS)

kaggle-dataset: ## Upload data/raw to Kaggle as a PRIVATE dataset and verify it stays private
	$(RUN) python -m src.data.kaggle_dataset --config $(PARAMS)

kaggle-push: ## Push the launcher pinned to HEAD (committed + pushed). Use ARGS=--no-run, then Save & Run All on kaggle.com (ADR-019)
	$(RUN) python -m src.pipelines.kaggle_job --config $(PARAMS) push $(ARGS)

kaggle-status: ## Status of the latest Kaggle training run
	$(RUN) python -m src.pipelines.kaggle_job --config $(PARAMS) status

kaggle-output: ## Download the latest Kaggle run's outputs to build/kaggle-output/
	$(RUN) python -m src.pipelines.kaggle_job --config $(PARAMS) output
