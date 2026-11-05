# Works with GNU Make in Git Bash (Windows) and on Linux. Needs uv on PATH.
.DEFAULT_GOAL := help

RUN := uv run --locked

# Tests that need data, a model, a human, or a long time are skipped by `make test`.
FAST_TESTS := not slow and not manual and not needs_data and not needs_model

# Pinned Kaggle copy of NEU (NEU-DET release, .jpg). See ADR-015 and docs/DATA_CARD.md.
NEU_DATASET := kaustubhdikshit/neu-surface-defect-database/1
RAW_DIR := data/raw
PARAMS := configs/params.yaml

.PHONY: help setup lock lint format test test-all check data split near-dups repro push pull

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
	$(RUN) pytest -m "$(FAST_TESTS)"

test-all: ## All tests, including slow / manual / needs_data / needs_model
	$(RUN) pytest

check: lint test ## Lint + fast tests (what CI runs)

data: ## Download NEU into data/raw (skips if already complete; needs ~/.kaggle/kaggle.json)
	$(RUN) python -m src.data.download --dataset $(NEU_DATASET) --out $(RAW_DIR)

split: ## Make the deterministic train_v1/val/test_frozen/production_stream split CSV
	$(RUN) python -m src.data.split --config $(PARAMS)

near-dups: ## pHash near-duplicate report across splits -> results/ (grid in results/private/)
	$(RUN) python -m src.data.near_duplicates --config $(PARAMS)

repro: ## Re-run the DVC pipeline (dvc.yaml) where inputs changed
	$(RUN) dvc repro

push: ## Upload DVC-tracked data to the private local remote (../dvc-storage)
	$(RUN) dvc push

pull: ## Restore DVC-tracked data from the private local remote
	$(RUN) dvc pull
