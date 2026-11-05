# Works with GNU Make in Git Bash (Windows) and on Linux. Needs uv on PATH.
.DEFAULT_GOAL := help

RUN := uv run --locked

# Tests that need data, a model, a human, or a long time are skipped by `make test`.
FAST_TESTS := not slow and not manual and not needs_data and not needs_model

.PHONY: help setup lock lint format test test-all check

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
