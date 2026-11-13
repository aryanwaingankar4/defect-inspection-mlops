# Works with GNU Make in Git Bash (Windows) and on Linux. Needs uv on PATH.
.DEFAULT_GOAL := help

RUN := uv run --locked
# Training deps (torch, timm, matplotlib) live in the `train` group.
RUN_TRAIN := uv run --locked --group train
# The API needs only the serve group (ONNX Runtime, FastAPI; no torch, no mlflow).
RUN_SERVE := uv run --locked --group serve
# The review UI (Streamlit) needs the ui group plus the database layer from the serve group.
RUN_UI := uv run --locked --group ui --group serve
# Publishing to MLflow loads secrets (DagsHub) from .env if it exists. Training and tests never do.
ENV_FILE := $(if $(wildcard .env),--env-file .env,)

# Tests that need data, a model, a human, or a long time are skipped by `make test`.
FAST_TESTS := not slow and not manual and not needs_data and not needs_model

# Pinned Kaggle copy of NEU (NEU-DET release, .jpg). See ADR-015 and docs/DATA_CARD.md.
NEU_DATASET := kaustubhdikshit/neu-surface-defect-database/1
RAW_DIR := data/raw
PARAMS := configs/params.yaml
IMAGE := defect-inspection-api:local
MANIFEST := models/serving/manifest.json
# Fixed, non-secret audit salt for the smoke checks only (ADR-030); the real salt is in .env.
SMOKE_AUDIT_SALT := smoke-test-audit-salt-not-secret

.PHONY: help setup lock lint format test test-all check data split near-dups near-copies near-dups-cnn repro push pull train train-smoke \
	compare-val calibrate threshold ood evaluate champion export-onnx quantize serving-bundle serve golden openapi-snapshot docker-build docker-up docker-down docker-smoke docker-smoke-postgres monitoring-smoke db-migrate pg-up pg-down test-db-postgres review-ui stream simulate-review db-reset feedback-export gate retrain challenger-serving rollback challenger loop-chunks loop-record fault-export loop-final loop-summary register kaggle-dataset kaggle-push kaggle-status kaggle-output \
	azure-up azure-deploy azure-smoke azure-github-oidc load-test pg-probe pg-stop pg-start azure-down

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
	$(RUN_TRAIN) --group serve --group ui pytest -m "$(FAST_TESTS)"

test-all: ## All tests, including slow / manual / needs_data / needs_model
	$(RUN_TRAIN) --group serve --group ui pytest

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

threshold: ## Review threshold on val for the main model (CPU; ADR-023). ARGS: --log-mlflow
	$(RUN_TRAIN) $(ENV_FILE) python -m src.training.threshold --config $(PARAMS) $(ARGS)

ood: ## Energy OOD threshold on val + synthetic non-steel evaluation (ADR-027). ARGS: --log-mlflow
	$(RUN_TRAIN) $(ENV_FILE) python -m src.training.ood --config $(PARAMS) $(ARGS)

evaluate: ## Both models on test_frozen -> results/eval/ (CPU; after compare-val, ADR-022). ARGS: --log-mlflow
	$(RUN_TRAIN) $(ENV_FILE) python -m src.training.evaluate --config $(PARAMS) $(ARGS)

export-onnx: ## FP32 ONNX of @champion + parity on val (CPU; ADR-025). ARGS: --log-mlflow
	$(RUN_TRAIN) $(ENV_FILE) python -m src.training.export_onnx --config $(PARAMS) $(ARGS)

quantize: ## Static INT8 QDQ of the champion ONNX + FP32 vs INT8 on val (ADR-026). ARGS: --log-mlflow
	$(RUN_TRAIN) $(ENV_FILE) python -m src.training.quantize --config $(PARAMS) $(ARGS)

serving-bundle: ## Download the @champion serving ONNX + tags into models/serving/ (read-only; ADR-028)
	$(RUN_TRAIN) $(ENV_FILE) python -m src.tracking.serving_bundle --config $(PARAMS)

serve: ## Run the API on http://127.0.0.1:8000 from models/serving/ (API_KEY from .env; ADR-028)
	$(RUN_SERVE) $(ENV_FILE) python -m src.serving.app --config $(PARAMS) $(ARGS)

docker-build: ## Build the API image with the serving bundle inside (needs `make serving-bundle`; TASK-021)
	@test -f $(MANIFEST) || { echo "missing $(MANIFEST): run 'make serving-bundle' first"; exit 1; }
	docker build -t $(IMAGE) \
		--build-arg MODEL_VERSION=$$($(RUN) python -c "import json; print(json.load(open('$(MANIFEST)'))['version'])") \
		--build-arg ONNX_SHA256=$$($(RUN) python -c "import json; print(json.load(open('$(MANIFEST)'))['onnx_sha256'])") \
		--build-arg GIT_COMMIT=$$(git rev-parse --short HEAD) .

docker-up: docker-build ## API :8000 + Prometheus :9090 + Grafana :3000 (secrets from .env); log in the Docker volume, not data/predictions.db
	docker compose up -d --no-build

docker-down: ## Stop the Docker stack (volumes are kept)
	docker compose down

docker-smoke: docker-build ## Build, start and check the API container, then remove it (synthetic image only)
	AUDIT_SALT=$(SMOKE_AUDIT_SALT) $(RUN) python scripts/docker_smoke.py

docker-smoke-postgres: docker-build ## The same smoke with the API on the compose test Postgres (ADR-038)
	AUDIT_SALT=$(SMOKE_AUDIT_SALT) $(RUN) python scripts/docker_smoke.py --postgres

monitoring-smoke: docker-build ## API + Prometheus + Grafana: metrics scraped, dashboard provisioned (TASK-032)
	AUDIT_SALT=$(SMOKE_AUDIT_SALT) $(RUN) python scripts/monitoring_smoke.py

# The review loop (review-ui, simulate-review, feedback-export) reads the LOCAL log from DATABASE_URL
# (data/predictions.db), which `make serve` writes. The Docker API writes to the `predictions` volume
# instead, which these targets do not read: run `make serve`, not `make docker-up`, for the loop (ADR-032).
review-ui: ## Review queue UI on http://127.0.0.1:8501; reads the local log of `make serve`, not the Docker volume (ADR-031/032)
	$(RUN_UI) $(ENV_FILE) streamlit run src/review/app.py --server.address 127.0.0.1 --server.port 8501 \
		--browser.gatherUsageStats false -- --config $(PARAMS)

N ?= 40
stream: ## SIMULATED stream: send N production_stream images to the API on :8000 (API_KEY from .env). ARGS: --start K | --chunk C --out F
	$(RUN) $(ENV_FILE) python -m src.pipelines.stream --config $(PARAMS) -n $(N) $(ARGS)

simulate-review: ## SIMULATED reviewer labels production_stream images in the local queue. ARGS: --limit, --error-rate, --out F
	$(RUN_SERVE) $(ENV_FILE) python -m src.review.simulated --config $(PARAMS) $(ARGS)

db-migrate: ## Upgrade the DATABASE_URL schema to the Alembic head; adopts a pre-Alembic SQLite log (ADR-038)
	$(RUN_SERVE) $(ENV_FILE) python -m src.db.migrate upgrade

# Local TEST Postgres only (ADR-038): fixed, non-secret password, bound to 127.0.0.1. Never a real database.
PG_TEST_URL := postgresql+psycopg://defect:defect-local-test@127.0.0.1:5433/defect

pg-up: ## Start the local TEST Postgres (compose profile postgres, 127.0.0.1:5433; ADR-038)
	docker compose --profile postgres up -d --wait postgres

pg-down: ## Stop and remove the local TEST Postgres and its volume
	docker compose --profile postgres rm -s -f -v postgres
	-docker volume rm defect-inspection_postgres

test-db-postgres: ## make db-migrate + the DB/migration tests against the local TEST Postgres (after pg-up)
	$(MAKE) db-migrate DATABASE_URL=$(PG_TEST_URL) ENV_FILE=
	TEST_DATABASE_URL=$(PG_TEST_URL) $(RUN_SERVE) pytest tests/test_db.py tests/test_migrations.py

db-reset: ## Move the local SQLite log to data/archive/ and start an empty one (stop serve/review-ui first; ADR-032)
	$(RUN_SERVE) $(ENV_FILE) python -m src.db.reset --config $(PARAMS)

feedback-export: ## production_stream labels -> data/feedback/feedback.csv (DVC) + results/feedback/ (ADR-032)
	$(RUN_SERVE) $(ENV_FILE) python -m src.review.export --config $(PARAMS)
	$(RUN) dvc add data/feedback
	@echo ""
	@echo "New feedback version. Commit it and back it up:"
	@echo "  git add data/feedback.dvc results/feedback/export_report.json && make push"

golden: ## Golden images + expected @champion outputs -> tests/golden/ (needs bundle + NEU val; ADR-017). ARGS: --val-predictions CSV
	$(RUN_SERVE) python -m src.serving.golden --config $(PARAMS) $(ARGS)

openapi-snapshot: ## Rewrite the API contract tests/contract/openapi.json (only for an intended API change)
	$(RUN_SERVE) python -m src.serving.contract --config $(PARAMS)

champion: ## Set the first @champion with linked evidence (ADR-024). Dry run; ARGS=--apply writes
	$(RUN_TRAIN) $(ENV_FILE) python -m src.tracking.promote --config $(PARAMS) $(ARGS)

gate: ## Non-inferiority gate @challenger vs @champion on val (ADR-033). ARGS: --challenger-checkpoint P [--apply]
	$(RUN_TRAIN) $(ENV_FILE) python -m src.tracking.gate --config $(PARAMS) $(ARGS)

retrain: ## SIMULATED-loop cycle: feedback -> Kaggle GPU -> @challenger -> gate record (resumes; ARGS=--fresh; ADR-034)
	$(RUN_TRAIN) --group serve $(ENV_FILE) python -m src.pipelines.retrain --config $(PARAMS) $(ARGS)

challenger-serving: ## Make @challenger servable: FP32 ONNX + parity, serving_precision, OOD threshold on val (ADR-035)
	$(RUN_TRAIN) $(ENV_FILE) python -m src.pipelines.challenger_serving --config $(PARAMS)

rollback: ## Move @champion back to its previous version (or VERSION=v) if servable; writes results/promotion/ (ADR-035)
	$(RUN_TRAIN) $(ENV_FILE) python -m src.tracking.rollback --config $(PARAMS) $(if $(VERSION),--version $(VERSION),)

challenger: ## Point @challenger at VERSION=v (servable, not the champion) to re-run the gate (ADR-035)
	@test -n "$(VERSION)" || { echo "usage: make challenger VERSION=v"; exit 1; }
	$(RUN_TRAIN) $(ENV_FILE) python -m src.tracking.rollback --config $(PARAMS) --set-challenger $(VERSION)

loop-chunks: ## Disjoint stream chunks after the pilot -> data/splits/loop_chunks.csv (needs the pilot export; ADR-036)
	$(RUN) python -m src.pipelines.loop --config $(PARAMS) chunks

loop-record: ## Record a loop cycle: CYCLE=n|fault GATE=gate file -> results/loop/ (ADR-036)
	@test -n "$(CYCLE)" || { echo "usage: make loop-record CYCLE=n GATE=results/promotion/gate_<v>.json"; exit 1; }
	$(RUN_SERVE) $(ENV_FILE) python -m src.pipelines.loop --config $(PARAMS) record --cycle $(CYCLE) $(if $(GATE),--gate-file $(GATE),)

fault-export: ## FAULT INJECTION: copy of the export with every label shifted -> data/feedback_fault/ (never real feedback; ADR-036)
	$(RUN) python -m src.pipelines.loop --config $(PARAMS) fault-export

loop-final: ## Once, after the last gate: original v2 vs final @champion on test_frozen, REPORT ONLY (ADR-036)
	$(RUN_TRAIN) $(ENV_FILE) python -m src.pipelines.loop --config $(PARAMS) final

loop-summary: ## results/loop/loop_summary.json + README.md table from the cycle records (ADR-036)
	$(RUN) python -m src.pipelines.loop --config $(PARAMS) summary

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

# Azure (ADR-039): rg-steelsentry in uaenorth. Needs `az login`; secrets come from .env only.
AZURE := $(RUN) $(ENV_FILE) python scripts/azure.py

azure-up: ## Azure infrastructure from infra/main.bicep (idempotent; no app) + $$10 budget (ADR-039)
	$(AZURE) up

azure-deploy: ## Build + push the image to ACR, migrate Azure Postgres, deploy the API app, cloud smoke
	$(AZURE) deploy

azure-smoke: ## Cloud smoke of the running app -> results/cloud/deploy_report.json (synthetic image only)
	$(AZURE) smoke

azure-github-oidc: ## OIDC deploy identity (no secret) + GitHub repo variables/secrets from .env (ADR-040)
	$(AZURE) github-setup

load-test: ## TASK-043: warm up, then Locust 20 users x 2 min on the Azure URL (synthetic image) -> results/load/
	uv run --locked --group load $(ENV_FILE) python scripts/load_test.py $(ARGS)

pg-probe: ## Create ONLY Postgres B1ms region by region (eastasia first), keep the first success (ADR-039)
	$(AZURE) pg-probe

pg-stop: ## Stop the Azure Postgres server (saves compute cost; Azure restarts it after 7 days)
	$(AZURE) pg-stop

pg-start: ## Start the Azure Postgres server
	$(AZURE) pg-start

azure-down: ## DELETE the whole Azure resource group (asks you to type its name)
	$(AZURE) down
