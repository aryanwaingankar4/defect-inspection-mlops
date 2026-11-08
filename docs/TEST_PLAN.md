# Test Plan

"Working" means these checks pass. Unit and integration tests live in `tests/` and run in CI unless marked manual.

## Data
- The split file is deterministic for a given seed. Re-running the split yields identical files.
- No image appears in more than one split.
- The pHash near-duplicate report exists in `results/` and is committed.

## Preprocessing
- **Parity:** the same image gives the same tensor through the training path and the serving path.
- Grayscale handling (ADR-016): any input (L, RGB, 3-channel JPEG) is converted to `L`, then replicated to 3 identical channels.

## Model
- **Golden images (ADR-017):**
  - a committed synthetic image gives a fixed output with the committed model (runs in CI);
  - a fixed real NEU image from `data/raw/` gives a fixed predicted class (`needs_data`; the image is never committed).
- Temperature `T` is applied in the serving path. Serving confidence matches the calibrated evaluation output.
- FP32 vs INT8 per-class F1 is recorded in `results/`.

## API
- Contract: `/predict` returns every documented field with the right types.
- No API key, or a wrong one → 401.
- Oversized file, wrong type, undecodable file, near-blank frame → clear 4xx error.
- `/health` returns the model version and alias.
- Logs contain the image hash and `request_id`, never image bytes.

## Routing
- Low confidence → routed, reason `low_confidence`.
- OOD flag → routed, reason `ood`.
- The audit sample rate matches `params.yaml` within tolerance over a large simulated batch.

## Promotion gate (must reject bad candidates)
- A candidate equal to the champion → rejected.
- A candidate with higher accuracy but one class recall dropping beyond tolerance → rejected.
- A candidate with a non-significant tiny gain → rejected.
- A clearly better candidate → promoted, and the alias moves.
- Every case writes an entry to `promotion_log.json`.

## Rollback
- After a promotion, rollback restores the previous `@champion` and serving reports it.

## End-to-end loop (manual first, then scripted)
- Simulated stream → review queue → simulated reviewer → retrain → gate → alias update, completed for one cycle from a clean state.

## Monitoring
- Prometheus exposes latency, request count, confidence histogram, queue size and audit error rate.
- An Evidently report on a clean stream and on a shifted stream is written to `results/`.

## Database
- The DB layer passes the same tests on SQLite (CI) and on Postgres (CI service container or manual), switched only by `DATABASE_URL`.

## Deployment (Azure)
- A push to `main` runs `deploy-azure.yml`: CI passes, the image is pushed to GHCR, and both Container Apps are updated.
- **Manual:** the public Azure API URL returns `/health` with the model version and alias, and classifies a sample image through `/predict`.
- **Manual:** `/predict` on the public URL without an API key → 401.
- **Manual:** the review UI app loads and shows queued items from Postgres.
- **Manual:** after an app restart or scale to zero, logged predictions are still in Postgres.
- **Manual:** rollback redeploy restores the previous image tag, and `/health` reports the previous version.
- No Azure credentials or other secrets in the repo (Trivy secret scan in CI).
- **Manual:** the Azure budget alert exists.
- **Optional:** if a Hugging Face Space is built, it loads, classifies a sample image and shows the model version.
