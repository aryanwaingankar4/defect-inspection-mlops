# Test Plan

"Working" means these checks pass. Unit and integration tests live in `tests/` and run in CI unless marked manual.

## Data
- The split file is deterministic for a given seed. Re-running the split yields identical files.
- No image appears in more than one split.
- The pHash near-duplicate report exists in `results/` and is committed.

## Preprocessing
- **Parity:** the same image gives the same tensor through the training path and the serving path.
- Grayscale input is handled exactly as documented (channel replication or `in_chans=1`).

## Model
- **Golden image:** a fixed image gives a fixed predicted class with the committed model.
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

## Deployment (manual)
- The live Hugging Face Space loads, classifies a sample image and shows the model version.
