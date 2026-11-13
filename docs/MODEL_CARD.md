# Model card: `steel-defect-classifier`

Every number below comes from a committed file in `results/`, and each table says which file. Nothing is estimated or hand-typed.

## Summary

| | |
|---|---|
| Task | Classify one hot-rolled steel surface image into one of 6 defect classes |
| Classes | crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches |
| Architecture | EfficientNet-B0 (timm, ImageNet weights), 4,015,234 parameters (`results/retrain/9b1c9dbd6d464eb291ba2605b3f35de6/val_metrics.json`) |
| Serving model | Version **5** in the MLflow registry on DagsHub, alias `@champion`, served as **FP32 ONNX** (`tests/golden/expected.json`, `results/cloud/deploy_report.json`) |
| Output | Class, temperature-calibrated confidence, energy OOD score, and whether to route to human review (low confidence, OOD or random audit) |
| Intended use | Portfolio demonstration of a human-in-the-loop MLOps loop. **Not for real production inspection.** |

## Data

- **Images:** NEU surface defect database, the Kaggle copy of the NEU-DET release. 1,800 grayscale images, 300 per class (`docs/DATA_CARD.md`). The bounding boxes are ignored, because this is classification only.
- **Licence:** NEU has no redistribution licence. The data stays private: in a private Kaggle dataset and a local DVC remote. No NEU pixel is committed, logged to the public MLflow, or sent to the cloud (ADR-015, ADR-018).
- **Split v2 (group-aware, ADR-021), per class:**
  - 25 `train_v1`;
  - 30 `val`;
  - 60 `test_frozen`;
  - 185 `production_stream`.

  That is 150 / 180 / 360 / 1,110 images in total (`configs/params.yaml`).
- **Leakage fix:**
  - NEU contains overlapping crops of the same steel. pHash found no cross-split duplicates within distance 8 (`results/archive/split_v1/near_duplicates_summary.json`).
  - A shift-tolerant check (CNN shortlist plus pixel correlation) found **193** confirmed near-copy pairs across split v1 (`results/archive/split_v1/near_duplicates_cnn_summary.json`).
  - Split v2 keeps each near-copy group in one split, leaving **0** across splits (`results/near_duplicates_cnn_summary.json`).
  - Val macro-F1 dropped from 0.9890 (v1, `results/archive/split_v1/train/val_metrics.json`) to 0.9780 (v2, `results/train/val_metrics.json`). That is the more honest number.
- **Feedback:** 121 simulated reviewer labels on `production_stream` images were added over 3 cycles (`results/loop/README.md`).
- **Not used:** X-SDD (a planned second production line) was not started.

## Training

- **Where:** Kaggle GPU only (ADR-019).
- **How:** AdamW, class-balanced sampling, early stopping on val macro-F1, with hyperparameters from `configs/params.yaml`. Every retrain starts from the ImageNet weights, never from the previous model (ADR-004).
- **v2, first champion:** trained on `train_v1` only (150 images).
- **v5, current champion:** trained on `train_v1` plus 121 feedback labels, 271 images in total. Best epoch 15 of 23 (`results/retrain/9b1c9dbd6d464eb291ba2605b3f35de6/val_metrics.json`).
- **Model choice (ADR-022):** EfficientNet-B0 was the pre-declared main model. On val, MobileNetV3-small was right on 4 discordant images and EfficientNet-B0 on 1. The exact McNemar p = 0.375 is not significant, so the main model stayed (`results/compare/val_mcnemar.json`).

## Metrics

### `test_frozen` (360 images; report only, never used for a decision)

Source: `results/loop/final_test_frozen.json`. 95% intervals come from a cluster bootstrap over near-copy groups.

| Model | Macro-F1 [95% CI] | Errors |
|---|---|---|
| v2 (first champion) | 0.9917 [0.9808, 1.0000] | 3 / 360 |
| v5 (current champion) | 0.9972 [0.9912, 1.0000] | 1 / 360 |

- **Difference (v5 − v2):** +0.0055 [+0.0000, +0.0141]. The interval touches 0, so this is **not** a demonstrated improvement.
- **Paired results:** both right on 357 images, only v5 right on 2, only v2 right on 0, both wrong on 1.
- **v5 per-class recall:** 1.0 for every class except scratches, at 0.9833. Its one error is a scratches image predicted as inclusion.

### v5 on `val` (180 images): calibration, routing and OOD

| Quantity | Value | Source |
|---|---|---|
| Val macro-F1 | 0.9944 (1 error) | `results/retrain/9b1c…/val_metrics.json` |
| Temperature T | 1.3706 (fitted on 180 images with 1 error, so uncertain) | `…/calibration.json` |
| Review threshold | 0.9655: routes 9 / 180 (5%) | `…/threshold.json` |
| Error-rate upper bound on accepted images | 0.0274 (one-sided 95% Clopper–Pearson) | `…/threshold.json` |
| OOD energy threshold | −6.199 (2% val flag budget) | `…/ood.json` |
| Synthetic non-steel images flagged OOD | 259 / 400 (64.75%); energy AUROC 0.855 | `…/ood.json` |
| ONNX parity (val) | max abs logit diff 1.47e-4; same accuracy as PyTorch | `…/onnx_export.json` |

`…` = `results/retrain/9b1c9dbd6d464eb291ba2605b3f35de6/`.

v5 flags fewer synthetic non-steel images than v2 did. v2 flagged 322 / 400 with AUROC 0.9715 (`results/train/ood.json`). OOD detection is weaker on the current champion.

### Precision: FP32 serves, INT8 rejected

Source: `results/onnx/quantization_report.json`, measured on v2.

| | FP32 | Static INT8 (QDQ, calibrated on `train_v1`) |
|---|---|---|
| Val macro-F1 | 0.9780 | **0.4264** |
| Size | 16,543,323 bytes | 5,175,095 bytes |
| Laptop CPU latency, batch 1 (median) | 37.3 ms | 108.6 ms |

INT8 failed the rule fixed beforehand (ADR-026), so FP32 serves. The latencies are rough, single-process numbers.

### Cloud serving

| | Value | Source |
|---|---|---|
| Cloud smoke | passed: `/health`, golden `/predict`, 401 without key | `results/cloud/deploy_report.json` |
| `/predict` p50, 20 sequential requests from a GitHub runner | 562.6 ms (includes the round trip to East Asia) | `results/cloud/deploy_report.json` |
| Load test (Locust, 20 users, 2 min, laptop client) | 14.84 req/s; p50 170 ms, p95 910 ms, p99 1,500 ms; 0 errors in 1,781 requests | `results/load/locust_report.json` |

## Limitations

- **Small evaluation sets.** Val has 180 images and test 360. Differences of a few images are within noise, and v5's calibration rests on 1 val error.
- **Easy, clean dataset.** NEU is lab-quality, 200×200 crops with one defect each. Real lines have mixed defects, other cameras, lighting drift and defect-free images; none of these are covered.
- **Classification only.** No localisation. The NEU-DET boxes exist but are unused, so detection is future work.
- **OOD is limited.** It is tested only on synthetic non-steel images and is weaker on v5 (above). Unseen defect types were not tested, because X-SDD was not started.
- **Inherited temperature.** ADR-037 lets a challenger inherit the champion's T when its own fit hits a bound. That T is not validated for the challenger.
- **No drift monitoring.** Evidently was not built.

## What is simulated

- **The reviewer** is simulated. Labels come from the dataset's true labels, with a seeded error rate of 0% (cycles 1–2) or 5% (cycle 3). The planned 15% cycle was not run.
- **Production traffic** is simulated by streaming the held-out `production_stream` split through the API.
- **The fault-injection run** used a broken reviewer: every label shifted. It was stopped by the review-threshold risk check before the promotion gate (`results/loop/fault_injection/cycle.json`).
- **Real:** the training (Kaggle GPU), the registry and gate decisions (DagsHub MLflow, `results/promotion/`), the serving and the Azure deployment.
