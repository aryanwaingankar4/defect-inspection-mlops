# Archive: results on split v1 (leaky)

Everything in this folder was produced on `data/splits/neu_split_v1.csv`, the first split
(TASK-003). TASK-008b later found that NEU crops often overlap: confirmed near-copies of the
same steel surface sit in different splits of v1, including `train_v1` / `val` and
`test_frozen` / other splits. The current split is the group-aware
`data/splits/neu_split_v2.csv` (ADR-021).

Kept for history and for the v1 vs v2 comparison. Do not quote these numbers as current
results: validation metrics here are somewhat optimistic because `val` shares surfaces with
`train_v1`.

| path | what |
|---|---|
| `train/` | baseline efficientnet_b0, MLflow `steel-defect-classifier` version 1 (TASK-010) |
| `compare/mobilenetv3_small_100/` | MobileNetV3-small comparison run (not registered) |
| `near_duplicates.csv`, `near_duplicates_summary.json` | pHash report on v1 (TASK-004) |
| `near_duplicates_cnn.csv`, `near_duplicates_cnn_summary.json` | near-copy report on v1 (TASK-008b), from `python -m src.data.near_duplicates_cnn report --split-file data/splits/neu_split_v1.csv ...` |
