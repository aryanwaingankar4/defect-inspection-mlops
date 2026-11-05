# Data Card

Status: **complete for NEU (TASK-006).** X-SDD is added when it is downloaded (Phase 2/3).

**Where the numbers come from.** Every count in this card sits in a generated block
(`<!-- generated:... -->`) rendered from a committed file named in the block's header.
`tests/test_data_card.py` re-renders each block and fails if the card differs; the failure
message prints the block to paste. Do not edit generated blocks by hand.

## NEU surface defect database

### Intended use
- Training and evaluating an **image classifier over the NEU defect classes** for hot-rolled steel strip surface defects, as the base model of this project's simulated human-in-the-loop retraining loop.
- Measuring calibration, selective prediction (risk–coverage), OOD scoring, statistical model promotion and drift monitoring on a small, well-known benchmark.
- Non-commercial academic / portfolio use only (see [Licence](#licence)).

### Out-of-scope uses
- **Any real production or quality-control decision.** The model has never seen a real line, camera or inspector; the stream and reviewers here are simulated.
- Other steel types, other materials, other cameras or other image sizes without new data and a new evaluation.
- Defect **localisation or segmentation**: NEU-DET has bounding boxes, but this project ignores them (detection is a stretch goal only).
- Claims of state-of-the-art accuracy: the benchmark is near-solved (see [Limitations](#limitations)).
- Redistributing the images in any form (no licence; see [Licence](#licence)).

### Source
- **Original authors:** Kechen Song and Yunhui Yan, Northeastern University (China).
- **Official page:** <https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm>
  - NEU-CLS (images only, `.bmp`): [Google Drive](https://drive.google.com/open?id=1NGlXT9sIaQpyxUoT6MLKm1Pr6x8oxOvc) · [Baidu Pan](https://pan.baidu.com/s/1l_RjTP7aTwr57ahcwelTpA)
  - NEU-DET (images + boxes, `.jpg`): [Google Drive](https://drive.google.com/open?id=1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k) · [Baidu Pan](https://pan.baidu.com/s/1nBbO-jWDm1_NHDQsc1dRkg) (code `pmqx`)
- **Copy we use:** Kaggle [`kaustubhdikshit/neu-surface-defect-database`](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database), **version 1** (last updated 2020-08-06). This is the NEU-DET release. See ADR-015.
- **Download command:** `make data` (runs `python -m src.data.download --dataset kaustubhdikshit/neu-surface-defect-database/1 --out data/raw`). Needs `~/.kaggle/kaggle.json`.
- **Downloaded:** 2026-11-05T21:55:07Z (UTC). **Zip SHA-256:** `2ccaff85d88567c52708c2f9dc4dec10cff8650749df2984f03b38716998460f`.
  Both are copied from `data/raw/download_manifest.json` (DVC-tracked via `data/raw.dvc`). A different hash on a fresh download means the Kaggle copy changed.
- **Storage:** `data/raw/` is DVC-tracked; the DVC remote is a private local folder (ADR-018). No image is in git.

### Classes
The class is the folder name. Per-class counts are in the [split table](#splits).

| class | what it looks like |
|---|---|
| `crazing` | a network of fine surface cracks |
| `inclusion` | non-metallic particles embedded in the surface |
| `patches` | irregular darker or lighter areas |
| `pitted_surface` | small pits or craters |
| `rolled-in_scale` | oxide scale pressed into the surface during rolling |
| `scratches` | thin linear marks along the strip |

### Image format
- **Files:** `.jpg` (JPEG, lossy), 200×200 pixels, at `data/raw/NEU-DET/{train,validation}/images/<class>/*.jpg`.
- **Channels:** the source images are grayscale, but the Kaggle JPEGs store **3 channels** (mode `RGB` in the file).
- **How we load them (ADR-016):** open as grayscale `L`, then replicate to 3 identical channels before normalisation. This happens only in `src/serving/preprocess.py` (TASK-007), shared by training, evaluation and serving. Whether the 3 stored channels are identical does not matter: forcing `L` first removes any difference.
- **Check:** format, size and mode of every image in the split CSV are asserted by `tests/test_data_card.py::test_image_format_claim_in_card` (`needs_data`; run with `make test-all`).
- **Ignored:** the XML bounding boxes in `annotations/`, and the uploader's own train/validation folders (we build our own split):

<!-- generated:uploader-folders (source: path prefixes in data/splits/neu_split_v1.csv) -->
| uploader folder | `crazing` | `inclusion` | `patches` | `pitted_surface` | `rolled-in_scale` | `scratches` |
|---|---|---|---|---|---|---|
| `train` | 240 | 240 | 240 | 240 | 240 | 240 |
| `validation` | 60 | 60 | 60 | 60 | 60 | 60 |
<!-- /generated:uploader-folders -->

### Splits
Made by `make split` (`src/data/split.py`, TASK-003): per class, sort paths, shuffle with `split.seed` from `configs/params.yaml`, cut in the order of `split.per_class`. The result is committed as `data/splits/neu_split_v1.csv` (`path,class,split`); changing the seed or sizes changes the split, and a `needs_data` test catches any mismatch with the committed file.

| split | role |
|---|---|
| `train_v1` | deliberately small first training set, so the retraining loop has room to improve |
| `val` | early stopping, calibration (temperature) and threshold choice |
| `test_frozen` | **only** the promotion gate and final reports read it; never used to train, tune or select |
| `production_stream` | simulated incoming images; labels are revealed only through the simulated reviewer and then join retraining as feedback |

<!-- generated:split-counts (source: data/splits/neu_split_v1.csv; split order from configs/params.yaml) -->
| split | `crazing` | `inclusion` | `patches` | `pitted_surface` | `rolled-in_scale` | `scratches` | total |
|---|---|---|---|---|---|---|---|
| `train_v1` | 25 | 25 | 25 | 25 | 25 | 25 | **150** |
| `val` | 30 | 30 | 30 | 30 | 30 | 30 | **180** |
| `test_frozen` | 60 | 60 | 60 | 60 | 60 | 60 | **360** |
| `production_stream` | 185 | 185 | 185 | 185 | 185 | 185 | **1,110** |
| **total** | **300** | **300** | **300** | **300** | **300** | **300** | **1,800** |
<!-- /generated:split-counts -->

### Duplicate and leakage checks
TASK-004 computed a perceptual hash (pHash) of every image and compared every pair of images in different splits (`make near-dups`, `src/data/near_duplicates.py`):

<!-- generated:near-dups (source: results/near_duplicates_summary.json) -->
- pHash size: 64 bits (`hash_size` = 8).
- Cross-split pairs compared: 911,700.
- Cross-split pairs with Hamming distance <= 8: 0.
- Smallest cross-split distance: 10.
- Within-split pairs with distance <= 2: 1.
  - `NEU-DET/train/images/patches/patches_101.jpg` and `NEU-DET/train/images/patches/patches_105.jpg`: distance 0, split `production_stream`, class `patches`.
<!-- /generated:near-dups -->

The full cross-split distance histogram is in the same JSON file; close cross-split pairs (none at present) would be listed in `results/near_duplicates.csv`. A picture grid of the closest pairs is in `results/private/` (git-ignored, NEU pixels).

### Known issues
- **Exact duplicate file:** `patches_101.jpg` and `patches_105.jpg` are byte-identical (block above). Both are in `production_stream`, so there is no cross-split leakage, but the simulated stream contains this image twice. Kept as is and documented here only (decision after TASK-004).
- **Possible shifted near-copies (pending TASK-008b):** pHash is not shift-tolerant, so "none found" does not prove none exist. A visual review of the grid and an ad-hoc correlation check (not committed, so not a result) suggest `patches_193.jpg` (`test_frozen`) and `patches_189.jpg` (`production_stream`) may be slightly shifted crops of the same surface. This matters because `production_stream` labels feed retraining, so a `test_frozen` near-copy could leak into training in later loop cycles. TASK-008b (CNN embeddings + cosine similarity on all cross-split pairs, results to `results/`) must run before TASK-011; if near-copies cross splits, the split becomes group-aware.
- **Kaggle `.jpg` vs original `.bmp`:** the original NEU-CLS release is lossless `.bmp`; our copy is the NEU-DET release in lossy `.jpg`. JPEG artefacts are present in every image, and published results on NEU-CLS `.bmp` are not directly comparable with ours. Serving-time uploads that are not JPEG will look slightly different from the training data.
- **3-channel files from a grayscale source:** handled by loading as `L` (ADR-016, above).

### What is simulated
Code, logs and later reports must say so wherever relevant.
- **Reviewer:** there are no real inspectors. A simulated reviewer (TASK-025) answers with the true NEU label, flipped at a configurable error rate.
- **Production stream:** `production_stream` is a held-out slice of the same NEU dataset replayed as if it were live traffic. It is not real factory data and has the same distribution as training unless we change it.
- **Drift:** drift is synthetic (brightness, blur, noise, contrast generators, TASK-033) or comes from a second dataset (X-SDD, TASK-031). No drift is observed in the wild.

### Limitations
- **Small dataset:** the per-class counts in the split table are small, and `train_v1` is kept small on purpose. Per-class metrics on `test_frozen` have wide confidence intervals; the promotion gate uses a statistical test for this reason.
- **One steel type, one setup:** hot-rolled steel strip only, one source, fixed image size, grayscale. Nothing here says how the model behaves on other materials, cameras, lighting or resolutions.
- **Near-solved benchmark:** published models already classify NEU almost perfectly, so differences between good models are small and can be within noise. The project's value is the loop, calibration, gating and monitoring, not the headline accuracy. Harder conditions come from the small `train_v1`, synthetic drift and X-SDD.
- **Balanced classes:** every class has the same count. Real lines are heavily imbalanced (most strip has no defect, and there is no "no defect" class here).
- **No formal licence:** limits what can be shown publicly (no images in public demos, MLflow or docs).

### Licence
- **No formal licence.** Kaggle lists it as "Unknown". The official page names no licence; it says "We would appreciate it if you cite our works when using the database", with a footer "Copyright 2020 Northeastern University". Community mirrors describe it as academic research use only, not commercial.
- Other Kaggle copies tagged MIT, Apache or CC0 were re-uploaded by people who do not own the data; those tags are not trusted.
- **How this project uses it:**
  - non-commercial academic/portfolio use only;
  - cite the 3 papers below;
  - **never redistribute the images**: not in git, not in a public DVC remote (the remote must be private), not in public MLflow artifacts (ADR-018), not in Docker images or public demos without a deliberate decision.

### Citations (requested by the authors)
1. K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," *Applied Surface Science*, vol. 285, pp. 858–864, Nov. 2013.
2. Y. He, K. Song, Q. Meng and Y. Yan, "An End-to-end Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features," *IEEE Transactions on Instrumentation and Measurement*, vol. 69, no. 4, pp. 1493–1504, 2020.
3. Y. Bao, K. Song, J. Liu, Y. Wang, Y. Yan, H. Yu and X. Li, "Triplet-Graph Reasoning Network for Few-shot Metal Generic Surface Defect Segmentation," *IEEE Transactions on Instrumentation and Measurement*, 2021.

## X-SDD
To be added when X-SDD is downloaded (Phase 2/3).
