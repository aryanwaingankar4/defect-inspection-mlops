# Data Card

Status: **source and licence only (TASK-002).** Composition, splits, duplicates and known issues are added in TASK-006.

## NEU surface defect database

### Source
- **Original authors:** Kechen Song and Yunhui Yan, Northeastern University (China).
- **Official page:** <https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm>
  - NEU-CLS (images only, `.bmp`): [Google Drive](https://drive.google.com/open?id=1NGlXT9sIaQpyxUoT6MLKm1Pr6x8oxOvc) · [Baidu Pan](https://pan.baidu.com/s/1l_RjTP7aTwr57ahcwelTpA)
  - NEU-DET (images + boxes, `.jpg`): [Google Drive](https://drive.google.com/open?id=1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k) · [Baidu Pan](https://pan.baidu.com/s/1nBbO-jWDm1_NHDQsc1dRkg) (code `pmqx`)
- **Copy we use:** Kaggle [`kaustubhdikshit/neu-surface-defect-database`](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database), **version 1** (last updated 2020-08-06). This is the NEU-DET release. See ADR-015.
- **Download command:** `make data` (runs `python -m src.data.download --dataset kaustubhdikshit/neu-surface-defect-database/1 --out data/raw`). Needs `~/.kaggle/kaggle.json`.
- **Downloaded:** 2026-11-05T21:55:07Z (UTC).
- **Zip SHA-256:** `2ccaff85d88567c52708c2f9dc4dec10cff8650749df2984f03b38716998460f`
  A different hash on a fresh download means the Kaggle copy changed.

### What was downloaded (checked by `src/data/download.py`)
- 1,800 `.jpg` images: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`, 300 each.
- Every file is 200×200 with 3 channels in the JPEG header. The source images are grayscale; whether the 3 channels are identical is checked in TASK-006/007.
- Layout: `data/raw/NEU-DET/{train,validation}/images/<class>/*.jpg`. The class is the folder name.
- Ignored: the uploader's train/validation split (240/60 per class; our split comes from TASK-003) and the XML bounding boxes in `annotations/`.

### Known issues (partial; completed in TASK-006)
- **Exact duplicate file:** `NEU-DET/train/images/patches/patches_101.jpg` and `patches_105.jpg` are byte-identical. Both are in `production_stream` (no cross-split leakage), so the simulated stream contains this image twice. Kept as is; documented here only. Found by TASK-004 (`results/near_duplicates_summary.json`, `within_split_alert_pairs`).
- **Shifted near-copies:** pHash does not detect slightly shifted crops of the same surface. A shift-tolerant check (TASK-008b, CNN embeddings) runs before TASK-011.

### Licence
- **No formal licence.** Kaggle lists it as "Unknown". The official page names no licence; it says "We would appreciate it if you cite our works when using the database", with a footer "Copyright 2020 Northeastern University". Community mirrors describe it as academic research use only, not commercial.
- Other Kaggle copies tagged MIT, Apache or CC0 were re-uploaded by people who do not own the data; those tags are not trusted.
- **How this project uses it:**
  - non-commercial academic/portfolio use only;
  - cite the 3 papers below;
  - **never redistribute the images**: not in git, not in a public DVC remote (the remote must be private), not in Docker images or public demos without a deliberate decision.

### Citations (requested by the authors)
1. K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," *Applied Surface Science*, vol. 285, pp. 858–864, Nov. 2013.
2. Y. He, K. Song, Q. Meng and Y. Yan, "An End-to-end Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features," *IEEE Transactions on Instrumentation and Measurement*, vol. 69, no. 4, pp. 1493–1504, 2020.
3. Y. Bao, K. Song, J. Liu, Y. Wang, Y. Yan, H. Yu and X. Li, "Triplet-Graph Reasoning Network for Few-shot Metal Generic Surface Defect Segmentation," *IEEE Transactions on Instrumentation and Measurement*, 2021.

## X-SDD
To be added when X-SDD is downloaded (Phase 2/3).
