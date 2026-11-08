"""Shift-tolerant near-copy check (TASK-008b, ADR-021).

pHash (TASK-004) compares pixels at fixed positions, so a shifted crop of the same surface
slips through. NEU images are crops of longer steel strips, and many crops overlap.

``pairs`` (split-independent; slow: CNN forward passes on CPU, no training):
  1. CNN shortlist. Every image goes through the shared preprocessing (``load_image`` ->
     ``to_model_input``) and an ImageNet-pretrained CNN with its head removed (an independent
     feature extractor, NOT our trained model). Global pooling makes the embedding nearly
     shift-invariant. Pairs (all 1,800 x 1,799 / 2) with cosine >= ``candidate_cosine`` are
     candidates.
  2. Pixel check on every candidate: remove the local mean (lighting gradients make unrelated
     steel images correlate), then take the best Pearson correlation of the remaining fine
     texture over all shifts up to ``max_shift`` px (FFT). Same surface: close to 1;
     different surfaces: near 0.
  A pair is a confirmed near-copy if it is a candidate with pixel correlation >=
  ``confirm_pixel_corr``. Candidates with pixel correlation >= ``keep_pixel_corr`` go to the
  pairs file (committed; paths and scores only). ``src.data.split`` groups the confirmed pairs.
  Calibration (in the summary), because all steel textures of one class look alike: cosine and
  pixel correlation of random same-class pairs, and cosine of synthetic shifted copies (crop A
  vs crop B moved s px, same image vs another same-class image).

``report`` (fast, per split): reads the pairs file and a split CSV and lists the overlaps that
cross splits. ``--require-clean`` exits 1 if any confirmed near-copy crosses splits.

Known recall limit: a copy shifted more than ``max_shift`` px, or whose cosine is below
``candidate_cosine``, can be missed.

Usage:
    python -m src.data.near_duplicates_cnn pairs --config configs/params.yaml
    python -m src.data.near_duplicates_cnn report --config configs/params.yaml [--require-clean]
        [--split-file CSV --out-csv CSV --out-summary JSON --out-grid PNG]
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.config import load_params
from src.data.groups import Overlap, connected_groups, read_overlaps, write_overlaps
from src.data.split import Row, list_images, read_split_csv
from src.serving.preprocess import PreprocessConfig, load_image, to_model_input

# Grayscale ``L`` images -> (n, d) embeddings.
Embedder = Callable[[Sequence[Image.Image]], np.ndarray]
Image_ = tuple[str, str]  # (path relative to raw_dir, class)

REPORT_FIELDS = (
    "pixel_corr",
    "cosine",
    "confirmed",
    "path_a",
    "class_a",
    "split_a",
    "path_b",
    "class_b",
    "split_b",
)
PERCENTILES = (1, 5, 50, 95, 99, 99.9)
DECIMALS = 4


@dataclass(frozen=True)
class CnnDupConfig:
    """Settings from the ``near_duplicates_cnn`` section of params.yaml."""

    model: str
    batch_size: int
    seed: int
    report_cosines: tuple[float, ...]
    report_pixel_corrs: tuple[float, ...]
    candidate_cosine: float
    confirm_pixel_corr: float
    keep_pixel_corr: float
    highpass_px: int
    max_shift: int
    calib_same_class_pairs: int
    calib_pixel_pairs: int
    calib_shift_images_per_class: int
    calib_shifts: tuple[int, ...]
    max_candidates: int
    summary_top_k: int
    grid_top_k: int
    watch_pairs: tuple[tuple[str, str], ...]
    pairs_file: Path
    pairs_summary: Path
    out_csv: Path
    out_summary: Path
    out_grid: Path

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> CnnDupConfig:
        c = params["near_duplicates_cnn"]
        return cls(
            model=str(c["model"]),
            batch_size=int(c["batch_size"]),
            seed=int(c["seed"]),
            report_cosines=tuple(float(t) for t in c["report_cosines"]),
            report_pixel_corrs=tuple(float(t) for t in c["report_pixel_corrs"]),
            candidate_cosine=float(c["candidate_cosine"]),
            confirm_pixel_corr=float(c["confirm_pixel_corr"]),
            keep_pixel_corr=float(c["keep_pixel_corr"]),
            highpass_px=int(c["highpass_px"]),
            max_shift=int(c["max_shift"]),
            calib_same_class_pairs=int(c["calib_same_class_pairs"]),
            calib_pixel_pairs=int(c["calib_pixel_pairs"]),
            calib_shift_images_per_class=int(c["calib_shift_images_per_class"]),
            calib_shifts=tuple(int(s) for s in c["calib_shifts"]),
            max_candidates=int(c["max_candidates"]),
            summary_top_k=int(c["summary_top_k"]),
            grid_top_k=int(c["grid_top_k"]),
            watch_pairs=tuple((str(a), str(b)) for a, b in c.get("watch_pairs", [])),
            pairs_file=Path(c["pairs_file"]),
            pairs_summary=Path(c["pairs_summary"]),
            out_csv=Path(c["out_csv"]),
            out_summary=Path(c["out_summary"]),
            out_grid=Path(c["out_grid"]),
        )


# ---------------------------------------------------------------- embeddings


def l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def timm_embedder(model_name: str, batch_size: int, pre: PreprocessConfig) -> Embedder:
    """ImageNet-pretrained timm backbone with no head (pooled features), eval mode, CPU."""
    import torch

    from src.training.train import build_model

    model = build_model(model_name, num_classes=0, pretrained=True).eval()

    def embed(images: Sequence[Image.Image]) -> np.ndarray:
        out = []
        with torch.inference_mode():
            for i in range(0, len(images), batch_size):
                batch = np.stack([to_model_input(im, pre) for im in images[i : i + batch_size]])
                out.append(model(torch.from_numpy(batch)).numpy())
        return np.concatenate(out).astype(np.float32)

    return embed


# ---------------------------------------------------------------- pixel check


def highpass(a: np.ndarray, box: int) -> np.ndarray:
    """Image minus its local mean over ``box`` x ``box`` (odd): keeps fine texture, drops
    lighting gradients. Returns the valid part, ``box - 1`` px smaller per side."""
    if box < 1 or box % 2 == 0:
        raise ValueError(f"highpass box must be odd and positive, got {box}")
    x = a.astype(np.float64)
    ii = np.pad(x.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    local = (ii[box:, box:] - ii[:-box, box:] - ii[box:, :-box] + ii[:-box, :-box]) / box**2
    r = box // 2
    return x[r : r + local.shape[0], r : r + local.shape[1]] - local


def _overlap_sums(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Sum of ``x`` over rows [max(0, dy), h + min(0, dy)) and the same for columns, for every
    (dy, dx) in ``d`` x ``d``, using an integral image."""
    h, w = x.shape
    ii = np.pad(x.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    r0, r1 = np.maximum(0, d), h + np.minimum(0, d)
    c0, c1 = np.maximum(0, d), w + np.minimum(0, d)
    return ii[np.ix_(r1, c1)] - ii[np.ix_(r0, c1)] - ii[np.ix_(r1, c0)] + ii[np.ix_(r0, c0)]


def best_shift_correlation(a: np.ndarray, b: np.ndarray, max_shift: int) -> float:
    """Highest Pearson correlation between ``a`` moved by (dy, dx) and ``b``, over every shift
    in [-max_shift, max_shift] on both axes (overlap only). 1.0 means ``b`` is an exact
    shifted crop of ``a``. Same result as looping over shifts, computed with FFTs."""
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} vs {b.shape}")
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    h, w = a.shape
    d = np.arange(-max_shift, max_shift + 1)
    n = np.outer(h - np.abs(d), w - np.abs(d)).astype(np.float64)
    sa, sa2 = _overlap_sums(a, d), _overlap_sums(a * a, d)
    sb, sb2 = _overlap_sums(b, -d), _overlap_sums(b * b, -d)
    shape = (h + max_shift, w + max_shift)  # zero padding: no circular wrap for |shift| <= max
    full = np.fft.irfft2(np.fft.rfft2(a, shape) * np.conj(np.fft.rfft2(b, shape)), shape)
    sab = full[np.ix_(d % shape[0], d % shape[1])]  # sum of a[y + dy, x + dx] * b[y, x]
    var = (sa2 - sa * sa / n) * (sb2 - sb * sb / n)
    ok = var > 1e-9 * sa2 * sb2 + 1e-12  # flat overlap: no correlation defined
    if not ok.any():
        return 0.0
    corr = (sab - sa * sb / n)[ok] / np.sqrt(var[ok])
    return float(np.clip(corr.max(), -1.0, 1.0))


# ---------------------------------------------------------------- helpers


def percentiles(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    out = {f"p{q:g}": round(float(np.percentile(v, q)), DECIMALS) for q in PERCENTILES}
    out["min"] = round(float(v.min()), DECIMALS)
    out["max"] = round(float(v.max()), DECIMALS)
    return out


def image_list_sha256(paths: Sequence[str]) -> str:
    """Identity of the image set (sorted paths), so a report can check it matches the pairs."""
    return hashlib.sha256("\n".join(sorted(paths)).encode("utf-8")).hexdigest()


def sample_same_class_pairs(
    classes: Sequence[str], n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` random pairs (i, j), i != j, of images with the same class."""
    c = np.asarray(classes)
    members = {k: np.flatnonzero(c == k) for k in sorted(set(classes))}
    a_idx, b_idx = [], []
    while len(a_idx) < n:
        a = int(rng.integers(len(c)))
        pool = members[c[a]]
        b = int(pool[int(rng.integers(len(pool)))])
        if b != a:
            a_idx.append(a)
            b_idx.append(b)
    return np.asarray(a_idx), np.asarray(b_idx)


def sample_per_class(classes: Sequence[str], n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` random image indices from every class (sorted)."""
    c = np.asarray(classes)
    picks = [
        rng.choice(np.flatnonzero(c == k), size=n, replace=False) for k in sorted(set(classes))
    ]
    return np.sort(np.concatenate(picks))


def shifted_crops(
    image: Image.Image, shift: int, max_shift: int
) -> tuple[Image.Image, Image.Image]:
    """Two equal crops of one image, the second moved ``shift`` px right and down."""
    w, h = image.size
    cw, ch = w - max_shift, h - max_shift
    return image.crop((0, 0, cw, ch)), image.crop((shift, shift, shift + cw, shift + ch))


def shift_calibration(
    images: Sequence[Image.Image],
    classes: Sequence[str],
    shifts: Sequence[int],
    embed: Embedder,
    rng: np.random.Generator,
) -> dict[str, dict[str, Any]]:
    """Cosine of crop A vs crop B shifted by s: same image (a real copy) vs another same-class
    image (a different surface)."""
    partner = np.empty(len(images), dtype=np.int64)  # a different image of the same class
    for c in sorted(set(classes)):
        members = rng.permutation(np.flatnonzero(np.asarray(classes) == c))
        if len(members) < 2:
            raise ValueError(f"shift calibration needs >= 2 images of class {c!r}")
        partner[members] = np.roll(members, 1)  # a cycle: never the image itself
    max_shift = max(shifts)
    out = {}
    for s in shifts:
        crops = [shifted_crops(im, s, max_shift) for im in images]
        a = l2_normalise(embed([c[0] for c in crops]))
        b = l2_normalise(embed([c[1] for c in crops]))
        out[str(s)] = {
            "same_image": percentiles(np.sum(a * b, axis=1)),
            "different_image": percentiles(np.sum(a * b[partner], axis=1)),
        }
    return out


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data, indent=2) + "\n").encode("utf-8"))


def overlap_order(o: Overlap) -> tuple[float, float, str, str]:
    return (-o.pixel_corr, -o.cosine, o.path_a, o.path_b)


# ---------------------------------------------------------------- step 1: pairs


def find_overlaps(
    images: Sequence[Image_], raw_dir: Path, cfg: CnnDupConfig, embed: Embedder
) -> tuple[dict[str, Any], list[Overlap]]:
    """Score every pair of images (any split). Returns (summary, kept overlaps, best first)."""
    rng = np.random.default_rng(cfg.seed)
    paths = [p for p, _ in images]
    classes = [c for _, c in images]
    pixels = [load_image(raw_dir / p) for p in paths]
    emb = l2_normalise(embed(pixels))
    sims = emb @ emb.T
    ii, jj = np.triu_indices(len(images), k=1)
    all_cos = sims[ii, jj]

    @functools.lru_cache(maxsize=4096)
    def texture(k: int) -> np.ndarray:
        return highpass(np.asarray(pixels[k]), cfg.highpass_px)

    def pixel(a: int, b: int) -> float:
        return best_shift_correlation(texture(a), texture(b), cfg.max_shift)

    cand = np.flatnonzero(all_cos >= cfg.candidate_cosine)
    if len(cand) > cfg.max_candidates:
        raise ValueError(
            f"{len(cand)} pairs have cosine >= {cfg.candidate_cosine}: more than "
            f"max_candidates={cfg.max_candidates}"
        )
    cand_pix = np.empty(len(cand))
    for n, k in enumerate(cand):  # ordered by image a (np.triu_indices): texture cache hits
        cand_pix[n] = pixel(int(ii[k]), int(jj[k]))

    overlaps = sorted(
        (
            Overlap(
                round(float(cand_pix[n]), DECIMALS),
                round(float(all_cos[k]), DECIMALS),
                paths[ii[k]],
                classes[ii[k]],
                paths[jj[k]],
                classes[jj[k]],
            )
            for n, k in enumerate(cand)
            if cand_pix[n] >= cfg.keep_pixel_corr
        ),
        key=overlap_order,
    )
    sure = [o for o in overlaps if o.pixel_corr >= cfg.confirm_pixel_corr]
    groups = connected_groups((o.path_a, o.path_b) for o in sure)
    cls_of = dict(zip(paths, classes, strict=True))

    # Calibration.
    sa, sb = sample_same_class_pairs(classes, cfg.calib_same_class_pairs, rng)
    pa, pb = sa[: cfg.calib_pixel_pairs], sb[: cfg.calib_pixel_pairs]
    pixel_baseline = np.asarray([pixel(int(a), int(b)) for a, b in zip(pa, pb, strict=True)])
    shift_idx = sample_per_class(classes, cfg.calib_shift_images_per_class, rng)
    shift = shift_calibration(
        [pixels[k] for k in shift_idx],
        [classes[k] for k in shift_idx],
        cfg.calib_shifts,
        embed,
        rng,
    )

    index = {p: k for k, p in enumerate(paths)}
    names = {Path(p).name: p for p in paths}
    if len(names) != len(paths):
        raise ValueError("file names are not unique; watch_pairs need unique names")

    def watch(pair: tuple[str, str]) -> dict[str, Any]:
        a, b = index[names[pair[0]]], index[names[pair[1]]]
        cos, corr = float(sims[a, b]), pixel(a, b)
        return {
            "path_a": paths[a],
            "path_b": paths[b],
            "cosine": round(cos, DECIMALS),
            "cosine_rank_among_all_pairs": int(np.sum(all_cos > cos)) + 1,
            "pixel_corr": round(corr, DECIMALS),
            "confirmed": bool(cos >= cfg.candidate_cosine and corr >= cfg.confirm_pixel_corr),
        }

    def pixel_counts(t: float) -> dict[str, int]:
        sel = [o for o in overlaps if o.pixel_corr >= t]
        same = sum(o.class_a == o.class_b for o in sel)
        return {"pairs": len(sel), "same_class": same, "different_class": len(sel) - same}

    summary = {
        "note": "Forward passes only (ImageNet-pretrained backbone, not our trained model).",
        "model": cfg.model,
        "embedding_dim": int(emb.shape[1]),
        "images": len(paths),
        "image_list_sha256": image_list_sha256(paths),
        "pairs_compared": int(len(all_cos)),
        "calibration": {
            "cosine_random_same_class": {"pairs": int(len(sa)), **percentiles(sims[sa, sb])},
            "cosine_shifted_crops": {
                "images": int(len(shift_idx)),
                "crop_px": f"image side - {max(cfg.calib_shifts)}",
                "by_shift_px": shift,
            },
            "pixel_corr_random_same_class": {"pairs": int(len(pa)), **percentiles(pixel_baseline)},
        },
        "pairs_cosine_ge": {f"{t:g}": int(np.sum(all_cos >= t)) for t in cfg.report_cosines},
        "rule": {
            "candidate_cosine_ge": cfg.candidate_cosine,
            "confirm_pixel_corr_ge": cfg.confirm_pixel_corr,
            "highpass_px": cfg.highpass_px,
            "max_shift_px": cfg.max_shift,
            "kept_in_pairs_file_pixel_corr_ge": cfg.keep_pixel_corr,
        },
        "candidates": int(len(cand)),
        "candidates_pixel_corr_ge": {f"{t:g}": pixel_counts(t) for t in cfg.report_pixel_corrs},
        "confirmed_pairs": len(sure),
        "confirmed_different_class": [
            {"pixel_corr": o.pixel_corr, "path_a": o.path_a, "path_b": o.path_b}
            for o in sure
            if o.class_a != o.class_b
        ],
        "groups": {
            "count": len(groups),
            "images": sum(len(g) for g in groups),
            "size_histogram": {
                str(s): n for s, n in sorted(Counter(len(g) for g in groups).items())
            },
            "images_by_class": dict(sorted(Counter(cls_of[p] for g in groups for p in g).items())),
            "largest": {"size": max((len(g) for g in groups), default=0)},
        },
        "top_pairs": [o.__dict__ for o in overlaps[: cfg.summary_top_k]],
        "watch_pairs": [watch(w) for w in cfg.watch_pairs],
    }
    return summary, overlaps


# ---------------------------------------------------------------- step 2: report per split


def split_pair_name(a: str, b: str) -> str:
    return " / ".join(sorted((a, b)))


def cross_split_report(
    overlaps: Sequence[Overlap], rows: Sequence[Row], cfg: CnnDupConfig
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Overlaps whose two images are in different splits. Returns (summary, table rows)."""
    split_of = {r[0]: r[2] for r in rows}
    missing = {p for o in overlaps for p in (o.path_a, o.path_b)} - split_of.keys()
    if missing:
        raise ValueError(f"{len(missing)} paths in the pairs file are not in the split")
    table = []
    for o in overlaps:
        sa, sb = split_of[o.path_a], split_of[o.path_b]
        if sa != sb:
            table.append(
                {
                    "pixel_corr": o.pixel_corr,
                    "cosine": o.cosine,
                    "confirmed": int(o.pixel_corr >= cfg.confirm_pixel_corr),
                    "path_a": o.path_a,
                    "class_a": o.class_a,
                    "split_a": sa,
                    "path_b": o.path_b,
                    "class_b": o.class_b,
                    "split_b": sb,
                }
            )

    def by_split(sel: Sequence[dict[str, Any]]) -> dict[str, int]:
        names = Counter(split_pair_name(t["split_a"], t["split_b"]) for t in sel)
        return dict(sorted(names.items()))

    sure = [t for t in table if t["confirmed"]]
    names = {Path(r[0]).name: r for r in rows}
    summary = {
        "confirm_pixel_corr_ge": cfg.confirm_pixel_corr,
        "confirmed_cross_split": len(sure),
        "confirmed_cross_split_by_split_pair": by_split(sure),
        "cross_split_overlaps_pixel_corr_ge": {
            f"{t:g}": {
                "pairs": sum(x["pixel_corr"] >= t for x in table),
                "by_split_pair": by_split([x for x in table if x["pixel_corr"] >= t]),
            }
            for t in (cfg.keep_pixel_corr, *cfg.report_pixel_corrs)
        },
        "top_cross_split": table[: cfg.summary_top_k],
        "watch_pairs": [
            {
                "path_a": names[a][0],
                "split_a": names[a][2],
                "path_b": names[b][0],
                "split_b": names[b][2],
                "same_split": names[a][2] == names[b][2],
            }
            for a, b in cfg.watch_pairs
        ],
    }
    return summary, table


def write_report_csv(table: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)


def draw_grid(
    table: Sequence[Mapping[str, Any]], raw_dir: Path, path: Path, tile: int = 160
) -> None:
    """One row per pair: image A | image B | pixel correlation, cosine, splits, classes."""
    label_w, pad = 380, 6
    width = 2 * tile + label_w + 4 * pad
    height = max(1, len(table)) * (tile + pad) + pad
    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)
    for n, t in enumerate(table):
        y = pad + n * (tile + pad)
        for k, key in enumerate(("path_a", "path_b")):
            tile_img = load_image(raw_dir / t[key]).resize((tile, tile))
            canvas.paste(tile_img, (pad + k * (tile + pad), y))
        text = (
            f"#{n + 1} pix={t['pixel_corr']:.3f} cos={t['cosine']:.4f}"
            f"{'  CONFIRMED' if t['confirmed'] else ''}\n"
            f"A: {t['split_a']} / {t['class_a']}\n{Path(t['path_a']).name}\n"
            f"B: {t['split_b']} / {t['class_b']}\n{Path(t['path_b']).name}"
        )
        draw.multiline_text((3 * pad + 2 * tile, y + 4), text, fill=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


# ---------------------------------------------------------------- CLI


def run_pairs(params: Mapping[str, Any], embed: Embedder | None = None) -> dict[str, Any]:
    cfg = CnnDupConfig.from_params(params)
    raw_dir = Path(params["data"]["raw_dir"])
    images = [(p, c) for c, ps in list_images(raw_dir).items() for p in ps]
    if embed is None:
        embed = timm_embedder(cfg.model, cfg.batch_size, PreprocessConfig.from_params(params))
    summary, overlaps = find_overlaps(images, raw_dir, cfg, embed)
    summary["params"] = {
        k: v
        for k, v in params["near_duplicates_cnn"].items()
        if not k.startswith(("out_", "pairs_"))
    }
    write_overlaps(overlaps, cfg.pairs_file)
    write_json(summary, cfg.pairs_summary)
    print(
        f"[near-copies] {summary['pairs_compared']:,} pairs, {summary['candidates']:,} candidates"
    )
    print(
        f"[near-copies] confirmed: {summary['confirmed_pairs']} pairs in "
        f"{summary['groups']['count']} groups ({summary['groups']['images']} images)"
    )
    print(f"[near-copies] wrote {cfg.pairs_file}, {cfg.pairs_summary}")
    return summary


def run_report(
    params: Mapping[str, Any],
    split_file: Path | None = None,
    out_csv: Path | None = None,
    out_summary: Path | None = None,
    out_grid: Path | None = None,
) -> dict[str, Any]:
    cfg = CnnDupConfig.from_params(params)
    raw_dir = Path(params["data"]["raw_dir"])
    split_file = split_file or Path(params["data"]["split_file"])
    rows = read_split_csv(split_file)
    pairs_summary = json.loads(cfg.pairs_summary.read_text(encoding="utf-8"))
    if image_list_sha256([r[0] for r in rows]) != pairs_summary["image_list_sha256"]:
        raise ValueError(f"{split_file} has a different image set than {cfg.pairs_file}")
    summary, table = cross_split_report(read_overlaps(cfg.pairs_file), rows, cfg)
    summary = {
        "split_file": split_file.as_posix(),
        "pairs_file": cfg.pairs_file.as_posix(),
        **summary,
    }
    write_report_csv(table, out_csv or cfg.out_csv)
    write_json(summary, out_summary or cfg.out_summary)
    draw_grid(table[: cfg.grid_top_k], raw_dir, out_grid or cfg.out_grid)
    print(
        f"[near-dups-cnn] {split_file}: {summary['confirmed_cross_split']} confirmed near-copies "
        f"cross splits {summary['confirmed_cross_split_by_split_pair']}"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shift-tolerant near-copy check (TASK-008b).")
    sub = parser.add_subparsers(dest="step", required=True)
    p = sub.add_parser("pairs", help="score all image pairs (slow; CNN on CPU)")
    p.add_argument("--config", required=True, type=Path)
    r = sub.add_parser("report", help="list overlaps that cross splits")
    r.add_argument("--config", required=True, type=Path)
    r.add_argument("--split-file", type=Path, help="override data.split_file")
    r.add_argument("--out-csv", type=Path)
    r.add_argument("--out-summary", type=Path)
    r.add_argument("--out-grid", type=Path, help="real NEU pixels: keep under results/private/")
    r.add_argument("--require-clean", action="store_true", help="exit 1 if any cross split")
    args = parser.parse_args(argv)

    params = load_params(args.config)
    if args.step == "pairs":
        run_pairs(params)
        return 0
    summary = run_report(params, args.split_file, args.out_csv, args.out_summary, args.out_grid)
    if args.require_clean and summary["confirmed_cross_split"]:
        print("[near-dups-cnn] NOT CLEAN: confirmed near-copies cross splits")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
