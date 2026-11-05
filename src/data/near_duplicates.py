"""Find near-duplicate images that land in different splits, using perceptual hashes (pHash).

Steel textures look alike, so small distances can be false positives: inspect the local PNG
grid before acting. This script only reports; it never changes the split.

Outputs (paths from params.yaml):
- CSV of cross-split pairs with Hamming distance <= max_distance (committed, in results/);
- JSON summary with cumulative counts at the report distances (committed, in results/);
- PNG grid of the closest pairs (real NEU pixels, git-ignored: ADR-015).

Usage:
    python -m src.data.near_duplicates --config configs/params.yaml
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image, ImageDraw

from src.config import load_params
from src.data.split import Row, read_split_csv

# Order matches split rows, which are (path, class, split).
PAIR_FIELDS = (
    "distance",
    "path_a",
    "class_a",
    "split_a",
    "path_b",
    "class_b",
    "split_b",
    "same_class",
)


@dataclass(frozen=True)
class Pair:
    distance: int
    a: Row
    b: Row

    @property
    def same_class(self) -> bool:
        return self.a[1] == self.b[1]


def phash_int(image_path: Path, hash_size: int) -> int:
    """pHash of an image (loaded as grayscale L, as in ADR-016) as a Python int."""
    with Image.open(image_path) as img:
        bits = imagehash.phash(img.convert("L"), hash_size=hash_size).hash.flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return (a ^ b).bit_count()


def cross_split_pairs(rows: Sequence[Row], hashes: Sequence[int]) -> Iterator[Pair]:
    """Yield every pair of images that are in different splits, with its distance."""
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i][2] != rows[j][2]:
                yield Pair(hamming(hashes[i], hashes[j]), rows[i], rows[j])


def within_split_close_pairs(
    rows: Sequence[Row], hashes: Sequence[int], max_distance: int
) -> list[Pair]:
    """Pairs in the SAME split within ``max_distance``: no leakage, but duplicates in a split."""
    pairs = [
        Pair(d, rows[i], rows[j])
        for i in range(len(rows))
        for j in range(i + 1, len(rows))
        if rows[i][2] == rows[j][2] and (d := hamming(hashes[i], hashes[j])) <= max_distance
    ]
    return sorted(pairs, key=_pair_key)


def _pair_key(p: Pair) -> tuple[int, str, str]:
    return (p.distance, p.a[0], p.b[0])


def find_pairs(
    rows: Sequence[Row], hashes: Sequence[int], max_distance: int, top_k: int
) -> tuple[list[Pair], list[Pair], Counter[int]]:
    """Return (pairs within max_distance, top_k closest pairs, histogram of all distances)."""
    within: list[Pair] = []
    histogram: Counter[int] = Counter()

    def tracked() -> Iterator[Pair]:
        for pair in cross_split_pairs(rows, hashes):
            histogram[pair.distance] += 1
            if pair.distance <= max_distance:
                within.append(pair)
            yield pair

    top = heapq.nsmallest(top_k, tracked(), key=_pair_key)  # one pass; fills within/histogram
    return sorted(within, key=_pair_key), top, histogram


def summarise(
    histogram: Counter[int],
    within: Sequence[Pair],
    report_distances: Sequence[int],
    alert_distance: int,
) -> dict:
    """Cumulative counts (<= d) by same/different class, plus split pairs within the alert band."""
    cumulative = {}
    for d in report_distances:
        selected = [p for p in within if p.distance <= d]
        cumulative[str(d)] = {
            "pairs": sum(n for dist, n in histogram.items() if dist <= d),
            "same_class": sum(p.same_class for p in selected),
            "different_class": sum(not p.same_class for p in selected),
        }
    alert = [p for p in within if p.distance <= alert_distance]
    by_split_pair = Counter(" / ".join(sorted((p.a[2], p.b[2]))) for p in alert)
    return {
        "cross_split_pairs_compared": sum(histogram.values()),
        "min_distance": min(histogram) if histogram else None,
        "cumulative_counts_le_distance": cumulative,
        "alert_distance": alert_distance,
        "alert_pairs": len(alert),
        "alert_pairs_by_split_pair": dict(sorted(by_split_pair.items())),
    }


def write_pairs_csv(pairs: Sequence[Pair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(PAIR_FIELDS)
        for p in pairs:
            writer.writerow([p.distance, *p.a, *p.b, int(p.same_class)])


def draw_grid(pairs: Sequence[Pair], raw_dir: Path, path: Path, tile: int = 160) -> None:
    """Save one row per pair: image A | image B | label with distance, splits and classes."""
    label_w, pad = 360, 6
    width = 2 * tile + label_w + 4 * pad
    height = max(1, len(pairs)) * (tile + pad) + pad
    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)
    for i, p in enumerate(pairs):
        y = pad + i * (tile + pad)
        for k, row in enumerate((p.a, p.b)):
            with Image.open(raw_dir / row[0]) as img:
                canvas.paste(img.convert("L").resize((tile, tile)), (pad + k * (tile + pad), y))
        text = (
            f"d={p.distance}\n"
            f"A: {p.a[2]} / {p.a[1]}\n{Path(p.a[0]).name}\n"
            f"B: {p.b[2]} / {p.b[1]}\n{Path(p.b[0]).name}"
        )
        draw.multiline_text((3 * pad + 2 * tile, y + 4), text, fill=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report cross-split near-duplicates (pHash).")
    parser.add_argument("--config", required=True, type=Path, help="Path to params.yaml")
    args = parser.parse_args(argv)

    params = load_params(args.config)
    cfg = params["near_duplicates"]
    raw_dir = Path(params["data"]["raw_dir"])
    rows = read_split_csv(Path(params["data"]["split_file"]))
    hashes = [phash_int(raw_dir / r[0], int(cfg["hash_size"])) for r in rows]

    within, top, histogram = find_pairs(
        rows, hashes, int(cfg["max_distance"]), int(cfg["grid_top_k"])
    )
    alert_distance = int(cfg["alert_distance"])
    summary = summarise(histogram, within, cfg["report_distances"], alert_distance)
    summary["cross_split_distance_histogram"] = {str(d): n for d, n in sorted(histogram.items())}
    summary["within_split_alert_pairs"] = [
        dict(zip(PAIR_FIELDS, [p.distance, *p.a, *p.b, int(p.same_class)], strict=True))
        for p in within_split_close_pairs(rows, hashes, alert_distance)
    ]
    summary["params"] = {
        k: cfg[k] for k in ("hash_size", "max_distance", "report_distances", "alert_distance")
    }

    write_pairs_csv(within, Path(cfg["out_csv"]))
    out_summary = Path(cfg["out_summary"])
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    draw_grid(top, raw_dir, Path(cfg["out_grid"]))

    print(f"[near-dups] compared {summary['cross_split_pairs_compared']} cross-split pairs")
    for d, c in summary["cumulative_counts_le_distance"].items():
        print(f"[near-dups] <= {d}: {c}")
    print(f"[near-dups] alert (<= {alert_distance}): {summary['alert_pairs']} cross-split pairs")
    print(
        f"[near-dups] within-split pairs <= {alert_distance}: "
        f"{len(summary['within_split_alert_pairs'])} (no leakage; listed in the summary)"
    )
    print(f"[near-dups] wrote {cfg['out_csv']}, {cfg['out_summary']}, {cfg['out_grid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
