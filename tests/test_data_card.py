"""The data card's numbers must come from committed files, never be typed by hand.

Each generated block in docs/DATA_CARD.md sits between
`<!-- generated:<name> ... -->` and `<!-- /generated:<name> -->`. These tests re-render
every block from its source file and fail if the card differs. On failure the message
prints the expected block, ready to paste.
"""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from src.config import load_params
from src.data.split import read_split_csv

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
CARD = REPO_ROOT / "docs" / "DATA_CARD.md"
SPLIT_CSV = REPO_ROOT / PARAMS["data"]["split_file"]
SUMMARY_JSON = REPO_ROOT / PARAMS["near_duplicates"]["out_summary"]
NEAR_COPIES_JSON = REPO_ROOT / PARAMS["near_duplicates_cnn"]["pairs_summary"]
CNN_REPORT_JSON = REPO_ROOT / PARAMS["near_duplicates_cnn"]["out_summary"]
ARCHIVE_V1 = REPO_ROOT / "results" / "archive" / "split_v1"
CNN_REPORT_V1_JSON = ARCHIVE_V1 / "near_duplicates_cnn_summary.json"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render_split_counts(rows: list[tuple[str, str, str]], split_order: list[str]) -> str:
    """Images per split x class, with totals. Splits in params.yaml order."""
    counts = Counter((cls, split) for _, cls, split in rows)
    classes = sorted({cls for _, cls, _ in rows})
    body = []
    for split in split_order:
        per = [counts[(c, split)] for c in classes]
        body.append([f"`{split}`", *(f"{n:,}" for n in per), f"**{sum(per):,}**"])
    col_totals = [sum(counts[(c, s)] for s in split_order) for c in classes]
    body.append(["**total**", *(f"**{n:,}**" for n in col_totals), f"**{len(rows):,}**"])
    return _table(["split", *(f"`{c}`" for c in classes), "total"], body)


def render_uploader_folders(rows: list[tuple[str, str, str]]) -> str:
    """Images per class in the uploader's own train/validation folders (ignored by us)."""
    counts = Counter((path.split("/")[1], cls) for path, cls, _ in rows)
    folders = sorted({folder for folder, _ in counts})
    classes = sorted({cls for _, cls, _ in rows})
    body = [[f"`{f}`", *(f"{counts[(f, c)]:,}" for c in classes)] for f in folders]
    return _table(["uploader folder", *(f"`{c}`" for c in classes)], body)


def render_near_dups(summary: dict) -> str:
    """Key facts from the TASK-004 pHash report."""
    p = summary["params"]
    max_d = p["max_distance"]
    at_max = summary["cumulative_counts_le_distance"][str(max_d)]
    lines = [
        f"- pHash size: {p['hash_size'] ** 2} bits (`hash_size` = {p['hash_size']}).",
        f"- Cross-split pairs compared: {summary['cross_split_pairs_compared']:,}.",
        f"- Cross-split pairs with Hamming distance <= {max_d}: {at_max['pairs']:,}.",
        f"- Smallest cross-split distance: {summary['min_distance']}.",
        f"- Within-split pairs with distance <= {summary['alert_distance']}: "
        f"{len(summary['within_split_alert_pairs'])}.",
    ]
    for pair in summary["within_split_alert_pairs"]:
        cls = pair["class_a"] if pair["same_class"] else f"{pair['class_a']}`/`{pair['class_b']}"
        lines.append(
            f"  - `{pair['path_a']}` and `{pair['path_b']}`: distance {pair['distance']}, "
            f"split `{pair['split_a']}`, class `{cls}`."
        )
    return "\n".join(lines)


def render_near_copies(s: dict) -> str:
    """Key facts from the TASK-008b pairs step (all image pairs, split-independent)."""
    r, cal = s["rule"], s["calibration"]
    pix = cal["pixel_corr_random_same_class"]
    shift16 = cal["cosine_shifted_crops"]["by_shift_px"]["16"]["same_image"]
    g = s["groups"]
    (w,) = s["watch_pairs"]
    by_class = ", ".join(f"`{c}` {n}" for c, n in g["images_by_class"].items())
    return "\n".join(
        [
            f"- Image pairs compared (any split): {s['pairs_compared']:,}.",
            f"- CNN candidates (cosine >= {r['candidate_cosine_ge']:g}): {s['candidates']:,}. "
            f"Synthetic copies shifted 16 px: cosine p5 {shift16['p5']}, p1 {shift16['p1']}.",
            f"- Pixel correlation of {pix['pairs']:,} random same-class pairs: "
            f"median {pix['p50']}, p95 {pix['p95']}, p99 {pix['p99']}.",
            f"- Confirmed near-copies (pixel correlation >= {r['confirm_pixel_corr_ge']:g}, shifts "
            f"up to {r['max_shift_px']} px): {s['confirmed_pairs']} pairs, "
            f"{len(s['confirmed_different_class'])} between different classes.",
            f"- Groups: {g['count']} groups, {g['images']} images, largest {g['largest']['size']} "
            f"images. Images in groups by class: {by_class}.",
            f"- `{Path(w['path_a']).name}` / `{Path(w['path_b']).name}` (suspected in TASK-004): "
            f"cosine {w['cosine']} (rank {w['cosine_rank_among_all_pairs']:,}), pixel correlation "
            f"{w['pixel_corr']}, confirmed: {'yes' if w['confirmed'] else 'no'}.",
        ]
    )


def render_near_copies_by_split(v1: dict, v2: dict) -> str:
    """Confirmed near-copies across splits: v1 (archive) vs the current split."""
    a, b = v1["confirmed_cross_split_by_split_pair"], v2["confirmed_cross_split_by_split_pair"]
    body = [[f"`{k}`", str(a.get(k, 0)), str(b.get(k, 0))] for k in sorted({*a, *b})]
    body.append(
        ["**total**", f"**{v1['confirmed_cross_split']}**", f"**{v2['confirmed_cross_split']}**"]
    )
    header = ["split pair", f"`{Path(v1['split_file']).name}`", f"`{Path(v2['split_file']).name}`"]
    return _table(header, body)


def expected_blocks() -> dict[str, str]:
    rows = read_split_csv(SPLIT_CSV)
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    split_order = list(PARAMS["split"]["per_class"])
    return {
        "split-counts": render_split_counts(rows, split_order),
        "uploader-folders": render_uploader_folders(rows),
        "near-dups": render_near_dups(summary),
        "near-copies": render_near_copies(_load(NEAR_COPIES_JSON)),
        "near-copies-by-split": render_near_copies_by_split(
            _load(CNN_REPORT_V1_JSON), _load(CNN_REPORT_JSON)
        ),
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def card_blocks(text: str) -> dict[str, str]:
    pattern = re.compile(
        r"<!-- generated:(?P<name>[\w-]+)[^>]*-->\n(?P<body>.*?)\n<!-- /generated:(?P=name) -->",
        re.DOTALL,
    )
    return {m["name"]: m["body"] for m in pattern.finditer(text)}


def test_card_has_every_generated_block() -> None:
    found = card_blocks(CARD.read_text(encoding="utf-8"))
    assert set(found) == set(expected_blocks())


@pytest.mark.parametrize("name", sorted(expected_blocks()))
def test_generated_block_matches_committed_files(name: str) -> None:
    found = card_blocks(CARD.read_text(encoding="utf-8"))
    expected = expected_blocks()[name]
    assert found.get(name) == expected, f"Regenerate block '{name}' as:\n\n{expected}\n"


def test_renderers_on_fake_rows() -> None:
    rows = [
        ("NEU-DET/train/images/a/a_1.jpg", "a", "s1"),
        ("NEU-DET/train/images/a/a_2.jpg", "a", "s2"),
        ("NEU-DET/validation/images/b/b_1.jpg", "b", "s2"),
    ]
    split_table = render_split_counts(rows, ["s1", "s2"])
    assert "| `s1` | 1 | 0 | **1** |" in split_table
    assert "| `s2` | 1 | 1 | **2** |" in split_table
    assert "| **total** | **2** | **1** | **3** |" in split_table
    folders = render_uploader_folders(rows)
    assert "| `train` | 2 | 0 |" in folders
    assert "| `validation` | 0 | 1 |" in folders


@pytest.mark.needs_data
def test_image_format_claim_in_card() -> None:
    """The card says every image is a 200x200 JPEG with 3 channels in the file."""
    from PIL import Image

    raw_dir = REPO_ROOT / PARAMS["data"]["raw_dir"]
    for path, _, _ in read_split_csv(SPLIT_CSV):
        with Image.open(raw_dir / path) as img:
            assert (img.format, img.size, img.mode) == ("JPEG", (200, 200), "RGB"), path
