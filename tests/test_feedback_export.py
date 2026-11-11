"""TASK-026 (ADR-032): the feedback export. A tiny generated archive (no NEU pixels), a temp SQLite
log, and one test per exclusion rule."""

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import event  # noqa: E402

from src.config import load_params  # noqa: E402
from src.db.store import PredictionLog  # noqa: E402
from src.review import export as ex  # noqa: E402
from src.review.archive import ImageArchive  # noqa: E402
from src.review.queue import submit  # noqa: E402
from tests import review_helpers as rh  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
CLASSES = rh.CLASSES

ITEMS = [
    ("crazing", "production_stream", 1),  # crazing_0
    ("inclusion", "production_stream", 2),  # inclusion_1
    ("patches", "test_frozen", 3),  # patches_2
    ("scratches", "val", 4),  # scratches_3
    ("pitted_surface", "train_v1", 5),  # pitted_surface_4
    ("rolled-in_scale", "production_stream", 6),  # rolled-in_scale_5
]


@pytest.fixture
def env(tmp_path):
    split, raw, shas = rh.make_archive(tmp_path / "archive", ITEMS)
    db = PredictionLog("sqlite:///" + (tmp_path / "db" / "p.db").as_posix())
    db.create_tables()
    yield db, ImageArchive.from_split_file(split, raw), raw, shas, tmp_path
    db.dispose()


def rid(i: int) -> str:
    return f"{i:032x}"


def label(db, i, sha, value, kind="simulated", **sim):
    db.add(rh.prediction(rid(i), sha, reasons=("low_confidence",)))
    submit(db, rid(i), value, CLASSES, reviewer_kind=kind, reviewer_id="r", **sim)


def run_export(env):
    db, archive, raw, _, tmp = env
    report = ex.export(
        db, archive, CLASSES, raw, tmp / "out", tmp / "results" / "report.json", "split.csv"
    )
    with (tmp / "out" / ex.CSV_NAME).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return report, rows


def test_exports_production_stream_labels_as_path_label_rows(env):
    db, _, raw, shas, tmp = env
    label(db, 1, shas["crazing_0"], "crazing")
    label(db, 2, shas["inclusion_1"], "inclusion", kind="human")
    report, rows = run_export(env)
    assert rows == [
        {"path": "crazing/crazing_0.png", "label": "crazing", "reviewer_kind": "simulated",
         "request_id": rid(1)},
        {"path": "inclusion/inclusion_1.png", "label": "inclusion", "reviewer_kind": "human",
         "request_id": rid(2)},
    ]  # fmt: skip
    assert all((raw / r["path"]).is_file() for r in rows)  # paths are relative to data/raw
    assert report["rows"] == 2 and report["label_rows_read"] == 2
    assert report["per_class"] == {c: int(c in ("crazing", "inclusion")) for c in CLASSES}
    assert report["per_reviewer_kind"] == {"human": 1, "simulated": 1}
    assert set(report["excluded"].values()) == {0}
    assert "SIMULATED" in report["note"]
    csv_bytes = (tmp / "out" / ex.CSV_NAME).read_bytes()
    assert report["csv_md5"] == hashlib.md5(csv_bytes).hexdigest()
    assert b"\r\n" not in csv_bytes
    saved = json.loads((tmp / "results" / "report.json").read_text(encoding="utf-8"))
    assert saved == report


def test_no_image_bytes_are_copied(env):
    db, _, _, shas, tmp = env
    label(db, 1, shas["crazing_0"], "crazing")
    run_export(env)
    assert [p.name for p in (tmp / "out").iterdir()] == [ex.CSV_NAME]
    header = (tmp / "out" / ex.CSV_NAME).read_text(encoding="utf-8").splitlines()[0]
    assert header == "path,label,reviewer_kind,request_id"


@pytest.mark.parametrize(
    ("image", "reason"),
    [
        ("patches_2", "test_frozen"),
        ("scratches_3", "val"),
        ("pitted_surface_4", "other_split"),
    ],
)
def test_never_exports_images_from_other_splits(env, image, reason):
    db, _, _, shas, _ = env
    label(db, 1, shas[image], image.rsplit("_", 1)[0])
    report, rows = run_export(env)
    assert rows == [] and report["rows"] == 0
    assert report["excluded"][reason] == 1


def test_excludes_hashes_not_in_the_archive(env):
    db, *_ = env
    label(db, 1, "0" * 64, "crazing")
    report, rows = run_export(env)
    assert rows == [] and report["excluded"]["not_in_archive"] == 1


def test_excludes_hashes_shared_by_two_classes(tmp_path):
    # the same bytes filed under two classes: no unique ground truth (ADR-031)
    split, raw, shas = rh.make_archive(
        tmp_path / "a",
        [
            ("crazing", "production_stream", 9),
            ("inclusion", "production_stream", 9),
            ("patches", "production_stream", 10),
        ],
    )
    db = PredictionLog("sqlite:///" + (tmp_path / "p.db").as_posix())
    db.create_tables()
    label(db, 1, shas["crazing_0"], "crazing")
    archive = ImageArchive.from_split_file(split, raw)
    assert sha_ambiguous(archive, shas["crazing_0"])
    try:
        report = ex.export(db, archive, CLASSES, raw, tmp_path / "o", tmp_path / "r.json", "s")
    finally:
        db.dispose()
    assert report["rows"] == 0 and report["excluded"]["not_in_archive"] == 1


def sha_ambiguous(archive, sha):
    return sha in archive.ambiguous and archive.lookup(sha) is None


def test_excludes_unknown_labels(env):
    db, _, _, shas, _ = env
    label(db, 1, shas["crazing_0"], "unknown", kind="human")
    label(db, 2, shas["inclusion_1"], "inclusion")
    report, rows = run_export(env)
    assert [r["label"] for r in rows] == ["inclusion"]
    assert report["excluded"]["unknown_label"] == 1


def test_unlabelled_predictions_are_not_exported(env):
    db, _, _, shas, _ = env
    db.add(rh.prediction(rid(1), shas["crazing_0"], reasons=("ood",)))  # routed, no label yet
    db.add(rh.prediction(rid(2), shas["inclusion_1"]))  # auto-accepted
    report, rows = run_export(env)
    assert rows == [] and report["label_rows_read"] == 0


def test_repeated_image_agreeing_labels_kept_once_conflicting_dropped(env):
    db, _, _, shas, _ = env
    label(db, 1, shas["crazing_0"], "crazing")
    label(db, 2, shas["crazing_0"], "crazing")  # same image again, same label
    label(db, 3, shas["inclusion_1"], "inclusion")
    label(db, 4, shas["inclusion_1"], "scratches")  # same image, different label
    report, rows = run_export(env)
    assert rows == [
        {"path": "crazing/crazing_0.png", "label": "crazing", "reviewer_kind": "simulated",
         "request_id": rid(2)},
    ]  # fmt: skip
    assert report["excluded"]["repeated_image"] == 1
    assert report["excluded"]["conflicting_labels"] == 2
    assert report["rows"] + sum(report["excluded"].values()) == report["label_rows_read"] == 4


def test_never_reads_the_simulated_noise_columns(env):
    db, _, _, shas, _ = env
    label(db, 1, shas["crazing_0"], "inclusion", simulated_error_rate=1.0, simulated_flipped=True)
    statements = []

    def capture(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(db.engine, "before_cursor_execute", capture)
    try:
        report, rows = run_export(env)
    finally:
        event.remove(db.engine, "before_cursor_execute", capture)
    assert statements, "the export ran no SQL"
    assert not any("simulated_flipped" in s or "simulated_error_rate" in s for s in statements)
    # a flipped (wrong) simulated label is exported like any other: retraining sees reviewer noise
    assert rows[0]["label"] == "inclusion" and set(rows[0]) == set(ex.COLUMNS)
    assert report["rows"] == 1


def test_refuses_to_export_without_the_archive(env, tmp_path):
    db, *_ = env
    empty = ImageArchive({}, set())
    with pytest.raises(ex.ExportError, match="no archive images"):
        ex.export(db, empty, CLASSES, "raw", tmp_path / "o", tmp_path / "r.json", "s")
    assert not (tmp_path / "o").exists()  # a good earlier export is never overwritten


def test_counts_must_add_up():
    with pytest.raises(ex.ExportError, match="add up"):
        ex.build_report([], ex.Counter(), 1, CLASSES, "c", "m", "s")


def test_main_uses_params_and_database_url(env, monkeypatch, capsys):
    db, _, raw, shas, tmp = env
    label(db, 1, shas["rolled-in_scale_5"], "rolled-in_scale")
    params = copy.deepcopy(PARAMS)
    params["data"]["split_file"] = str(tmp / "archive" / "split.csv")
    params["data"]["raw_dir"] = str(raw)
    params["review"]["export"] = {"out_dir": str(tmp / "fb"), "report": str(tmp / "rep.json")}
    monkeypatch.setattr(ex, "load_params", lambda _: params)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + (tmp / "db" / "p.db").as_posix())
    ex.main([])
    assert "SIMULATED" in capsys.readouterr().out
    assert json.loads((tmp / "rep.json").read_text(encoding="utf-8"))["rows"] == 1
    assert (tmp / "fb" / ex.CSV_NAME).is_file()


def test_export_config_paths():
    cfg = PARAMS["review"]["export"]
    assert cfg["out_dir"] == "data/feedback" and cfg["report"].startswith("results/feedback/")
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/data/feedback/" in ignore  # the CSV goes to DVC, never to git
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "dvc add data/feedback" in makefile
