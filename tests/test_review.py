"""TASK-024 (ADR-031): archive lookup by hash, the review queue and feedback.
Uses a tiny generated archive (no NEU pixels) and a temp SQLite log."""

from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import LargeBinary  # noqa: E402

from src.config import load_params  # noqa: E402
from src.db import migrate  # noqa: E402
from src.db.models import Feedback  # noqa: E402
from src.db.store import PredictionLog  # noqa: E402
from src.review import queue as rq  # noqa: E402
from src.review.archive import ImageArchive  # noqa: E402
from tests import review_helpers as rh  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
CLASSES = rh.CLASSES


@pytest.fixture
def db(tmp_path):
    log = PredictionLog("sqlite:///" + (tmp_path / "db" / "p.db").as_posix())
    migrate.upgrade(log.engine)
    yield log
    log.dispose()


@pytest.fixture
def archive(tmp_path):
    items = [
        ("crazing", "production_stream", 1),
        ("inclusion", "production_stream", 2),
        ("patches", "test_frozen", 3),
        ("scratches", "production_stream", 4),
    ]
    split, raw, shas = rh.make_archive(tmp_path / "archive", items)
    return ImageArchive.from_split_file(split, raw), shas


def rid(i: int) -> str:
    return f"{i:032x}"


# ---------------------------------------------------------------- archive


def test_archive_finds_images_by_hash(archive):
    arc, shas = archive
    entry = arc.lookup(shas["crazing_0"])
    assert entry.label == "crazing" and entry.split == "production_stream" and entry.path.is_file()
    assert arc.in_split(shas["patches_2"], "test_frozen")
    assert arc.lookup("0" * 64) is None and len(arc) == 4


def test_archive_flags_identical_bytes_with_different_classes(tmp_path):
    split, raw, shas = rh.make_archive(
        tmp_path, [("crazing", "val", 7), ("inclusion", "production_stream", 7)]
    )
    arc = ImageArchive.from_split_file(split, raw)
    assert shas["crazing_0"] == shas["inclusion_1"]
    assert arc.lookup(shas["crazing_0"]) is None and shas["crazing_0"] in arc.ambiguous


def test_identical_copies_count_as_test_frozen_if_any_copy_is(tmp_path):
    split, raw, shas = rh.make_archive(
        tmp_path, [("patches", "production_stream", 9), ("patches", "test_frozen", 9)]
    )
    arc = ImageArchive.from_split_file(split, raw)
    assert arc.in_split(shas["patches_0"], "test_frozen")


def test_archive_skips_missing_files(tmp_path):
    split, raw, _ = rh.make_archive(tmp_path, [("crazing", "val", 1)])
    for p in raw.rglob("*.png"):
        p.unlink()
    assert len(ImageArchive.from_split_file(split, raw)) == 0


# ---------------------------------------------------------------- queue and feedback


def test_feedback_table_holds_no_image_data():
    cols = Feedback.__table__.columns
    assert not [c.key for c in cols if isinstance(c.type, LargeBinary)]
    assert {c.key for c in cols} == {
        "id",
        "request_id",
        "label",
        "reviewer_kind",
        "reviewer_id",
        "created_at",
        "simulated_error_rate",
        "simulated_flipped",
    }


def test_queue_is_routed_unlabelled_oldest_first(db):
    db.add(rh.prediction(rid(1)))  # accepted: never queued
    db.add(rh.prediction(rid(2), reasons=("low_confidence",)))
    db.add(rh.prediction(rid(3), reasons=("audit",)))
    db.add(rh.prediction(rid(4), reasons=("low_confidence", "ood")))
    assert [r["request_id"] for r in db.review_queue()] == [rid(2), rid(3), rid(4)]
    assert db.review_queue_size() == 3
    assert [r["request_id"] for r in db.review_queue(reason="audit")] == [rid(3)]
    assert db.review_queue_size("ood") == 1
    rq.submit(db, rid(2), "crazing", CLASSES, reviewer_kind="human", reviewer_id="ana")
    assert [r["request_id"] for r in db.review_queue()] == [rid(3), rid(4)]
    assert db.review_queue_size() == 2  # the gauge: routed and not yet labelled


def test_submit_validates(db):
    db.add(rh.prediction(rid(1)))
    db.add(rh.prediction(rid(2), reasons=("ood",)))
    human = {"reviewer_kind": "human", "reviewer_id": "ana"}
    with pytest.raises(rq.ReviewError, match="label must be"):
        rq.submit(db, rid(2), "cat", CLASSES, **human)
    with pytest.raises(rq.ReviewError, match="not routed"):
        rq.submit(db, rid(1), "crazing", CLASSES, **human)
    with pytest.raises(rq.ReviewError, match="no prediction"):
        rq.submit(db, rid(9), "crazing", CLASSES, **human)
    with pytest.raises(rq.ReviewError, match="reviewer_kind"):
        rq.submit(db, rid(2), "crazing", CLASSES, reviewer_kind="robot", reviewer_id="x")
    rq.submit(db, rid(2), rq.UNKNOWN, CLASSES, **human)  # "unknown / not steel" is allowed
    with pytest.raises(rq.ReviewError, match="already"):
        rq.submit(db, rid(2), "crazing", CLASSES, **human)
    fb = db.feedback_for(rid(2))
    assert fb["label"] == "unknown" and fb["reviewer_kind"] == "human"
    assert fb["simulated_flipped"] is None and fb["created_at"].tzinfo is not None


def test_class_names_come_from_the_split_file():
    assert rq.class_names(PARAMS) == CLASSES
    assert rq.labels(CLASSES)[-1] == "unknown"
