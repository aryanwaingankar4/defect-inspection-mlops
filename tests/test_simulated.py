"""TASK-025 (ADR-031): the simulated reviewer and the simulated stream.
Uses a tiny generated archive (no NEU pixels) and a temp SQLite log."""

import copy
import random
import uuid
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

from src.config import load_params  # noqa: E402
from src.db import migrate  # noqa: E402
from src.db.store import PredictionLog  # noqa: E402
from src.pipelines import stream as st  # noqa: E402
from src.review import simulated as sim  # noqa: E402
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


# ---------------------------------------------------------------- simulated reviewer


def test_simulated_label_is_deterministic_and_never_flips_to_the_truth():
    for i in range(500):
        a = sim.simulated_label("crazing", CLASSES, 0.5, 31, rid(i))
        assert a == sim.simulated_label("crazing", CLASSES, 0.5, 31, rid(i))
        assert a[1] == (a[0] != "crazing")
    assert sim.simulated_label("crazing", CLASSES, 0.0, 31, rid(1)) == ("crazing", False)
    assert sim.simulated_label("crazing", CLASSES, 1.0, 31, rid(1))[1] is True


def test_simulated_error_rate_over_many_items():
    rng = random.Random(0)
    ids = [uuid.UUID(int=rng.getrandbits(128), version=4).hex for _ in range(20_000)]
    for rate in (0.05, 0.15):
        flips = sum(sim.simulated_label("patches", CLASSES, rate, 31, r)[1] for r in ids)
        sd = (rate * (1 - rate) / len(ids)) ** 0.5
        assert abs(flips / len(ids) - rate) < 4 * sd, rate


def test_wrong_labels_are_spread_over_the_other_classes():
    wrong = [sim.simulated_label("patches", CLASSES, 1.0, 31, rid(i))[0] for i in range(5000)]
    counts = {c: wrong.count(c) for c in CLASSES}
    assert counts["patches"] == 0
    assert all(800 < counts[c] < 1200 for c in CLASSES if c != "patches")  # about 1000 each


def test_run_labels_with_truth_and_skips_test_frozen_and_unknown(db, archive):
    arc, shas = archive
    db.add(rh.prediction(rid(1), shas["crazing_0"], reasons=("low_confidence",)))
    db.add(rh.prediction(rid(2), shas["patches_2"], reasons=("ood",)))  # test_frozen
    db.add(rh.prediction(rid(3), "0" * 64, reasons=("ood",)))  # not in the archive
    db.add(rh.prediction(rid(4), shas["scratches_3"], reasons=("audit",)))
    db.add(rh.prediction(rid(5), shas["inclusion_1"]))  # accepted: not in the queue
    cfg = sim.SimConfig(error_rate=0.0, seed=31, reviewer_id="sim", limit=100)
    out = sim.run(db, arc, CLASSES, cfg)
    assert out["labelled"] == 2 and out["flipped"] == 0
    assert out["skipped_test_frozen"] == 1 and out["skipped_no_ground_truth"] == 1
    assert out["queue_left"] == 2 and "simulated" in out["reviewer"]
    assert db.feedback_for(rid(1))["label"] == "crazing"
    fb = db.feedback_for(rid(4))
    assert fb["label"] == "scratches" and fb["reviewer_kind"] == "simulated"
    assert fb["simulated_error_rate"] == 0.0 and fb["simulated_flipped"] is False
    assert db.feedback_for(rid(2)) is None  # a test image is never labelled
    assert sim.run(db, arc, CLASSES, cfg)["labelled"] == 0  # rerun: nothing new


def test_run_refuses_every_split_but_production_stream(tmp_path, db):
    items = [
        ("crazing", "production_stream", 1),
        ("inclusion", "val", 2),
        ("patches", "train_v1", 3),
        ("scratches", "test_frozen", 4),
    ]
    split, raw, shas = rh.make_archive(tmp_path / "splits", items)
    arc = ImageArchive.from_split_file(split, raw)
    for i, name in enumerate(shas, start=1):
        db.add(rh.prediction(rid(i), shas[name], reasons=("low_confidence",)))
    cfg = sim.SimConfig(error_rate=0.0, seed=31, reviewer_id="sim", limit=100)
    out = sim.run(db, arc, CLASSES, cfg)
    assert out["labelled"] == 1 and db.feedback_for(rid(1))["label"] == "crazing"
    assert out["skipped_not_production_stream"] == 2  # val + train_v1
    assert out["skipped_test_frozen"] == 1
    assert [db.feedback_for(rid(i)) for i in (2, 3, 4)] == [None, None, None]
    assert out["queue_left"] == 3


def test_run_respects_the_limit_and_is_order_independent(tmp_path, archive):
    arc, shas = archive
    cfg = sim.SimConfig(error_rate=0.5, seed=31, reviewer_id="sim", limit=100)
    results = []
    for name, order in (("a", range(1, 41)), ("b", range(40, 0, -1))):
        log = PredictionLog("sqlite:///" + (tmp_path / f"{name}.db").as_posix())
        migrate.upgrade(log.engine)
        for i in order:
            log.add(rh.prediction(rid(i), shas["crazing_0"], reasons=("low_confidence",)))
        sim.run(log, arc, CLASSES, cfg)
        results.append({rid(i): log.feedback_for(rid(i))["label"] for i in range(1, 41)})
        log.dispose()
    assert results[0] == results[1]  # same labels, whatever the order
    log = PredictionLog("sqlite:///" + (tmp_path / "c.db").as_posix())
    migrate.upgrade(log.engine)
    for i in range(1, 11):
        log.add(rh.prediction(rid(i), shas["crazing_0"], reasons=("ood",)))
    out = sim.run(log, arc, CLASSES, sim.SimConfig(0.0, 31, "sim", limit=3))
    assert out["labelled"] == 3 and out["queue_left"] == 7
    log.dispose()


@pytest.mark.parametrize("bad", [{"error_rate": 1.5}, {"limit": 0}])
def test_sim_config_validates(bad):
    with pytest.raises(ValueError):
        sim.SimConfig.from_params(PARAMS, **bad)


def test_params_default_error_rate_is_zero():
    assert sim.SimConfig.from_params(PARAMS).error_rate == 0.0


# ---------------------------------------------------------------- simulated stream


def stream_params(tmp_path):
    items = [("crazing", "production_stream", i) for i in range(5)]
    items += [("patches", "test_frozen", 10), ("inclusion", "val", 11)]
    split, raw, _ = rh.make_archive(tmp_path / "s", items)
    p = copy.deepcopy(PARAMS)
    p["data"]["split_file"], p["data"]["raw_dir"] = str(split), str(raw)
    return p


def test_stream_sends_only_production_stream_in_a_fixed_order(tmp_path, monkeypatch):
    params = stream_params(tmp_path)
    order = st.stream_order(params)
    assert order == st.stream_order(params) and len(order) == 5
    assert {r["split"] for r in order} == {"production_stream"}
    sent = []

    def fake_post(url, key, data, name):
        sent.append(data)
        return 200, {"reasons": ["low_confidence"]}

    monkeypatch.setattr(st, "post_image", fake_post)
    out = st.send(params, "http://x", "k", n=3, start=1)
    raw = Path(params["data"]["raw_dir"])
    assert sent == [(raw / r["path"]).read_bytes() for r in order[1:4]]
    assert out["sent"] == 3 and out["statuses"] == {200: 3} and "simulated" in out["stream"]


def test_stream_refuses_other_splits(tmp_path):
    params = stream_params(tmp_path)
    for split in ("test_frozen", "val", "train_v1"):
        params["review"]["stream"]["split"] = split
        with pytest.raises(st.StreamError, match="only sends"):
            st.stream_order(params)
