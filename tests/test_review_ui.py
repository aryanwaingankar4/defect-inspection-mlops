"""TASK-024 (ADR-031): the Streamlit review UI, run headless with Streamlit's AppTest."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("sqlalchemy")

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.db import migrate  # noqa: E402
from src.db.store import PredictionLog  # noqa: E402
from tests import review_helpers as rh  # noqa: E402
from tests.db_helpers import migrated  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "src" / "review" / "app.py"
PASSWORD = "test-reviewer-password"


@pytest.fixture
def db_url(tmp_path, monkeypatch):
    url = "sqlite:///" + (tmp_path / "db" / "review.db").as_posix()
    url = migrated(url)  # what `make db-migrate` does (ADR-038)
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("REVIEWER_PASSWORD", PASSWORD)
    log = PredictionLog(url)
    migrate.upgrade(log.engine)
    yield url
    log.dispose()


def add(url: str, *rows) -> None:
    log = PredictionLog(url)
    for row in rows:
        log.add(row)
    log.dispose()


def logged_in() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.text_input(key="password").input(PASSWORD).run()
    assert not at.exception, at.exception
    return at


def text_of(at: AppTest) -> str:
    return "\n".join(
        [m.value for m in at.markdown]
        + [w.value for w in at.warning]
        + [i.value for i in at.info]
        + [s.value for s in at.success]
        + [e.value for e in at.error]
    )


def test_refuses_to_start_without_a_password(db_url, monkeypatch):
    monkeypatch.delenv("REVIEWER_PASSWORD")
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert "REVIEWER_PASSWORD is not set" in text_of(at)
    assert not at.button


def test_wrong_password_shows_nothing(db_url):
    add(db_url, rh.prediction("a" * 32, reasons=("low_confidence",)))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.text_input(key="password").input("wrong").run()
    assert "Wrong password" in text_of(at)
    assert "Model prediction" not in text_of(at) and not at.button


def test_low_confidence_card_shows_the_model_answer_and_labels(db_url):
    add(db_url, rh.prediction("a" * 32, reasons=("low_confidence", "ood"), prediction="inclusion"))
    at = logged_in()
    text = text_of(at)
    assert "Simulated environment" in text
    assert "low confidence" in text and "OOD" in text
    assert "**Model prediction:** inclusion" in text and "Calibrated confidence" in text
    assert "Image not in the local archive" in text  # only the hash is logged
    keys = {b.key for b in at.button}
    assert {f"label-{c}" for c in rh.CLASSES} | {"label-unknown"} <= keys
    at.button(key="label-scratches").click().run()
    fb = PredictionLog(db_url).feedback_for("a" * 32)
    assert fb["label"] == "scratches" and fb["reviewer_kind"] == "human"
    assert fb["reviewer_id"] == "reviewer-1"
    assert "The review queue is empty" in text_of(at)


def test_audit_card_is_blind_until_the_label_is_submitted(db_url):
    add(db_url, rh.prediction("b" * 32, reasons=("audit",), prediction="patches"))
    at = logged_in()
    text = text_of(at)
    assert "random audit" in text and "Blind audit" in text
    assert "Model prediction" not in text and "patches (" not in text
    assert "Calibrated confidence" not in text and "Energy" not in text
    at.button(key="label-crazing").click().run()
    text = text_of(at)
    assert "**Your label:** crazing" in text and "**Model prediction:** patches" in text
    assert "disagrees" in text
    assert PredictionLog(db_url).feedback_for("b" * 32)["label"] == "crazing"
    at.button(key="next").click().run()
    assert "The review queue is empty" in text_of(at)


def test_cards_come_oldest_first_and_the_filter_works(db_url):
    add(
        db_url,
        rh.prediction("1" * 32, reasons=("low_confidence",), prediction="crazing"),
        rh.prediction("2" * 32, reasons=("ood",), prediction="patches"),
    )
    at = logged_in()
    assert "**Model prediction:** crazing" in text_of(at)
    at.selectbox(key="reason").select("ood").run()
    assert "**Model prediction:** patches" in text_of(at)
    assert at.sidebar.metric[0].value == "1"
