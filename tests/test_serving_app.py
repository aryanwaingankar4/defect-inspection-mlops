"""TASK-018: FastAPI service (ADR-028). A tiny hand-built ONNX bundle; no network, no NEU data."""

import ast
import copy
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.config import load_params  # noqa: E402
from src.db.store import PredictionLog  # noqa: E402
from src.serving import app as api  # noqa: E402
from src.serving import bundle as bd  # noqa: E402
from src.serving import scoring  # noqa: E402
from src.serving.features import image_features  # noqa: E402
from src.serving.json_logging import JsonFormatter  # noqa: E402
from src.serving.onnx_model import OnnxClassifier  # noqa: E402
from src.serving.preprocess import PreprocessConfig, load_image, to_model_input  # noqa: E402
from tests import serving_helpers as sh  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
KEY = "test-key-0123456789"
DARK, MID = 25.0, 124.0  # mean grey levels: confident class vs unsure (both reasons)


def params_with(**serving) -> dict:
    p = copy.deepcopy(PARAMS)
    p["serving"].update(serving)
    return p


@pytest.fixture(autouse=True)
def database_url(tmp_path, monkeypatch):
    """A fresh SQLite prediction log per test (ADR-029); tests never read .env."""
    url = "sqlite:///" + (tmp_path / "db" / "predictions.db").as_posix()
    monkeypatch.setenv("DATABASE_URL", url)
    return url


@pytest.fixture
def bundle_dir(tmp_path):
    return sh.make_bundle(tmp_path / "bundle", PARAMS["preprocess"])


@pytest.fixture
def client(bundle_dir):
    app = api.create_app(PARAMS, api_key=KEY, bundle_dir=bundle_dir)
    with TestClient(app) as c:
        yield c


def post_image(c, data, key=KEY, name="img.png"):
    headers = {} if key is None else {"X-API-Key": key}
    return c.post("/predict", files={"file": (name, data, "image/png")}, headers=headers)


def assert_error(r, status, code):
    assert r.status_code == status, r.text
    body = r.json()
    assert body["error"] == code and body["detail"]
    assert body["request_id"] == r.headers["X-Request-ID"]


# ---------------------------------------------------------------- startup


def test_refuses_to_start_without_an_api_key(bundle_dir, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API_KEY"):
        api.create_app(PARAMS, bundle_dir=bundle_dir)
    with pytest.raises(RuntimeError, match="API_KEY"):
        api.create_app(PARAMS, api_key="  ", bundle_dir=bundle_dir)


def test_api_key_comes_from_the_environment(bundle_dir, monkeypatch):
    monkeypatch.setenv("API_KEY", "from-env")
    with TestClient(api.create_app(PARAMS, bundle_dir=bundle_dir)) as c:
        assert post_image(c, sh.image_bytes(DARK), key="from-env").status_code == 200
        assert post_image(c, sh.image_bytes(DARK), key=KEY).status_code == 403


def test_health_is_503_until_the_bundle_is_loaded(bundle_dir):
    app = api.create_app(PARAMS, api_key=KEY, bundle_dir=bundle_dir)
    c = TestClient(app)  # no context manager: startup has not run
    assert c.get("/health").status_code == 503
    assert_error(post_image(c, sh.image_bytes(DARK)), 503, "not_ready")


def test_health_reports_the_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model"] == {
        "name": "steel-defect-classifier",
        "version": "7",
        "alias": "champion",
        "precision": "fp32",
    }


@pytest.mark.parametrize(
    "breakage",
    ["changed_onnx", "classes", "version", "preprocess_params", "temperature", "no_manifest"],
)
def test_refuses_to_start_on_a_bad_bundle(bundle_dir, breakage):
    params = copy.deepcopy(PARAMS)
    manifest_path = bundle_dir / bd.MANIFEST_NAME
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if breakage == "changed_onnx":
        onnx_path = bundle_dir / m["onnx_file"]
        onnx_path.write_bytes(onnx_path.read_bytes() + b"\0")
    elif breakage == "classes":
        m["classes"] = list(reversed(m["classes"]))
    elif breakage == "version":
        m["version"] = "8"
    elif breakage == "preprocess_params":
        params["preprocess"]["image_size"] = 256
    elif breakage == "temperature":
        m["temperature"] = 0.0
    if breakage == "no_manifest":
        manifest_path.unlink()
    else:
        manifest_path.write_bytes(json.dumps(m).encode())
    app = api.create_app(params, api_key=KEY, bundle_dir=bundle_dir)
    with pytest.raises(bd.BundleError), TestClient(app):
        pass


# ---------------------------------------------------------------- auth


def test_predict_needs_the_api_key(client):
    assert_error(post_image(client, sh.image_bytes(DARK), key=None), 401, "missing_api_key")
    assert_error(post_image(client, sh.image_bytes(DARK), key="wrong"), 403, "invalid_api_key")


def test_feedback_needs_the_api_key(client):
    body = {"request_id": "a" * 32, "label": "crazing"}
    assert_error(client.post("/feedback", json=body), 401, "missing_api_key")
    r = client.post("/feedback", json=body, headers={"X-API-Key": "x"})
    assert_error(r, 403, "invalid_api_key")


def test_health_and_metrics_are_open(client):
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200


# ---------------------------------------------------------------- predict


def logits_for(bundle_dir: Path, data: bytes) -> np.ndarray:
    """Logits from the bundle's ONNX file on the shared preprocessing, outside the app."""
    pre = PreprocessConfig.from_params(PARAMS)
    onnx_path = next(bundle_dir.glob("*.onnx"))
    return OnnxClassifier(onnx_path).logits(to_model_input(load_image(data), pre)[None])


def test_predict_applies_temperature_and_scores(client, bundle_dir):
    data = sh.image_bytes(DARK)
    r = post_image(client, data)
    assert r.status_code == 200, r.text
    body = r.json()
    logits = logits_for(bundle_dir, data)
    calibrated = scoring.calibrated_probs(logits, sh.TEMPERATURE)[0]
    raw = scoring.calibrated_probs(logits, 1.0)[0]
    assert body["prediction"] == sh.CLASSES[int(calibrated.argmax())] == "rolled-in_scale"
    assert body["calibrated_confidence"] == pytest.approx(float(calibrated.max()), abs=1e-6)
    assert abs(body["calibrated_confidence"] - float(raw.max())) > 1e-3  # T is applied (ADR-025)
    assert body["energy"] == pytest.approx(
        float(scoring.energy(logits, sh.TEMPERATURE)[0]), abs=1e-5
    )
    assert body["routed_to_review"] is False and body["reasons"] == []
    assert body["model_version"] == "7" and body["model"]["precision"] == "fp32"
    assert body["thresholds"] == {
        "temperature": sh.TEMPERATURE,
        "review": sh.REVIEW,
        "ood_energy": sh.OOD,
    }
    assert body["image_sha256"] == hashlib.sha256(data).hexdigest()
    assert body["request_id"] == r.headers["X-Request-ID"] and len(body["request_id"]) == 32
    assert body["latency_ms"] >= 0


def test_unsure_image_is_routed_with_both_reasons(client):
    body = post_image(client, sh.image_bytes(MID)).json()
    assert body["routed_to_review"] is True
    assert body["reasons"] == ["low_confidence", "ood"]
    assert body["calibrated_confidence"] < sh.REVIEW and body["energy"] > sh.OOD


@pytest.mark.parametrize("fmt", ["JPEG", "BMP"])
def test_other_allowed_formats(client, fmt):
    assert post_image(client, sh.image_bytes(DARK, fmt=fmt)).status_code == 200


def test_request_ids_are_unique(client):
    ids = {post_image(client, sh.image_bytes(DARK)).json()["request_id"] for _ in range(3)}
    assert len(ids) == 3


# ---------------------------------------------------------------- input validation


@pytest.mark.parametrize(
    ("data", "status", "code"),
    [
        (b"", 422, "empty_file"),
        (b"this is not an image", 422, "undecodable"),
        (sh.image_bytes(DARK)[:200], 422, "undecodable"),  # valid header, truncated pixels
        (sh.image_bytes(DARK, fmt="GIF"), 415, "unsupported_format"),
        (sh.image_bytes(DARK, fmt="TIFF"), 415, "unsupported_format"),
        (sh.image_bytes(DARK, size=16), 422, "too_small"),
        (sh.image_bytes(128.0, std=0.0), 422, "near_blank"),
        (sh.image_bytes(128.0, std=0.4), 422, "near_blank"),
    ],
)
def test_bad_inputs_get_a_clear_4xx(client, data, status, code):
    assert_error(post_image(client, data), status, code)


def test_too_large_by_declared_length(client):
    limit = PARAMS["serving"]["max_upload_bytes"]
    data = b"\0" * (limit + api.MULTIPART_OVERHEAD + 1)
    assert_error(post_image(client, data), 413, "too_large")


def test_too_large_by_actual_bytes(bundle_dir):
    data = sh.image_bytes(DARK)
    app = api.create_app(
        params_with(max_upload_bytes=len(data) - 1), api_key=KEY, bundle_dir=bundle_dir
    )
    with TestClient(app) as c:
        assert_error(post_image(c, data), 413, "too_large")


def test_too_many_pixels_is_checked_before_decoding(bundle_dir):
    app = api.create_app(params_with(max_pixels=100 * 100), api_key=KEY, bundle_dir=bundle_dir)
    with TestClient(app) as c:
        assert_error(post_image(c, sh.image_bytes(DARK)), 422, "too_many_pixels")


def test_missing_file_field(client):
    r = client.post("/predict", data={"x": "1"}, headers={"X-API-Key": KEY})
    assert_error(r, 422, "invalid_request")


def test_length_required(client):
    r = client.post(
        "/predict", content=iter([b"chunk"]), headers={"X-API-Key": KEY, "Content-Type": "x"}
    )
    assert_error(r, 411, "length_required")


def test_unexpected_errors_are_json_500(client, monkeypatch):
    def boom(_):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(client.app.state.bundle.model, "logits", boom)
    r = post_image(client, sh.image_bytes(DARK))
    assert_error(r, 500, "internal_error")
    assert "exploded" not in r.text


# ---------------------------------------------------------------- feedback stub


def test_feedback_stub_validates_and_stores_nothing(client):
    h = {"X-API-Key": KEY}
    r = client.post("/feedback", json={"request_id": "a" * 32, "label": "scratches"}, headers=h)
    assert r.status_code == 202 and r.json()["stored"] is False
    bad = client.post("/feedback", json={"request_id": "a" * 32, "label": "cat"}, headers=h)
    assert_error(bad, 422, "unknown_label")
    bad = client.post("/feedback", json={"request_id": "nope", "label": "scratches"}, headers=h)
    assert_error(bad, 422, "invalid_request")


# ---------------------------------------------------------------- metrics and logs


def test_metrics_count_requests_predictions_and_routing(client):
    post_image(client, sh.image_bytes(DARK))
    post_image(client, sh.image_bytes(MID))
    post_image(client, b"junk")
    post_image(client, sh.image_bytes(DARK), key=None)
    text = client.get("/metrics").text
    assert 'defect_api_requests_total{endpoint="/predict",status="200"} 2.0' in text
    assert 'defect_api_predictions_total{prediction="rolled-in_scale"} 2.0' in text
    assert "defect_api_routed_to_review_total 1.0" in text
    assert 'defect_api_routed_reason_total{reason="ood"} 1.0' in text
    assert 'defect_api_rejections_total{reason="undecodable"} 1.0' in text
    assert 'defect_api_rejections_total{reason="missing_api_key"} 1.0' in text
    assert "defect_api_calibrated_confidence_bucket" in text
    assert "defect_api_energy_bucket" in text
    assert (
        'defect_api_model_info{precision="fp32",registered_model="steel-defect-classifier",'
        'version="7"} 1.0' in text
    )


def test_logs_are_json_with_request_id_and_hash_only(client, caplog):
    caplog.set_level(logging.INFO, logger="src.serving")
    data = sh.image_bytes(DARK)
    secret_name = "line7-camera3-secret-name.png"
    r = post_image(client, data, name=secret_name)
    post_image(client, data, key="wrong-key-value")
    lines = [JsonFormatter().format(rec) for rec in caplog.records if rec.name.startswith("src.")]
    entries = [json.loads(line) for line in lines]
    ok = [e for e in entries if e.get("request_id") == r.json()["request_id"]]
    assert len(ok) == 1 and ok[0]["event"] == "request" and ok[0]["status"] == 200
    assert ok[0]["image_sha256"] == hashlib.sha256(data).hexdigest()
    assert ok[0]["image_bytes"] == len(data) and ok[0]["prediction"] == "rolled-in_scale"
    assert any(e.get("error") == "invalid_api_key" for e in entries)
    blob = "\n".join(lines)
    for forbidden in (KEY, "wrong-key-value", secret_name, data[:64].hex(), "image/png"):
        assert forbidden not in blob


# ---------------------------------------------------------------- code rules


def test_serving_code_imports_no_torch_or_mlflow():
    for path in (REPO_ROOT / "src" / "serving").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        roots = {n.split(".")[0] for n in names}
        assert not roots & {"torch", "timm", "mlflow", "torchvision"}, path.name
        assert not {n for n in names if n.startswith("src.training")}, path.name


# ---------------------------------------------------------------- prediction log (TASK-019)


def test_refuses_to_start_without_a_database_url(bundle_dir, monkeypatch):
    monkeypatch.delenv("DATABASE_URL")
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        api.create_app(PARAMS, api_key=KEY, bundle_dir=bundle_dir)


def test_unreachable_database_stops_startup(bundle_dir, tmp_path):
    app = api.create_app(
        PARAMS, api_key=KEY, bundle_dir=bundle_dir, database_url="sqlite:///" + tmp_path.as_posix()
    )  # a folder, not a file: SQLite cannot open it
    with pytest.raises(Exception, match="unable to open"), TestClient(app):
        pass


def test_predict_writes_one_row_without_image_bytes(client, bundle_dir, database_url):
    data = sh.image_bytes(DARK)
    body = post_image(client, data, name="secret-name.png").json()
    rows = PredictionLog(database_url).recent(5)
    assert len(rows) == 1
    row = rows[0]
    assert row["request_id"] == body["request_id"]
    assert row["image_sha256"] == hashlib.sha256(data).hexdigest() == body["image_sha256"]
    for key in ("prediction", "routed_to_review", "model_version"):
        assert row[key] == body[key], key
    assert row["calibrated_confidence"] == pytest.approx(body["calibrated_confidence"])
    assert row["energy"] == pytest.approx(body["energy"])
    assert (row["reason_low_confidence"], row["reason_ood"]) == (False, False)
    assert (row["model_name"], row["model_precision"]) == ("steel-defect-classifier", "fp32")
    assert (row["image_format"], row["image_width"], row["image_height"]) == ("PNG", 200, 200)
    assert row["image_bytes"] == len(data)
    assert row["created_at"].tzinfo is not None
    assert row["inference_ms"] >= 0 and row["latency_ms"] >= row["inference_ms"]
    feats = image_features(load_image(data))
    for key, value in feats.items():
        assert row[key] == pytest.approx(value), key
    np.testing.assert_allclose(row["logits"], logits_for(bundle_dir, data)[0], rtol=1e-6)
    stored = repr(row)
    assert "secret-name" not in stored and KEY not in stored
    assert not any(isinstance(v, bytes | bytearray | memoryview) for v in row.values())


def test_routed_image_row_has_both_reasons(client, database_url):
    post_image(client, sh.image_bytes(MID))
    row = PredictionLog(database_url).recent(1)[0]
    assert row["routed_to_review"] and row["reason_low_confidence"] and row["reason_ood"]


def test_rejected_and_unauthorised_requests_write_no_row(client, database_url):
    post_image(client, b"junk")
    post_image(client, sh.image_bytes(128.0, std=0.0))
    post_image(client, sh.image_bytes(DARK), key=None)
    post_image(client, sh.image_bytes(DARK), key="wrong")
    assert PredictionLog(database_url).count() == 0


def test_log_write_failure_fails_closed(client, database_url, monkeypatch):
    def broken(row):
        raise RuntimeError("disk full")

    monkeypatch.setattr(client.app.state.db, "add", broken)
    r = post_image(client, sh.image_bytes(DARK))
    assert_error(r, 503, "log_unavailable")
    assert "prediction" not in r.json() and "disk full" not in r.text
    assert "defect_api_db_write_failures_total 1.0" in client.get("/metrics").text
    assert PredictionLog(database_url).count() == 0


def test_health_reports_the_database(client, monkeypatch):
    assert client.get("/health").json()["database"] == "ok"
    monkeypatch.setattr(client.app.state.db, "ping", lambda: False)
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "degraded" and r.json()["database"] == "unavailable"


def test_rows_accumulate_across_requests(client, database_url):
    for i in range(3):
        post_image(client, sh.image_bytes(DARK, seed=i))
    rows = PredictionLog(database_url).recent(10)
    assert len(rows) == 3 and len({r["request_id"] for r in rows}) == 3


def test_review_queue_gauge_counts_routed_rows(client, monkeypatch):
    post_image(client, sh.image_bytes(DARK))  # accepted
    post_image(client, sh.image_bytes(MID))  # routed
    post_image(client, sh.image_bytes(MID, seed=1))  # routed
    assert "defect_api_review_queue_size 2.0" in client.get("/metrics").text
    monkeypatch.setattr(client.app.state.db, "review_queue_size", lambda: 1 / 0)
    r = client.get("/metrics")  # a broken database never breaks the scrape
    assert r.status_code == 200 and "defect_api_review_queue_size NaN" in r.text
