"""TASK-020: API contract. The live OpenAPI schema equals the committed snapshot, and every answer
(success and error) has exactly the documented fields with the documented types."""

import json
from pathlib import Path

import pytest

pytest.importorskip("onnx")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.config import load_params  # noqa: E402
from src.serving import app as api  # noqa: E402
from src.serving.contract import openapi_schema, render  # noqa: E402
from tests import serving_helpers as sh  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
SNAPSHOT = REPO_ROOT / "tests" / "contract" / "openapi.json"
SCHEMA = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
KEY = "contract-key"
TYPES = {
    "string": str,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


@pytest.fixture(autouse=True)
def database_url(tmp_path, monkeypatch):
    url = "sqlite:///" + (tmp_path / "db" / "p.db").as_posix()
    monkeypatch.setenv("DATABASE_URL", url)
    return url


@pytest.fixture
def client(tmp_path):
    bundle = sh.make_bundle(tmp_path / "bundle", PARAMS["preprocess"])
    with TestClient(api.create_app(PARAMS, api_key=KEY, bundle_dir=bundle)) as c:
        yield c


# ---------------------------------------------------------------- a small schema checker


def resolve(schema: dict) -> dict:
    ref = schema.get("$ref")
    return SCHEMA["components"]["schemas"][ref.split("/")[-1]] if ref else schema


def check(value, schema: dict, where: str = "$") -> None:
    """Raise AssertionError unless ``value`` fits ``schema`` (the subset FastAPI emits here).
    Objects must have every required field and no field the schema does not document."""
    schema = resolve(schema)
    if "anyOf" in schema:
        errors = []
        for option in schema["anyOf"]:
            try:
                check(value, option, where)
                return
            except AssertionError as exc:
                errors.append(str(exc))
        raise AssertionError(f"{where}: fits no anyOf option: {errors}")
    if "enum" in schema:
        assert value in schema["enum"], f"{where}: {value!r} not in {schema['enum']}"
    if "const" in schema:
        assert value == schema["const"], f"{where}: {value!r} != {schema['const']!r}"
    kind = schema.get("type")
    if kind in ("number", "integer"):
        ok = isinstance(value, int | float) and not isinstance(value, bool)
        assert ok and (kind == "number" or isinstance(value, int)), f"{where}: not {kind}"
    elif kind is not None:
        assert isinstance(value, TYPES[kind]), f"{where}: expected {kind}, got {type(value)}"
    if kind == "object":
        props = schema.get("properties", {})
        missing = set(schema.get("required", [])) - set(value)
        extra = set(value) - set(props)
        assert not missing, f"{where}: missing {missing}"
        assert not extra, f"{where}: undocumented {extra}"
        for key, item in value.items():
            check(item, props[key], f"{where}.{key}")
    if kind == "array":
        for i, item in enumerate(value):
            check(item, schema.get("items", {}), f"{where}[{i}]")


def documented(path: str, method: str, status: int) -> dict:
    responses = SCHEMA["paths"][path][method]["responses"]
    assert str(status) in responses, f"{method.upper()} {path} does not document {status}"
    return responses[str(status)]["content"]["application/json"]["schema"]


def assert_contract(r, path: str, method: str = "post") -> dict:
    body = r.json()
    check(body, documented(path, method, r.status_code))
    assert r.headers["X-Request-ID"] and len(r.headers["X-Request-ID"]) == 32
    if "request_id" in body and path == "/predict":
        assert body["request_id"] == r.headers["X-Request-ID"]
    if "error" in body:
        assert body["request_id"] == r.headers["X-Request-ID"]
    return body


def post_image(c, data, key=KEY):
    headers = {} if key is None else {"X-API-Key": key}
    return c.post("/predict", files={"file": ("x.png", data, "image/png")}, headers=headers)


# ---------------------------------------------------------------- the schema itself


def test_openapi_matches_the_committed_snapshot():
    assert render(openapi_schema(PARAMS)) == SNAPSHOT.read_bytes(), (
        "the API contract changed: if this is intended, run `make openapi-snapshot` and commit it"
    )


def test_snapshot_documents_every_endpoint_and_error():
    paths = SCHEMA["paths"]
    assert set(paths) == {"/predict", "/health", "/metrics", "/feedback"}
    predict = paths["/predict"]["post"]["responses"]
    assert set(predict) == {"200", "401", "403", "411", "413", "415", "422", "500", "503"}
    feedback = paths["/feedback"]["post"]["responses"]
    assert set(feedback) == {"202", "401", "403", "422", "500", "503"}
    assert set(paths["/health"]["get"]["responses"]) == {"200", "503"}
    for responses in (predict, feedback):
        for code, entry in responses.items():
            if code.startswith(("4", "5")):
                ref = entry["content"]["application/json"]["schema"]["$ref"]
                assert ref.endswith("/ErrorResponse"), code
    assert "HTTPValidationError" not in SCHEMA["components"]["schemas"]  # our 422 body is used
    required = set(SCHEMA["components"]["schemas"]["PredictResponse"]["required"])
    assert required >= {
        "request_id",
        "prediction",
        "calibrated_confidence",
        "energy",
        "routed_to_review",
        "reasons",
        "model_version",
    }


# ---------------------------------------------------------------- live answers


def test_predict_success_fits_the_contract(client):
    for mean in (25.0, 124.0):  # confident and routed
        body = assert_contract(post_image(client, sh.image_bytes(mean)), "/predict")
        assert body["prediction"] in sh.CLASSES


def test_health_fits_the_contract(client, tmp_path):
    assert_contract(client.get("/health"), "/health", "get")
    app = api.create_app(PARAMS, api_key=KEY, bundle_dir=tmp_path / "bundle")
    r = TestClient(app).get("/health")  # startup not run: not ready
    assert r.status_code == 503
    assert assert_contract(r, "/health", "get") == {"status": "not_ready"}


def test_feedback_fits_the_contract(client):
    r = client.post(
        "/feedback", json={"request_id": "a" * 32, "label": "crazing"}, headers={"X-API-Key": KEY}
    )
    assert r.status_code == 202
    assert_contract(r, "/feedback")


@pytest.mark.parametrize(
    ("make", "status"),
    [
        (lambda c: post_image(c, sh.image_bytes(25.0), key=None), 401),
        (lambda c: post_image(c, sh.image_bytes(25.0), key="wrong"), 403),
        (
            lambda c: c.post(
                "/predict", content=iter([b"x"]), headers={"X-API-Key": KEY, "Content-Type": "x"}
            ),
            411,
        ),
        (lambda c: post_image(c, b"\0" * (PARAMS["serving"]["max_upload_bytes"] + 20000)), 413),
        (lambda c: post_image(c, sh.image_bytes(25.0, fmt="GIF")), 415),
        (lambda c: post_image(c, b"not an image"), 422),
        (lambda c: post_image(c, sh.image_bytes(128.0, std=0.0)), 422),
        (lambda c: c.post("/predict", data={"x": "1"}, headers={"X-API-Key": KEY}), 422),
    ],
)
def test_predict_errors_fit_the_contract(client, make, status):
    r = make(client)
    assert r.status_code == status
    assert_contract(r, "/predict")


@pytest.mark.parametrize(
    ("body", "key", "status"),
    [
        ({"request_id": "a" * 32, "label": "crazing"}, None, 401),
        ({"request_id": "a" * 32, "label": "crazing"}, "wrong", 403),
        ({"request_id": "a" * 32, "label": "cat"}, KEY, 422),
        ({"request_id": "short", "label": "crazing"}, KEY, 422),
    ],
)
def test_feedback_errors_fit_the_contract(client, body, key, status):
    headers = {} if key is None else {"X-API-Key": key}
    r = client.post("/feedback", json=body, headers=headers)
    assert r.status_code == status
    assert_contract(r, "/feedback")


def test_503_and_500_fit_the_contract(client, monkeypatch):
    monkeypatch.setattr(client.app.state.db, "add", lambda row: 1 / 0)
    r = post_image(client, sh.image_bytes(25.0))
    assert r.status_code == 503
    assert_contract(r, "/predict")
    monkeypatch.setattr(client.app.state.bundle.model, "logits", lambda x: 1 / 0)
    r = post_image(client, sh.image_bytes(25.0))
    assert r.status_code == 500
    assert_contract(r, "/predict")


def test_metrics_is_prometheus_text(client):
    r = client.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert "# TYPE defect_api_requests_total counter" in r.text


def test_checker_catches_undocumented_and_missing_fields():
    schema = {"$ref": "#/components/schemas/ErrorResponse"}
    check({"error": "x", "detail": "y", "request_id": "z"}, schema)
    with pytest.raises(AssertionError, match="undocumented"):
        check({"error": "x", "detail": "y", "request_id": "z", "trace": "..."}, schema)
    with pytest.raises(AssertionError, match="missing"):
        check({"error": "x", "detail": "y"}, schema)
    with pytest.raises(AssertionError, match="expected string"):
        check({"error": 1, "detail": "y", "request_id": "z"}, schema)
