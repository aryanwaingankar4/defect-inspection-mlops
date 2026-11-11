"""FastAPI service (TASK-018, ADR-028): ``/predict``, ``/health``, ``/metrics``, ``/feedback`` stub.

Serves the verified bundle built by ``make serving-bundle``: the ONNX file named by the
``@champion`` version's ``serving_precision`` tag, with ``T``, the review threshold and the OOD
threshold from its tags. Uses the shared ``preprocess.py`` and ``scoring.py`` only.

    make serve        # http://127.0.0.1:8000, API_KEY from .env

Security: ``/predict`` and ``/feedback`` need the ``X-API-Key`` header (env ``API_KEY``,
compared in constant time). Logs are JSON lines with ``request_id`` and the image SHA-256;
never image bytes, file names or the key. Each served prediction is written to the prediction
log (``DATABASE_URL``, ADR-029) before it is returned; if the write fails, ``/predict`` returns 503.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, File, Request, Security, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.security import APIKeyHeader
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from src.config import load_params
from src.db.store import DatabaseConfigError, PredictionLog, database_url_from_env
from src.serving import json_logging
from src.serving.bundle import Bundle, load_bundle
from src.serving.features import image_features
from src.serving.preprocess import to_model_input
from src.serving.routing import with_audit
from src.serving.scoring import score
from src.serving.validation import Decoded, InputRejected, ValidationConfig, decode_checked

API_KEY_ENV = "API_KEY"
API_KEY_HEADER = "X-API-Key"
AUDIT_SALT_ENV = "AUDIT_SALT"
# Documents the key in OpenAPI; the check itself runs in the middleware, before the body.
API_KEY_SCHEME = APIKeyHeader(
    name=API_KEY_HEADER,
    scheme_name="ApiKey",
    description="API key from the API_KEY env var: missing -> 401, wrong -> 403",
    auto_error=False,
)
REQUEST_ID_HEADER = "X-Request-ID"
PROTECTED = frozenset({"/predict", "/feedback"})
KNOWN_PATHS = frozenset({"/predict", "/feedback", "/health", "/metrics"})
MULTIPART_OVERHEAD = 16 * 1024  # form boundaries and part headers around the file
log = logging.getLogger(f"{json_logging.LOGGER_NAME}.app")


class ApiError(Exception):
    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status, self.code, self.detail = status, code, detail


# ---------------------------------------------------------------- response schemas


class ModelInfo(BaseModel):
    name: str
    version: str
    alias: str
    precision: str


class ThresholdInfo(BaseModel):
    temperature: float
    review: float
    ood_energy: float
    audit_rate: float


class PredictResponse(BaseModel):
    request_id: str
    prediction: str
    calibrated_confidence: float = Field(description="max softmax(logits / T), ADR-006/025")
    energy: float = Field(description="-T * logsumexp(logits / T); higher = less familiar")
    routed_to_review: bool
    reasons: list[Literal["low_confidence", "ood", "audit"]] = Field(
        description="why the image goes to review: low_confidence and/or ood (ADR-023, "
        "ADR-027), or audit for a random sample of otherwise accepted images (ADR-030); "
        "empty if it does not"
    )
    model_version: str
    model: ModelInfo
    thresholds: ThresholdInfo
    image_sha256: str
    latency_ms: float


class ErrorResponse(BaseModel):
    """Body of every 4xx/5xx answer."""

    error: str = Field(description="short machine-readable code, e.g. near_blank")
    detail: str
    request_id: str


class HealthResponse(BaseModel):
    """``model``, ``bundle_built_at`` and ``database`` are present once the model is loaded."""

    status: Literal["ok", "degraded", "not_ready"]
    model: ModelInfo | None = None
    bundle_built_at: str | None = None
    database: Literal["ok", "unavailable"] | None = None


def error_docs(*codes: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI entries for the error answers an endpoint can give (all use ``ErrorResponse``)."""
    text = {
        401: "X-API-Key header missing",
        403: "X-API-Key wrong",
        411: "Content-Length missing",
        413: "file too large",
        415: "image format not allowed",
        422: "invalid request or image (undecodable, too small, too many pixels, near blank)",
        500: "unexpected error",
        503: "model not loaded or prediction could not be logged",
    }
    return {c: {"model": ErrorResponse, "description": text[c]} for c in codes}


class FeedbackRequest(BaseModel):
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    label: str = Field(max_length=64)


class FeedbackResponse(BaseModel):
    request_id: str
    stored: bool
    detail: str


# ---------------------------------------------------------------- metrics


class Metrics:
    """Prometheus metrics in a registry owned by one app (tests build many apps)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            "defect_api_requests_total", "HTTP requests", ["endpoint", "status"], registry=r
        )
        self.latency = Histogram(
            "defect_api_request_latency_seconds", "Request latency", ["endpoint"], registry=r
        )
        self.inference = Histogram(
            "defect_api_inference_latency_seconds", "ONNX Runtime time per image", registry=r
        )
        self.rejections = Counter(
            "defect_api_rejections_total", "Requests rejected with a 4xx", ["reason"], registry=r
        )
        self.predictions = Counter(
            "defect_api_predictions_total", "Predictions by class", ["prediction"], registry=r
        )
        self.routed = Counter(
            "defect_api_routed_to_review_total", "Images routed to review", registry=r
        )
        self.routed_reason = Counter(
            "defect_api_routed_reason_total", "Routing reasons", ["reason"], registry=r
        )
        self.confidence = Histogram(
            "defect_api_calibrated_confidence",
            "Calibrated confidence of the predicted class",
            buckets=(0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0),
            registry=r,
        )
        self.energy = Histogram(
            "defect_api_energy",
            "Energy score",
            buckets=(-30, -25, -20, -15, -12, -10, -8, -6, -4, -2, 0),
            registry=r,
        )
        self.queue_size = Gauge(
            "defect_api_review_queue_size",
            "Predictions waiting for review (read from the prediction log at scrape time)",
            registry=r,
        )
        self.db_failures = Counter(
            "defect_api_db_write_failures_total",
            "Predictions not served because the log write failed (ADR-029)",
            registry=r,
        )
        self.model_info = Gauge(
            "defect_api_model_info",
            "Model being served (value 1)",
            ["registered_model", "version", "precision"],
            registry=r,
        )


# ---------------------------------------------------------------- helpers


def error_response(status: int, code: str, detail: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        {"error": code, "detail": detail, "request_id": request_id},
        status_code=status,
        headers={REQUEST_ID_HEADER: request_id},
    )


def key_ok(given: str | None, expected: str) -> bool:
    return given is not None and hmac.compare_digest(given.encode(), expected.encode())


def queue_size(db: PredictionLog) -> float:
    """Review queue size for the gauge; NaN if the database cannot be read (visible as a gap)."""
    try:
        return float(db.review_queue_size())
    except Exception:  # noqa: BLE001 - a scrape must never fail
        return float("nan")


def endpoint_label(path: str) -> str:
    return path if path in KNOWN_PATHS else "other"


def predict_one(bundle: Bundle, image: Any) -> tuple[dict[str, Any], float, list[float]]:
    """Shared preprocessing -> ONNX raw logits -> shared scoring.

    Returns the result, the inference time in seconds and the raw logits."""
    x = to_model_input(image, bundle.preprocess)[np.newaxis]
    t0 = time.perf_counter()
    logits = bundle.model.logits(x)
    seconds = time.perf_counter() - t0
    s = score(logits, bundle.thresholds)
    return (
        {
            "prediction": bundle.classes[int(s.pred[0])],
            "calibrated_confidence": float(s.confidence[0]),
            "energy": float(s.energy[0]),
            "routed_to_review": bool(s.routed[0]),
            "reasons": s.reasons(0),
        },
        seconds,
        [float(v) for v in logits[0]],
    )


def log_row(
    request_id: str,
    sha: str,
    n_bytes: int,
    decoded: Decoded,
    result: Mapping[str, Any],
    logits: list[float],
    info: Mapping[str, str],
    latency_ms: float,
    inference_ms: float,
) -> dict[str, Any]:
    """The prediction-log row (ADR-029): hashes, results and summary features, never bytes."""
    return {
        "request_id": request_id,
        "created_at": datetime.now(UTC),
        "image_sha256": sha,
        "prediction": result["prediction"],
        "calibrated_confidence": result["calibrated_confidence"],
        "energy": result["energy"],
        "routed_to_review": result["routed_to_review"],
        "reason_low_confidence": "low_confidence" in result["reasons"],
        "reason_ood": "ood" in result["reasons"],
        "reason_audit": "audit" in result["reasons"],
        "model_name": info["name"],
        "model_version": info["version"],
        "model_precision": info["precision"],
        "latency_ms": latency_ms,
        "inference_ms": inference_ms,
        "image_bytes": n_bytes,
        "image_width": decoded.width,
        "image_height": decoded.height,
        "image_format": decoded.format,
        **image_features(decoded.image),
        "logits": logits,
    }


# ---------------------------------------------------------------- app


def create_app(
    params: Mapping[str, Any],
    *,
    api_key: str | None = None,
    bundle_dir: Path | str | None = None,
    database_url: str | None = None,
    audit_salt: str | None = None,
) -> FastAPI:
    """Build the app. Refuses to start without an API key, a database URL, or (when the audit
    rate is above 0) an audit salt; the bundle and the prediction log open at startup."""
    api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
    if not api_key.strip():
        raise RuntimeError(f"{API_KEY_ENV} is not set: the API refuses to start without it")
    try:
        db_url = database_url or database_url_from_env()
    except DatabaseConfigError as exc:
        raise RuntimeError(f"{exc}: the API refuses to start without it (ADR-029)") from exc
    audit_rate = float(params["serving"]["audit_rate"])
    if not 0.0 <= audit_rate <= 1.0:
        raise RuntimeError(f"serving.audit_rate must be in [0, 1], got {audit_rate}")
    salt = audit_salt if audit_salt is not None else os.environ.get(AUDIT_SALT_ENV, "")
    if audit_rate > 0 and not salt.strip():
        raise RuntimeError(
            f"{AUDIT_SALT_ENV} is not set and audit_rate > 0: the API refuses to start (ADR-030)"
        )
    vcfg = ValidationConfig.from_params(params)
    bundle_path = Path(bundle_dir or params["serving"]["bundle_dir"])
    metrics = Metrics()
    json_logging.configure()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        bundle = load_bundle(bundle_path, params["preprocess"])  # raises: the API does not start
        db = PredictionLog(db_url)
        db.create_tables()  # raises if the database is unreachable: the API does not start
        app.state.bundle, app.state.db = bundle, db
        metrics.queue_size.set_function(lambda: queue_size(db))
        info = bundle.model_info
        metrics.model_info.labels(info["name"], info["version"], info["precision"]).set(1)
        log.info("startup", extra={"fields": {"model": info, "bundle_dir": str(bundle_path)}})
        yield
        app.state.bundle = app.state.db = None
        db.dispose()

    app = FastAPI(
        title="Steel defect inspection API",
        version="0.1.0",
        description="NEU steel-defect classifier (ONNX) with calibrated confidence and review "
        "routing. Reviewer and drift are simulated in this project.",
        lifespan=lifespan,
    )
    app.state.bundle = app.state.db = None
    app.state.metrics = metrics

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Response:
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.log = {}
        start = time.perf_counter()
        path = request.url.path
        try:
            response = early_rejection(request, path, request_id)
            if response is None:
                response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 - one JSON 500 for anything unexpected
            log.error("unhandled", exc_info=exc, extra={"fields": {"request_id": request_id}})
            response = error_response(500, "internal_error", "internal error", request_id)
        seconds = time.perf_counter() - start
        response.headers[REQUEST_ID_HEADER] = request_id
        endpoint = endpoint_label(path)
        metrics.requests.labels(endpoint, str(response.status_code)).inc()
        metrics.latency.labels(endpoint).observe(seconds)
        if path != "/metrics":
            fields = {
                "request_id": request_id,
                "method": request.method,
                "path": endpoint,
                "status": response.status_code,
                "latency_ms": round(seconds * 1000, 3),
                **request.state.log,
            }
            log.info("request", extra={"fields": fields})
        return response

    def early_rejection(request: Request, path: str, request_id: str) -> JSONResponse | None:
        """Key and declared size are checked before the body is read."""
        if path not in PROTECTED:
            return None
        given = request.headers.get(API_KEY_HEADER)
        if given is None:
            return reject(request, 401, "missing_api_key", f"{API_KEY_HEADER} header required")
        if not key_ok(given, api_key):
            return reject(request, 403, "invalid_api_key", "invalid API key")
        if path == "/predict":
            length = request.headers.get("content-length")
            if length is None:
                return reject(request, 411, "length_required", "Content-Length required")
            if not length.isdigit() or int(length) > vcfg.max_upload_bytes + MULTIPART_OVERHEAD:
                return reject(
                    request,
                    413,
                    "too_large",
                    f"the file is larger than {vcfg.max_upload_bytes} bytes",
                )
        return None

    def reject(request: Request, status: int, code: str, detail: str) -> JSONResponse:
        metrics.rejections.labels(code).inc()
        request.state.log["error"] = code
        return error_response(status, code, detail, request.state.request_id)

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        return reject(request, exc.status, exc.code, exc.detail)

    @app.exception_handler(InputRejected)
    async def input_rejected(request: Request, exc: InputRejected) -> JSONResponse:
        return reject(request, exc.status, exc.code, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Field locations and messages only: the submitted values are never echoed or logged.
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', '')}"
            for e in exc.errors()
        )
        return reject(request, 422, "invalid_request", detail)

    def ready_bundle(request: Request) -> Bundle:
        bundle = request.app.state.bundle
        if bundle is None:
            raise ApiError(503, "not_ready", "the model is not loaded")
        return bundle

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse, "description": "not ready or degraded"}},
    )
    def health(request: Request) -> JSONResponse:
        bundle, db = request.app.state.bundle, request.app.state.db
        if bundle is None or db is None:
            body = HealthResponse(status="not_ready")
            return JSONResponse(body.model_dump(exclude_none=True), status_code=503)
        db_ok = db.ping()
        body = HealthResponse(
            status="ok" if db_ok else "degraded",
            model=ModelInfo(**bundle.model_info),
            bundle_built_at=bundle.manifest["built_at"],
            database="ok" if db_ok else "unavailable",
        )
        return JSONResponse(body.model_dump(exclude_none=True), status_code=200 if db_ok else 503)

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses=error_docs(401, 403, 411, 413, 415, 422, 500, 503),
        dependencies=[Security(API_KEY_SCHEME)],
    )
    def predict(request: Request, file: UploadFile = File(...)) -> PredictResponse:  # noqa: B008
        start = time.perf_counter()
        bundle = ready_bundle(request)
        data = file.file.read(vcfg.max_upload_bytes + 1)
        request.state.log["image_bytes"] = len(data)
        if data:
            request.state.log["image_sha256"] = hashlib.sha256(data).hexdigest()
        decoded = decode_checked(data, vcfg)
        result, seconds, logits = predict_one(bundle, decoded.image)
        result["reasons"] = with_audit(
            result["reasons"], request.state.request_id, salt, audit_rate
        )
        result["routed_to_review"] = bool(result["reasons"])
        metrics.inference.observe(seconds)
        metrics.predictions.labels(result["prediction"]).inc()
        metrics.confidence.observe(result["calibrated_confidence"])
        metrics.energy.observe(result["energy"])
        if result["routed_to_review"]:
            metrics.routed.inc()
        for reason in result["reasons"]:
            metrics.routed_reason.labels(reason).inc()
        info = bundle.model_info
        request.state.log.update({**result, "model_version": info["version"]})
        sha = request.state.log["image_sha256"]
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        row = log_row(
            request.state.request_id,
            sha,
            len(data),
            decoded,
            result,
            logits,
            info,
            latency_ms,
            round(seconds * 1000, 3),
        )
        try:
            request.app.state.db.add(row)
        except Exception as exc:  # noqa: BLE001 - fail closed (ADR-029)
            metrics.db_failures.inc()
            log.error("log_write_failed", extra={"fields": {"exc_type": type(exc).__name__}})
            raise ApiError(
                503, "log_unavailable", "the prediction could not be logged; retry later"
            ) from exc
        request.state.log["logged"] = True
        th = bundle.thresholds
        return PredictResponse(
            request_id=request.state.request_id,
            **result,
            model_version=info["version"],
            model=ModelInfo(**info),
            thresholds=ThresholdInfo(
                temperature=th.temperature,
                review=th.review_threshold,
                ood_energy=th.ood_energy_threshold,
                audit_rate=audit_rate,
            ),
            image_sha256=sha,
            latency_ms=latency_ms,
        )

    @app.post(
        "/feedback",
        status_code=202,
        response_model=FeedbackResponse,
        responses=error_docs(401, 403, 422, 500, 503),
        dependencies=[Security(API_KEY_SCHEME)],
    )
    def feedback(request: Request, body: FeedbackRequest) -> FeedbackResponse:
        bundle = ready_bundle(request)
        if body.label not in bundle.classes:
            raise ApiError(422, "unknown_label", f"label must be one of {bundle.classes}")
        request.state.log.update({"feedback_for": body.request_id, "label": body.label})
        return FeedbackResponse(
            request_id=body.request_id,
            stored=False,
            detail="stub: feedback is validated but not stored until TASK-026",
        )

    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/params.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    app = create_app(load_params(args.config))
    # Our JSON request log replaces uvicorn's access log (which would log client addresses).
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
