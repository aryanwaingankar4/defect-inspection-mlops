# syntax=docker/dockerfile:1
# API image (TASK-021, ADR-012, ADR-028).
# - serve group only: ONNX Runtime + FastAPI, no torch, no mlflow;
# - the verified serving bundle (models/serving/, from `make serving-bundle`) is copied in;
# - runs as a non-root user; API_KEY and DATABASE_URL come from the environment at run time.
# Build with `make docker-build`. The build context is an allowlist (.dockerignore).

ARG PYTHON_IMAGE=python:3.11.18-slim-trixie

FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-default-groups --group serve --no-install-project

FROM ${PYTHON_IMAGE}
ARG MODEL_VERSION=unknown
ARG ONNX_SHA256=unknown
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.title="defect-inspection-api" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      io.defect-inspection.model-version="${MODEL_VERSION}" \
      io.defect-inspection.onnx-sha256="${ONNX_SHA256}"
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data \
    && chown app:app /data
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY src/__init__.py src/config.py src/
COPY src/db/ src/db/
COPY src/serving/ src/serving/
COPY configs/params.yaml configs/params.yaml
COPY models/serving/ models/serving/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]
CMD ["python", "-m", "src.serving.app", "--config", "configs/params.yaml", "--host", "0.0.0.0", "--port", "8000"]
