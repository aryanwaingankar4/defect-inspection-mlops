"""MLflow tracking (TASK-009, ADR-018, ADR-020). Only ``src/tracking/`` talks to MLflow.

Server: the ``MLFLOW_TRACKING_URI`` env var (DagsHub; credentials are read by MLflow itself
from ``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD``). If it is unset, runs go to
a local SQLite store (``tracking.local_uri``), so tests and offline work never reach DagsHub.
This module never reads, prints or logs credentials.

The DagsHub server is PUBLIC: every file goes through ``artifact_policy.safe_log_artifact``,
model files are checked before upload, and models are logged without an input example
(an example would upload a real NEU image).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit, urlunsplit

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # quiet a log line on import

import mlflow  # noqa: E402
import mlflow.pytorch  # noqa: E402
import numpy as np  # noqa: E402
from mlflow.models import ModelSignature  # noqa: E402
from mlflow.types import Schema, TensorSpec  # noqa: E402

from src.tracking.artifact_policy import (  # noqa: E402
    ArtifactPolicyError,
    find_violations,
    safe_log_artifact,
)

ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"
LOCAL_SCHEMES = frozenset({"", "file", "sqlite"})
MODEL_ARTIFACT_NAME = "model"


@dataclass(frozen=True)
class TrackingConfig:
    """Experiment and registry names for one run, from ``params.yaml`` -> ``tracking``."""

    experiment: str
    registered_model: str
    local_uri: str

    @classmethod
    def from_params(cls, params: Mapping[str, Any], *, smoke: bool) -> TrackingConfig:
        t = params["tracking"]
        return cls(
            experiment=t["smoke_experiment"] if smoke else t["experiment"],
            registered_model=t["smoke_registered_model"] if smoke else t["registered_model"],
            local_uri=t["local_uri"],
        )


def resolve_tracking_uri(local_uri: str) -> str:
    """``MLFLOW_TRACKING_URI`` if set and non-empty, else the local fallback."""
    return os.environ.get(ENV_TRACKING_URI, "").strip() or local_uri


def is_remote(uri: str) -> bool:
    return urlsplit(uri).scheme.lower() not in LOCAL_SCHEMES


def redact_uri(uri: str) -> str:
    """Drop any ``user:password@`` part so a URI is safe to print or store."""
    parts = urlsplit(uri)
    if not parts.username and not parts.password:
        return uri
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def flatten_params(params: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """``{"train": {"lr": 3e-4}}`` -> ``{"train.lr": "0.0003"}``; lists become JSON."""
    flat: dict[str, str] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, f"{name}."))
        else:
            flat[name] = json.dumps(value) if isinstance(value, list | tuple) else str(value)
    return flat


def _local_artifact_root(uri: str) -> str | None:
    """For a local SQLite store, keep artifacts in ``artifacts/`` next to the DB file."""
    if not uri.startswith("sqlite:///"):
        return None
    db = Path(uri.removeprefix("sqlite:///"))
    db.parent.mkdir(parents=True, exist_ok=True)
    return (db.parent / "artifacts").resolve().as_uri()


def tensor_signature(input_shape: Sequence[int], n_outputs: int) -> ModelSignature:
    """Shapes and dtypes only (batch dim = -1); contains no data."""
    f32 = np.dtype(np.float32)
    return ModelSignature(
        inputs=Schema([TensorSpec(f32, (-1, *input_shape))]),
        outputs=Schema([TensorSpec(f32, (-1, n_outputs))]),
    )


def _experiment_id(cfg: TrackingConfig, tracking_uri: str, *, create: bool) -> str | None:
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(cfg.experiment)
    if exp is not None:
        return exp.experiment_id
    if not create:
        return None
    root = None if is_remote(tracking_uri) else _local_artifact_root(tracking_uri)
    return mlflow.create_experiment(cfg.experiment, root)


class RunTracker:
    """One MLflow run as a context manager. Marks the run FAILED if the block raises.

    ``start_time_ms`` / ``end_time_ms`` let a run published later (ADR-020) keep the real
    training times instead of the time it was uploaded.
    """

    def __init__(
        self,
        cfg: TrackingConfig,
        *,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.tracking_uri = resolve_tracking_uri(cfg.local_uri)
        self.start_time_ms = start_time_ms
        self.end_time_ms = end_time_ms
        self.experiment_id: str | None = None
        self.run_id: str | None = None

    def __enter__(self) -> RunTracker:
        self.experiment_id = _experiment_id(self.cfg, self.tracking_uri, create=True)
        run_id = None
        if self.start_time_ms is not None:
            client = mlflow.MlflowClient()
            run_id = client.create_run(
                self.experiment_id, start_time=self.start_time_ms
            ).info.run_id
        self.run_id = mlflow.start_run(run_id=run_id, experiment_id=self.experiment_id).info.run_id
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        status = "FAILED" if exc_type else "FINISHED"
        mlflow.end_run(status=status)
        if self.end_time_ms is not None and self.run_id is not None:
            mlflow.MlflowClient().set_terminated(self.run_id, status, end_time=self.end_time_ms)

    @property
    def is_remote(self) -> bool:
        return is_remote(self.tracking_uri)

    @property
    def run_url(self) -> str:
        """Browser link for remote servers (no credentials); a locator for local stores."""
        base = redact_uri(self.tracking_uri).rstrip("/")
        if self.is_remote:
            return f"{base}/#/experiments/{self.experiment_id}/runs/{self.run_id}"
        return f"{base} (run {self.run_id})"

    def log_params(self, params: Mapping[str, Any]) -> None:
        mlflow.log_params(flatten_params(params))

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_file(self, local_path: Path | str, artifact_dir: str | None = None) -> None:
        """Log one file through the artifact policy (raises ``ArtifactPolicyError``)."""
        safe_log_artifact(local_path, artifact_dir, mlflow.log_artifact)

    def log_model(
        self,
        model: Any,
        *,
        input_shape: Sequence[int],
        n_outputs: int,
        metadata: Mapping[str, Any],
        pip_requirements: Sequence[str],
        register: bool,
    ) -> str | None:
        """Policy-check and log a PyTorch model; register it only if ``register``.

        Returns the registered version, or None if it was not registered.
        """
        kwargs = {
            # MLflow 3.18 defaults to "pt2", which needs an input example; we never log one.
            "serialization_format": "pickle",
            "signature": tensor_signature(input_shape, n_outputs),
            "metadata": dict(metadata),
            "pip_requirements": list(pip_requirements),
        }
        check_model_files(model, kwargs)
        info = mlflow.pytorch.log_model(
            model,
            name=MODEL_ARTIFACT_NAME,
            registered_model_name=self.cfg.registered_model if register else None,
            **kwargs,
        )
        version = info.registered_model_version
        return None if version is None else str(version)


def find_finished_run(cfg: TrackingConfig, tag: str, value: str) -> str | None:
    """Run id of a FINISHED run in the experiment whose ``tag`` equals ``value``, if any."""
    tracking_uri = resolve_tracking_uri(cfg.local_uri)
    exp_id = _experiment_id(cfg, tracking_uri, create=False)
    if exp_id is None:
        return None
    runs = mlflow.MlflowClient().search_runs(
        [exp_id], filter_string=f"tags.{tag} = '{value}' and attributes.status = 'FINISHED'"
    )
    return runs[0].info.run_id if runs else None


def register_run_model(cfg: TrackingConfig, run_id: str) -> str:
    """Register the model logged by ``run_id`` (once). Returns the version number."""
    mlflow.set_tracking_uri(resolve_tracking_uri(cfg.local_uri))
    client = mlflow.MlflowClient()
    for v in client.search_model_versions(f"name = '{cfg.registered_model}'"):
        if v.run_id == run_id:
            return str(v.version)  # already registered: do not create a duplicate version
    run = client.get_run(run_id)
    models = [
        m
        for m in client.search_logged_models(experiment_ids=[run.info.experiment_id])
        if m.source_run_id == run_id and m.name == MODEL_ARTIFACT_NAME
    ]
    if len(models) != 1:
        raise RuntimeError(f"expected one logged model in run {run_id}, found {len(models)}")
    version = mlflow.register_model(f"models:/{models[0].model_id}", cfg.registered_model)
    return str(version.version)


def check_model_files(model: Any, save_kwargs: Mapping[str, Any]) -> None:
    """Save the model to a temp folder first and refuse it if any file breaks the policy."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / MODEL_ARTIFACT_NAME
        mlflow.pytorch.save_model(model, target, **save_kwargs)
        files = [
            f"{MODEL_ARTIFACT_NAME}/{p.relative_to(target).as_posix()}"
            for p in target.rglob("*")
            if p.is_file()
        ]
    bad = find_violations(files)
    if bad:
        raise ArtifactPolicyError(f"Refusing to log model: files break the policy: {bad}")


def add_to_finished_run(
    cfg: TrackingConfig,
    run_id: str,
    *,
    metrics: Mapping[str, float],
    tags: Mapping[str, Any],
    files: Sequence[tuple[Path, str | None]],
) -> str:
    """Add metrics, tags and files to an existing run without reopening it (its status and
    times stay as they are). Files go through the artifact policy. Returns the run's link."""
    tracking_uri = resolve_tracking_uri(cfg.local_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    for path, artifact_dir in files:  # policy-check every file before anything is sent
        safe_log_artifact(path, artifact_dir, lambda *_: None)
    client.log_batch(
        run_id,
        metrics=[mlflow.entities.Metric(k, float(v), _now_ms(), 0) for k, v in metrics.items()],
        tags=[mlflow.entities.RunTag(k, str(v)) for k, v in tags.items()],
    )
    for path, artifact_dir in files:
        safe_log_artifact(path, artifact_dir, lambda p, d: client.log_artifact(run_id, p, d))
    base = redact_uri(tracking_uri).rstrip("/")
    if is_remote(tracking_uri):
        return f"{base}/#/experiments/{run.info.experiment_id}/runs/{run_id}"
    return f"{base} (run {run_id})"


def _now_ms() -> int:
    return int(time.time() * 1000)


def version_by_alias(cfg: TrackingConfig, alias: str) -> dict[str, Any] | None:
    """Version number, run id and tags of ``registered_model@alias`` (None if not set)."""
    client = mlflow.MlflowClient(tracking_uri=resolve_tracking_uri(cfg.local_uri))
    try:
        mv = client.get_model_version_by_alias(cfg.registered_model, alias)
    except mlflow.exceptions.MlflowException:
        return None
    return {"version": str(mv.version), "run_id": mv.run_id, "tags": dict(mv.tags)}


def set_version_tags(cfg: TrackingConfig, version: str, tags: Mapping[str, Any]) -> None:
    """Add tags to one registered version (text only)."""
    client = mlflow.MlflowClient(tracking_uri=resolve_tracking_uri(cfg.local_uri))
    for key, value in tags.items():
        client.set_model_version_tag(cfg.registered_model, version, key, str(value))
