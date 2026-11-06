"""MLflow tracking (TASK-009, ADR-018). The only module in ``src/`` that talks to MLflow.

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


class RunTracker:
    """One MLflow run as a context manager. Marks the run FAILED if the block raises."""

    def __init__(self, cfg: TrackingConfig) -> None:
        self.cfg = cfg
        self.tracking_uri = resolve_tracking_uri(cfg.local_uri)
        self.experiment_id: str | None = None
        self.run_id: str | None = None

    def __enter__(self) -> RunTracker:
        mlflow.set_tracking_uri(self.tracking_uri)
        exp = mlflow.get_experiment_by_name(self.cfg.experiment)
        if exp is None:
            root = None if is_remote(self.tracking_uri) else _local_artifact_root(self.tracking_uri)
            self.experiment_id = mlflow.create_experiment(self.cfg.experiment, root)
        else:
            self.experiment_id = exp.experiment_id
        self.run_id = mlflow.start_run(experiment_id=self.experiment_id).info.run_id
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        mlflow.end_run(status="FAILED" if exc_type else "FINISHED")

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
    ) -> str | None:
        """Policy-check, log and register a PyTorch model. Returns the registered version."""
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
            registered_model_name=self.cfg.registered_model,
            **kwargs,
        )
        version = info.registered_model_version
        return None if version is None else str(version)


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
