"""MLflow tracking (TASK-009). Local SQLite store only (see conftest.offline_mlflow)."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mlflow")

import mlflow.pytorch  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from src.config import load_params  # noqa: E402
from src.tracking import mlflow_tracking as mt  # noqa: E402
from src.tracking.artifact_policy import ArtifactPolicyError, find_violations  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
CFG = mt.TrackingConfig(experiment="exp", registered_model="m", local_uri="sqlite:///unused.db")


def tiny_model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))


def log_tiny_model(tracker: mt.RunTracker) -> str | None:
    return tracker.log_model(
        tiny_model(),
        input_shape=(3, 4, 4),
        n_outputs=2,
        metadata={"classes": ["a", "b"]},
        pip_requirements=[f"torch=={torch.__version__.split('+')[0]}"],
    )


def all_artifacts(client: MlflowClient, run_id: str, path: str | None = None) -> list[str]:
    out = []
    for item in client.list_artifacts(run_id, path):
        out += all_artifacts(client, run_id, item.path) if item.is_dir else [item.path]
    return out


# ---------------------------------------------------------------- pure helpers


def test_offline_fixture_points_to_local_store(offline_mlflow: str) -> None:
    assert mt.resolve_tracking_uri("sqlite:///fallback.db") == offline_mlflow
    assert not mt.is_remote(offline_mlflow)


def test_fallback_to_local_uri_when_env_unset(monkeypatch) -> None:
    local = "sqlite:///mlruns/mlflow.db"
    with monkeypatch.context() as m:  # undone before the offline fixture's final check
        m.delenv("MLFLOW_TRACKING_URI")
        assert mt.resolve_tracking_uri(local) == local
        m.setenv("MLFLOW_TRACKING_URI", "  ")
        assert mt.resolve_tracking_uri(local) == local


@pytest.mark.parametrize(
    ("uri", "remote"),
    [
        ("https://dagshub.com/u/r.mlflow", True),
        ("http://localhost:5000", True),
        ("sqlite:///mlruns/mlflow.db", False),
        ("file:///tmp/mlruns", False),
    ],
)
def test_is_remote(uri: str, remote: bool) -> None:
    assert mt.is_remote(uri) is remote


def test_redact_uri_removes_credentials() -> None:
    assert mt.redact_uri("https://user:s3cret@dagshub.com/u/r.mlflow") == (
        "https://dagshub.com/u/r.mlflow"
    )
    assert mt.redact_uri("https://dagshub.com/u/r.mlflow") == "https://dagshub.com/u/r.mlflow"


def test_run_url_has_no_credentials() -> None:
    tracker = mt.RunTracker(CFG)
    tracker.tracking_uri = "https://user:s3cret@dagshub.com/u/r.mlflow"
    tracker.experiment_id, tracker.run_id = "3", "abc"
    assert tracker.run_url == "https://dagshub.com/u/r.mlflow/#/experiments/3/runs/abc"
    assert "s3cret" not in tracker.run_url


def test_flatten_params() -> None:
    flat = mt.flatten_params({"train": {"lr": 0.0003, "splits": ["a", "b"], "aug": {"p": 0.5}}})
    assert flat == {"train.lr": "0.0003", "train.splits": '["a", "b"]', "train.aug.p": "0.5"}


def test_smoke_and_real_names_come_from_params() -> None:
    t = PARAMS["tracking"]
    real = mt.TrackingConfig.from_params(PARAMS, smoke=False)
    smoke = mt.TrackingConfig.from_params(PARAMS, smoke=True)
    assert (real.experiment, real.registered_model) == (t["experiment"], t["registered_model"])
    assert (smoke.experiment, smoke.registered_model) == (
        t["smoke_experiment"],
        t["smoke_registered_model"],
    )
    assert real.registered_model != smoke.registered_model


def test_signature_has_shapes_only() -> None:
    sig = mt.tensor_signature((3, 224, 224), 6)
    assert sig.inputs.inputs[0].shape == (-1, 3, 224, 224)
    assert sig.outputs.inputs[0].shape == (-1, 6)


# ---------------------------------------------------------------- against a local store


def test_run_logs_everything_and_registers_model(tmp_path: Path) -> None:
    plot = tmp_path / "training_curves.png"
    plot.write_bytes(b"\x89PNG fake")
    table = tmp_path / "val_metrics.json"
    table.write_text("{}", encoding="utf-8")

    with mt.RunTracker(CFG) as tracker:
        tracker.log_params({"train": {"lr": 0.001}})
        tracker.set_tags({"git_commit": "abc123", "smoke": True})
        tracker.log_metrics({"val_macro_f1": 0.5}, step=1)
        tracker.log_metrics({"val_macro_f1": 0.7}, step=2)
        tracker.log_file(plot, "plots")
        tracker.log_file(table)
        version = log_tiny_model(tracker)
        run_id = tracker.run_id

    client = MlflowClient()
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["train.lr"] == "0.001"
    assert run.data.tags["git_commit"] == "abc123"
    history = client.get_metric_history(run_id, "val_macro_f1")
    assert [(m.step, m.value) for m in history] == [(1, 0.5), (2, 0.7)]
    artifacts = all_artifacts(client, run_id)
    assert {"plots/training_curves.png", "val_metrics.json"} <= set(artifacts)
    assert find_violations(artifacts) == []

    assert version == "1"
    loaded = mlflow.pytorch.load_model("models:/m/1")
    assert loaded(torch.zeros(1, 3, 4, 4)).shape == (1, 2)
    model_info = mlflow.models.get_model_info("models:/m/1")
    assert model_info.saved_input_example_info is None  # never upload NEU pixels as an example
    assert model_info.metadata["classes"] == ["a", "b"]


def test_failed_block_marks_run_failed() -> None:
    with pytest.raises(RuntimeError), mt.RunTracker(CFG) as tracker:
        run_id = tracker.run_id
        raise RuntimeError("boom")
    assert MlflowClient().get_run(run_id).info.status == "FAILED"


def test_log_file_refuses_non_allowlisted_image(tmp_path: Path) -> None:
    sample = tmp_path / "sample_prediction.png"
    sample.write_bytes(b"\x89PNG fake")
    with mt.RunTracker(CFG) as tracker:
        with pytest.raises(ArtifactPolicyError):
            tracker.log_file(sample, "plots")
        run_id = tracker.run_id
    assert all_artifacts(MlflowClient(), run_id) == []


def test_model_with_image_file_is_refused_before_upload(monkeypatch) -> None:
    real_save = mlflow.pytorch.save_model

    def save_with_image(model, path, **kwargs):
        real_save(model, path, **kwargs)
        (Path(path) / "example.jpg").write_bytes(b"not really a jpeg")

    uploads = []
    monkeypatch.setattr(mlflow.pytorch, "save_model", save_with_image)
    monkeypatch.setattr(mlflow.pytorch, "log_model", lambda *a, **k: uploads.append(a))
    with mt.RunTracker(CFG) as tracker, pytest.raises(ArtifactPolicyError, match="example.jpg"):
        log_tiny_model(tracker)
    assert uploads == []
