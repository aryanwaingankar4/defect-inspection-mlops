"""Publishing a finished training run from the laptop (ADR-020). Local SQLite store only."""

import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mlflow")

from mlflow import MlflowClient  # noqa: E402

from src.tracking import publish as pub  # noqa: E402
from src.tracking.artifact_policy import find_violations  # noqa: E402
from src.training import train as train_mod  # noqa: E402
from tests.training_helpers import PARAMS, make_synthetic_data, tiny_params  # noqa: E402

REAL = PARAMS["tracking"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict]:
    """One tiny CPU training run (2 epochs), shared by the tests in this module (read-only)."""
    root = tmp_path_factory.mktemp("trained")
    raw, csv_path = make_synthetic_data(root)
    params = tiny_params(raw, csv_path, root / "out", max_epochs=2, patience=5)
    metrics = train_mod.run(params, torch.device("cpu"))
    return root / "out" / "results", root / "out" / "best.pt", metrics


@pytest.fixture
def run_copy(trained, tmp_path: Path) -> tuple[Path, Path, dict]:
    """A private copy, so a test can write mlflow_run.json or damage the checkpoint."""
    run_dir, ckpt, metrics = trained
    shutil.copytree(run_dir, tmp_path / "results")
    shutil.copy2(ckpt, tmp_path / "best.pt")
    return tmp_path / "results", tmp_path / "best.pt", metrics


def artifacts(client: MlflowClient, run_id: str, path: str | None = None) -> list[str]:
    out = []
    for item in client.list_artifacts(run_id, path):
        out += artifacts(client, run_id, item.path) if item.is_dir else [item.path]
    return out


def test_publish_logs_the_whole_run_without_registering(run_copy) -> None:
    run_dir, ckpt, metrics = run_copy
    record = pub.publish(PARAMS, run_dir, ckpt)

    client = MlflowClient()
    run = client.get_run(record["run_id"])
    assert run.info.status == "FINISHED"
    assert client.get_experiment(run.info.experiment_id).name == REAL["experiment"]
    timing = metrics["timing"]
    assert (run.info.start_time, run.info.end_time) == (
        timing["started_at_ms"],
        timing["finished_at_ms"],
    )  # real training times, not upload time

    tags, prov = run.data.tags, metrics["provenance"]
    for key in ("git_commit", "dvc_data_md5", "data_fingerprint", "runtime"):
        assert tags[key] == str(prov[key])
    assert tags["git_dirty"] == str(prov["git_dirty"])
    assert tags[pub.RUN_UID_TAG] == metrics["run_uid"]
    assert tags["role"] == "candidate" and tags["model"] == metrics["model"]
    assert tags["device"] == "cpu" and tags["smoke"] == "False"
    assert run.data.params["train.lr"] == str(PARAMS["train"]["lr"])

    history = client.get_metric_history(run.info.run_id, "val_macro_f1")
    assert [m.step for m in history] == [1, 2]  # per-epoch, step = epoch
    assert run.data.metrics["best_val_macro_f1"] == metrics["val_macro_f1"]

    files = artifacts(client, run.info.run_id)
    expected = {"val_metrics.json", "config/params.yaml", "training_curves.csv"}
    assert expected | {"plots/training_curves.png"} <= set(files)
    assert find_violations(files) == []
    logged = client.search_logged_models(experiment_ids=[run.info.experiment_id])
    assert [m.source_run_id for m in logged] == [run.info.run_id]  # the model, logged once

    assert record["registered_version"] is None and record["registered_model"] is None
    assert client.search_model_versions(f"name = '{REAL['registered_model']}'") == []
    saved = json.loads((run_dir / pub.PUBLISH_RECORD).read_text(encoding="utf-8"))
    assert saved["run_id"] == record["run_id"]


def test_publish_twice_is_refused_then_register_later(run_copy) -> None:
    run_dir, ckpt, _ = run_copy
    first = pub.publish(PARAMS, run_dir, ckpt)
    with pytest.raises(pub.PublishError, match="already published"):
        pub.publish(PARAMS, run_dir, ckpt)

    later = pub.publish(PARAMS, run_dir, ckpt, register=True)
    assert later["run_id"] == first["run_id"]
    assert later["registered_version"] == "1"
    again = pub.publish(PARAMS, run_dir, ckpt, register=True)
    assert again["registered_version"] == "1"  # no second version for the same run


def test_register_on_first_publish_and_model_loads(run_copy) -> None:
    import mlflow.pytorch

    run_dir, ckpt, metrics = run_copy
    record = pub.publish(PARAMS, run_dir, ckpt, register=True)
    assert record["registered_version"] == "1"
    model = mlflow.pytorch.load_model(f"models:/{REAL['registered_model']}/1")
    size = metrics["params"]["preprocess"]["image_size"]
    assert model(torch.zeros(1, 3, size, size)).shape == (1, len(metrics["classes"]))


def test_comparison_runs_are_tagged_and_never_registered(run_copy) -> None:
    run_dir, ckpt, _ = run_copy
    with pytest.raises(pub.PublishError, match="never registered"):
        pub.publish(PARAMS, run_dir, ckpt, role="comparison", register=True)
    record = pub.publish(PARAMS, run_dir, ckpt, role="comparison")
    assert MlflowClient().get_run(record["run_id"]).data.tags["role"] == "comparison"
    with pytest.raises(pub.PublishError, match="never registered"):
        pub.publish(PARAMS, run_dir, ckpt, role="comparison", register=True)


def test_wrong_checkpoint_is_refused(run_copy, offline_mlflow: str) -> None:
    run_dir, ckpt, _ = run_copy
    with ckpt.open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(pub.PublishError, match="SHA-256"):
        pub.publish(PARAMS, run_dir, ckpt)
    # This test's own store (the global MLflow URI may still point at a previous test's).
    client = MlflowClient(tracking_uri=offline_mlflow)
    assert client.search_experiments(filter_string=f"name = '{REAL['experiment']}'") == []


def test_incomplete_or_old_runs_are_refused(run_copy, tmp_path: Path) -> None:
    run_dir, ckpt, _ = run_copy
    (run_dir / "training_curves.png").unlink()
    with pytest.raises(pub.PublishError, match="incomplete"):
        pub.publish(PARAMS, run_dir, ckpt)

    old = tmp_path / "old"
    old.mkdir()
    for name, _ in pub.ARTIFACTS:
        (old / name).write_text("{}" if name.endswith(".json") else "x", encoding="utf-8")
    with pytest.raises(pub.PublishError, match="before ADR-020"):
        pub.publish(PARAMS, old, ckpt)


def test_copy_results_never_copies_the_checkpoint(run_copy, tmp_path: Path) -> None:
    run_dir, ckpt, _ = run_copy
    pub.publish(PARAMS, run_dir, ckpt, role="comparison")
    copied = pub.copy_results(run_dir, tmp_path / "results" / "compare" / "m")
    names = sorted(p.name for p in copied)
    assert names == sorted([n for n, _ in pub.ARTIFACTS] + [pub.PUBLISH_RECORD])
    assert not any(p.suffix == ".pt" for p in (tmp_path / "results").rglob("*"))


def test_default_paths_follow_params() -> None:
    k = PARAMS["kaggle"]
    run_dir, ckpt = pub.default_paths(PARAMS, smoke=False)
    assert run_dir == Path(k["output_dir"]) / k["output_results"]
    assert ckpt == Path(k["output_dir"]) / k["output_checkpoint"]
    smoke_dir = Path(PARAMS["train"]["smoke"]["out_dir"])
    assert pub.default_paths(PARAMS, smoke=True) == (smoke_dir / "results", smoke_dir / "best.pt")


def test_main_reports_refusal(run_copy, capsys) -> None:
    run_dir, ckpt, _ = run_copy
    args = ["--config", "configs/params.yaml", "--run-dir", str(run_dir), "--checkpoint", str(ckpt)]
    assert pub.main([*args, "--role", "comparison", "--register"]) == 1
    assert "never registered" in capsys.readouterr().err
