"""TASK-018: ``make serving-bundle`` against the per-test local MLflow store (ADR-028)."""

import copy
import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("onnx")
pytest.importorskip("mlflow")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from src.config import load_params  # noqa: E402
from src.serving import app as api  # noqa: E402
from src.serving import bundle as bd  # noqa: E402
from src.tracking import serving_bundle as sb  # noqa: E402
from tests import serving_helpers as sh  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
NAME = PARAMS["tracking"]["registered_model"]
ALIAS = PARAMS["promotion"]["champion_alias"]


@pytest.fixture
def client(offline_mlflow: str) -> MlflowClient:
    return MlflowClient(tracking_uri=offline_mlflow)


@pytest.fixture
def params(tmp_path):
    p = copy.deepcopy(PARAMS)
    p["serving"]["bundle_dir"] = str(tmp_path / "models" / "serving")
    return p


def register(client, tmp_path, *, precision="fp32", **tag_changes) -> dict:
    """A run holding tiny fp32 and int8 files under onnx/, registered and set as @champion."""
    exp = client.create_experiment(f"e-{tmp_path.name}")
    run_id = client.create_run(exp).info.run_id
    files = {}
    for prec in ("fp32", "int8"):
        path = sh.tiny_onnx(
            tmp_path / "src" / f"{NAME}-v1.{prec}.onnx",
            PARAMS["preprocess"],
            version="1",
            registered_model=NAME,
            precision=None if prec == "fp32" else "int8",
        )
        client.log_artifact(run_id, str(path), "onnx")
        files[prec] = path
    client.set_terminated(run_id)
    client.create_registered_model(NAME)
    version = client.create_model_version(NAME, f"runs:/{run_id}/model", run_id).version
    tags = {
        "checkpoint_sha256": "c" * 64,
        "temperature": "1.5",
        "review_threshold": "0.8161",
        "ood_energy_threshold": "-6.1446",
        "serving_precision": precision,
        "onnx_sha256": sh.sha(files["fp32"]),
        "onnx_artifact": f"onnx/{files['fp32'].name}",
        "onnx_int8_sha256": sh.sha(files["int8"]),
        "onnx_int8_artifact": f"onnx/{files['int8'].name}",
        **tag_changes,
    }
    for k, v in tags.items():
        if v is not None:
            client.set_model_version_tag(NAME, version, k, v)
    client.set_registered_model_alias(NAME, ALIAS, version)
    return {"run_id": run_id, "version": str(version), "files": files, "tags": tags}


def test_build_writes_a_bundle_the_api_serves(client, tmp_path, params):
    reg = register(client, tmp_path)
    m = sb.build(params)
    out = Path(params["serving"]["bundle_dir"])
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [bd.MANIFEST_NAME, f"{NAME}-v1.fp32.onnx"]
    )
    assert m["version"] == reg["version"] and m["run_id"] == reg["run_id"]
    assert m["precision"] == "fp32" and m["onnx_sha256"] == reg["tags"]["onnx_sha256"]
    assert (m["temperature"], m["review_threshold"], m["ood_energy_threshold"]) == (
        1.5,
        0.8161,
        -6.1446,
    )
    assert m["classes"] == sh.CLASSES and m["preprocess"] == PARAMS["preprocess"]
    assert json.loads((out / bd.MANIFEST_NAME).read_text(encoding="utf-8")) == m
    app = api.create_app(params, api_key="k")
    with TestClient(app) as c:
        assert c.get("/health").json()["model"]["version"] == reg["version"]
        r = c.post(
            "/predict",
            files={"file": ("x.png", sh.image_bytes(25.0), "image/png")},
            headers={"X-API-Key": "k"},
        )
        assert r.status_code == 200 and r.json()["thresholds"]["review"] == 0.8161


def test_int8_precision_takes_the_int8_file(client, tmp_path, params):
    register(client, tmp_path, precision="int8")
    m = sb.build(params)
    assert m["precision"] == "int8" and m["onnx_file"].endswith(".int8.onnx")


def test_hash_mismatch_keeps_the_old_bundle(client, tmp_path, params):
    out = Path(params["serving"]["bundle_dir"])
    sh.make_bundle(out, PARAMS["preprocess"])
    before = (out / bd.MANIFEST_NAME).read_bytes()
    register(client, tmp_path, onnx_sha256="0" * 64)
    with pytest.raises(sb.ServingBundleError, match="does not match"):
        sb.build(params)
    assert (out / bd.MANIFEST_NAME).read_bytes() == before


@pytest.mark.parametrize("missing", ["ood_energy_threshold", "temperature", "onnx_artifact"])
def test_missing_tags_are_refused(client, tmp_path, params, missing):
    register(client, tmp_path, **{missing: None})
    with pytest.raises(sb.ServingBundleError, match=missing):
        sb.build(params)


def test_bad_precision_tag_is_refused(client, tmp_path, params):
    register(client, tmp_path, precision="fp16")
    with pytest.raises(sb.ServingBundleError, match="serving_precision"):
        sb.build(params)


def test_no_champion(client, params):
    with pytest.raises(sb.ServingBundleError, match="no @"):
        sb.build(params)


# ---------------------------------------------------------------- the real bundle


@pytest.mark.needs_data
@pytest.mark.needs_model
def test_real_bundle_reproduces_committed_val_results():
    """The API on the real bundle gives the classes, calibrated confidences and routing counts
    recorded offline on val (calibrate's CSV from PyTorch; ood.json from the same ONNX file)."""
    bundle_dir = REPO_ROOT / PARAMS["serving"]["bundle_dir"]
    if not (bundle_dir / bd.MANIFEST_NAME).is_file():
        pytest.skip("no serving bundle: run `make serving-bundle`")
    res = REPO_ROOT / PARAMS["compare_val"]["models"][0]["results_dir"]
    ood = json.loads((res / "ood.json").read_text(encoding="utf-8"))
    with (res / "val_predictions.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    raw = REPO_ROOT / PARAMS["data"]["raw_dir"]
    if not (raw / rows[0]["path"]).is_file():
        pytest.skip("NEU images not on disk")
    app = api.create_app(PARAMS, api_key="k", bundle_dir=bundle_dir)
    with TestClient(app) as c:
        m = c.app.state.bundle.manifest
        assert m["version"] == ood["version"]
        assert m["temperature"] == ood["temperature"]
        assert m["review_threshold"] == ood["review_threshold"]
        assert m["ood_energy_threshold"] == ood["ood_energy_threshold"]
        low = flagged = routed = 0
        for r in rows:
            data = (raw / r["path"]).read_bytes()
            resp = c.post(
                "/predict", files={"file": ("x", data, "image/jpeg")}, headers={"X-API-Key": "k"}
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["prediction"] == r["pred"], r["path"]
            assert body["calibrated_confidence"] == pytest.approx(
                float(r["calibrated_confidence"]), abs=6e-5
            )
            low += "low_confidence" in body["reasons"]
            flagged += "ood" in body["reasons"]
            routed += body["routed_to_review"]
    v = ood["val"]
    assert (low, flagged, routed) == (
        v["routed_low_confidence"],
        v["flagged_ood"],
        v["routed_any_reason"],
    )
