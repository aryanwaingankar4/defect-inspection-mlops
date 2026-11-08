"""TASK-015: ONNX export and parity (ADR-025). Fast tests use an untrained model and synthetic
images; the registry is the per-test local MLflow store (conftest), never DagsHub."""

import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import timm  # noqa: E402
import torch  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import load_params  # noqa: E402
from src.serving import onnx_model  # noqa: E402
from src.serving.onnx_model import OnnxClassifier  # noqa: E402
from src.serving.preprocess import PreprocessConfig, load_image, to_model_input  # noqa: E402
from src.tracking.artifact_policy import is_allowed_artifact  # noqa: E402
from src.training import export_onnx as ex  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
CFG = ex.ExportConfig.from_params(PARAMS)
CLASSES = ["a", "b", "c", "d", "e", "f"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- synthetic export (CI)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """An untrained efficientnet_b0 exported once for this module."""
    torch.manual_seed(0)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=6).eval()
    path = tmp_path_factory.mktemp("onnx") / "m.onnx"
    meta = {"classes": json.dumps(CLASSES), "registry_version": "7", "output": "raw logits"}
    size = PreprocessConfig.from_params(PARAMS).image_size
    ex.export_model(model, path, size, CFG.opset, meta)
    return model, path


def synthetic_batch(n: int) -> np.ndarray:
    """Grey images with stripes and noise, through the shared preprocessing."""
    rng = np.random.default_rng(n)
    pre = PreprocessConfig.from_params(PARAMS)
    out = []
    for i in range(n):
        img = (rng.normal(128, 30, (200, 200)) + 40 * np.sin(np.arange(200) / (3 + i))).clip(0, 255)
        out.append(to_model_input(load_image(_png(img)), pre))
    return np.stack(out)


def _png(img: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("n", [1, 5])
def test_onnx_matches_pytorch_on_synthetic_images(exported, n):
    model, path = exported
    x = synthetic_batch(n)
    got = OnnxClassifier(path).logits(x)
    ref = ex.torch_logits(model, x)
    assert got.shape == (n, 6)
    assert ex.tolerance_ratio(ref, got, CFG.atol, CFG.rtol) <= 1.0
    assert np.array_equal(got.argmax(axis=1), ref.argmax(axis=1))


def test_graph_is_one_file_with_raw_logits_and_metadata(exported):
    _, path = exported
    assert sorted(p.name for p in path.parent.iterdir()) == ["m.onnx"]  # no external data
    clf = OnnxClassifier(path)
    assert clf.classes == CLASSES and clf.metadata["registry_version"] == "7"
    inp, out = clf.session.get_inputs()[0], clf.session.get_outputs()[0]
    assert inp.name == "input" and inp.shape[1:] == [3, 224, 224] and isinstance(inp.shape[0], str)
    assert out.name == "logits" and out.shape[1:] == [6]


def test_onnx_logits_are_not_probabilities(exported):
    """ADR-025: the graph has no softmax and no temperature."""
    model, path = exported
    logits = OnnxClassifier(path).logits(synthetic_batch(3))
    assert not np.allclose(logits.sum(axis=1), 1.0)
    graph_ops = {n.op_type for n in __import__("onnx").load(str(path)).graph.node}
    assert "Softmax" not in graph_ops


def test_onnx_model_rejects_a_bad_batch(exported):
    with pytest.raises(ValueError):
        OnnxClassifier(exported[1]).logits(np.zeros((3, 224, 224), dtype=np.float32))


def test_serving_onnx_module_imports_no_torch():
    tree = ast.parse(Path(onnx_model.__file__).read_text(encoding="utf-8"))
    names = {
        a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    names |= {
        n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    }
    assert names <= {"__future__", "json", "collections", "pathlib", "numpy", "onnxruntime"}


def test_onnx_artifacts_are_allowed_by_the_policy():
    assert is_allowed_artifact("onnx/steel-defect-classifier-v2.fp32.onnx")
    assert is_allowed_artifact("onnx/export_report.json")


# ---------------------------------------------------------------- parity report


def test_parity_report_detects_class_and_routing_changes():
    ref = np.array([[4.0, 0.0, 0.0], [0.0, 1.0, 0.9], [0.0, 0.0, 5.0]])
    same = ex.parity_report(ref, ref + 1e-6, 1.0, 0.5, 1e-4, 1e-4)
    assert same["passed"] and same["same_class"] == 3 and same["same_routing"] == 3
    flipped = ref.copy()
    flipped[1] = [0.0, 0.9, 1.0]  # class changes, gap 0.1
    bad = ex.parity_report(ref, flipped, 1.0, 0.5, 1e-4, 1e-4)
    assert not bad["passed"] and bad["same_class"] == 2
    near = ex.parity_report(ref, ref, 1.0, 0.98, 1e-4, 1e-4)
    assert near["min_distance_to_threshold"] < 0.02


def test_tolerance_scales_with_the_logit():
    ref = np.array([[20.0, -17.0, 0.0]])
    got = ref + np.array([[2.5e-3, -1.3e-4, 0.0]])  # allowed at 20: 1e-4 + 1e-4 * 20 = 2.1e-3
    assert ex.tolerance_ratio(ref, got, 1e-4, 1e-4) > 1.0
    got[0, 0] = ref[0, 0] + 1.9e-3  # within 2.1e-3; -1.3e-4 at -17 is within 1.8e-3
    assert ex.tolerance_ratio(ref, got, 1e-4, 1e-4) <= 1.0
    assert ex.tolerance_ratio(np.zeros((1, 2)), np.full((1, 2), 2e-4), 1e-4, 1e-4) == 2.0


def test_run_refuses_test_frozen():
    params = copy.deepcopy(PARAMS)
    params["train"]["val_split"] = "test_frozen"
    with pytest.raises(ex.ExportError, match="test_frozen"):
        ex.run(params)


# ---------------------------------------------------------------- champion lookup (local store)


@pytest.fixture
def registry(offline_mlflow, tmp_path):
    """A local registry where version 1 of the main model holds @champion."""
    client = MlflowClient(tracking_uri=offline_mlflow)
    exp = client.create_experiment("export-test")
    run = client.create_run(exp)
    client.set_terminated(run.info.run_id)
    name = PARAMS["tracking"]["registered_model"]
    client.create_registered_model(name)
    mv = client.create_model_version(name, f"runs:/{run.info.run_id}/model", run.info.run_id)
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"weights")
    tags = {
        "model": PARAMS["train"]["model"],
        "checkpoint_sha256": _sha(ckpt),
        "temperature": "1.5",
        "review_threshold": "0.8",
    }
    for k, v in tags.items():
        client.set_model_version_tag(name, mv.version, k, v)
    client.set_registered_model_alias(name, PARAMS["promotion"]["champion_alias"], mv.version)
    res = tmp_path / "res"
    res.mkdir()
    (res / "mlflow_run.json").write_text(json.dumps({"run_id": run.info.run_id}), encoding="utf-8")
    params = copy.deepcopy(PARAMS)
    main = {**params["compare_val"]["models"][0], "checkpoint": "best.pt", "results_dir": "res"}
    params["compare_val"]["models"] = [main, params["compare_val"]["models"][1]]
    return {"client": client, "name": name, "version": str(mv.version), "params": params}


def test_champion_resolves_the_alias(registry):
    champ = ex.champion(registry["params"])
    assert champ["version"] == registry["version"]
    assert champ["tags"]["temperature"] == "1.5"


def test_checkpoint_for_accepts_the_matching_checkpoint(registry, tmp_path):
    champ = ex.champion(registry["params"])
    spec = ex.checkpoint_for(registry["params"], champ, tmp_path)
    assert spec.checkpoint == Path("best.pt")


def test_checkpoint_for_refuses_another_checkpoint(registry, tmp_path):
    champ = ex.champion(registry["params"])
    (tmp_path / "best.pt").write_bytes(b"retrained")
    with pytest.raises(ex.ExportError, match="SHA-256"):
        ex.checkpoint_for(registry["params"], champ, tmp_path)


def test_checkpoint_for_refuses_another_run(registry, tmp_path):
    champ = {**ex.champion(registry["params"]), "run_id": "0" * 32}
    with pytest.raises(ex.ExportError, match="another run"):
        ex.checkpoint_for(registry["params"], champ, tmp_path)


def test_champion_refuses_without_alias(registry):
    registry["client"].delete_registered_model_alias(
        registry["name"], PARAMS["promotion"]["champion_alias"]
    )
    with pytest.raises(ex.ExportError, match="make champion"):
        ex.champion(registry["params"])


def test_champion_refuses_missing_tags(registry):
    registry["client"].delete_model_version_tag(
        registry["name"], registry["version"], "temperature"
    )
    with pytest.raises(ex.ExportError, match="lacks tags"):
        ex.champion(registry["params"])


# ---------------------------------------------------------------- committed report


def test_committed_export_report_is_consistent():
    r = json.loads((REPO_ROOT / CFG.report_file).read_text(encoding="utf-8"))
    log = json.loads((REPO_ROOT / PARAMS["promotion"]["log_file"]).read_text(encoding="utf-8"))
    champ = [e for e in log if e["alias"] == PARAMS["promotion"]["champion_alias"]][-1]
    assert r["decision_record"] == "ADR-025"
    assert (r["version"], r["run_id"]) == (champ["version"], champ["run_id"])
    assert r["checkpoint_sha256"] == champ["checkpoint_sha256"]
    assert r["parity_applied_with"] == {
        "temperature": champ["temperature"],
        "review_threshold": champ["review_threshold"],
    }
    p = r["parity"]
    assert p["passed"] and p["n_images"] > 0
    assert (p["atol"], p["rtol"]) == (CFG.atol, CFG.rtol)
    assert p["max_tolerance_ratio"] <= 1.0
    assert all(b["passed"] and b["max_tolerance_ratio"] <= 1.0 for b in p["batch_check"])
    assert p["same_class"] == p["same_routing"] == p["n_images"]
    assert [b["batch"] for b in p["batch_check"]] == list(CFG.batch_sizes)
    assert r["parity_split"] == PARAMS["train"]["val_split"] != "test_frozen"
    assert r["opset"] == CFG.opset and r["graph"]["output_is"].startswith("raw logits")
    assert r["provenance"]["git_dirty"] is False


@pytest.mark.needs_data
@pytest.mark.needs_model
def test_export_reproduces_committed_parity(tmp_path, monkeypatch):
    """Re-export the champion checkpoint (registry answer taken from the committed report)."""
    monkeypatch.chdir(REPO_ROOT)
    r = json.loads((REPO_ROOT / CFG.report_file).read_text(encoding="utf-8"))
    spec = PARAMS["compare_val"]["models"][0]
    if not Path(spec["checkpoint"]).exists():
        pytest.skip(f"missing checkpoint {spec['checkpoint']}")
    champ = {
        "name": r["registered_model"],
        "version": r["version"],
        "alias": r["alias"],
        "run_id": r["run_id"],
        "tags": {
            "model": r["model"],
            "checkpoint_sha256": r["checkpoint_sha256"],
            "temperature": repr(r["parity_applied_with"]["temperature"]),
            "review_threshold": repr(r["parity_applied_with"]["review_threshold"]),
        },
    }
    monkeypatch.setattr(ex, "champion", lambda p: champ)
    params = copy.deepcopy(PARAMS)
    params["export"]["out_dir"] = str(tmp_path / "onnx")
    params["export"]["report_file"] = str(tmp_path / "report.json")
    new = ex.run(params)["report"]
    assert new["parity"]["passed"]
    for key in (
        "same_class",
        "same_routing",
        "routed_torch",
        "val_accuracy_torch",
        "val_accuracy_onnx",
    ):
        assert new["parity"][key] == r["parity"][key], key
    assert new["classes"] == r["classes"]
