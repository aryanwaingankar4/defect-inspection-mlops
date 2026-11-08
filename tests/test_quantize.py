"""TASK-016: static INT8 QDQ (ADR-002, ADR-026). Fast tests use an untrained model and
synthetic images; never DagsHub, never val/test images."""

import copy
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import onnx  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402

from src.config import load_params  # noqa: E402
from src.serving.onnx_model import OnnxClassifier  # noqa: E402
from src.tracking.artifact_policy import is_allowed_artifact  # noqa: E402
from src.training import export_onnx as ex  # noqa: E402
from src.training import quantize as qz  # noqa: E402
from src.training.metrics import clopper_pearson_upper  # noqa: E402
from src.training.train import sha256_file  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
CFG = qz.QuantizeConfig.from_params(PARAMS)
CLASSES = ["a", "b", "c", "d", "e", "f"]


# ---------------------------------------------------------------- synthetic quantisation (CI)


@pytest.fixture(scope="module")
def quantized(tmp_path_factory):
    torch.manual_seed(0)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=6).eval()
    d = tmp_path_factory.mktemp("q")
    fp32, int8 = d / "m.fp32.onnx", d / "m.int8.onnx"
    ex.export_model(model, fp32, 224, 18, {"classes": json.dumps(CLASSES), "registry_version": "7"})
    rng = np.random.default_rng(0)
    batches = [rng.normal(size=(4, 3, 224, 224)).astype(np.float32) for _ in range(2)]
    qz.quantize_file(fp32, int8, batches, CFG)
    return fp32, int8


def test_int8_graph_is_qdq_smaller_and_keeps_metadata(quantized):
    fp32, int8 = quantized
    ops = {n.op_type for n in onnx.load(str(int8)).graph.node}
    assert {"QuantizeLinear", "DequantizeLinear"} <= ops
    assert int8.stat().st_size < 0.4 * fp32.stat().st_size
    clf = OnnxClassifier(int8)
    assert clf.classes == CLASSES and clf.metadata["registry_version"] == "7"
    assert clf.metadata["precision"] == "int8"
    assert clf.metadata["fp32_sha256"] == sha256_file(fp32)
    assert clf.metadata["calibration_split"] == CFG.calibration_split


def test_int8_runs_with_a_dynamic_batch(quantized):
    clf = OnnxClassifier(quantized[1])
    for n in (1, 3):
        out = clf.logits(np.zeros((n, 3, 224, 224), dtype=np.float32))
        assert out.shape == (n, 6) and np.isfinite(out).all()


def test_batch_reader_feeds_each_batch_once():
    batches = [np.zeros((2, 3, 4, 4)), np.ones((1, 3, 4, 4))]
    reader = qz.BatchReader(batches)
    first, second = reader.get_next(), reader.get_next()
    assert first["input"].dtype == np.float32 and second["input"].shape == (1, 3, 4, 4)
    assert reader.get_next() is None


@pytest.mark.parametrize("split", ["val", "test_frozen", "production_stream"])
def test_calibration_refuses_held_out_and_non_training_splits(split):
    with pytest.raises(qz.QuantizeError):
        qz.check_calibration_split(split, PARAMS)


def test_calibration_split_in_params_is_a_training_split():
    qz.check_calibration_split(CFG.calibration_split, PARAMS)


def test_run_refuses_test_frozen_as_val():
    params = copy.deepcopy(PARAMS)
    params["train"]["val_split"] = "test_frozen"
    with pytest.raises(qz.QuantizeError, match="test_frozen"):
        qz.run(params)


# ---------------------------------------------------------------- rule + routing


def _scores(f1s: list[float]) -> dict:
    return {
        "macro_f1": float(np.mean(f1s)),
        "per_class": {c: {"f1": f} for c, f in zip(CLASSES, f1s, strict=True)},
    }


def test_rule_passes_and_fails_on_each_check():
    fp32 = _scores([1.0] * 6)
    ok = qz.decide(fp32, _scores([1.0] * 5 + [0.985]), {"risk_ucb": 0.03}, CFG, 0.05)
    assert ok["passed"] and ok["serving_precision"] == "int8"
    one_class = qz.decide(fp32, _scores([1.0] * 5 + [0.95]), {"risk_ucb": 0.03}, CFG, 0.05)
    assert not one_class["checks"]["max_class_f1_drop"]["passed"]
    assert one_class["checks"]["max_class_f1_drop"]["worst_class"] == "f"
    assert one_class["serving_precision"] == "fp32"
    macro = qz.decide(fp32, _scores([0.985] * 6), {"risk_ucb": 0.03}, CFG, 0.05)
    assert not macro["checks"]["macro_f1_drop"]["passed"] and not macro["passed"]
    risk = qz.decide(fp32, fp32, {"risk_ucb": 0.06}, CFG, 0.05)
    assert risk["serving_precision"] == "fp32"


def test_routing_counts_and_bound():
    y = np.array([0, 0, 1, 1])
    logits = np.array([[5.0, 0.0], [0.1, 0.0], [0.0, 5.0], [5.0, 0.0]])  # last one wrong, confident
    r = qz.routing(y, logits, 1.0, 0.9, 0.95)
    assert (r["n_routed"], r["errors_routed"], r["errors_accepted"]) == (1, 0, 1)
    assert r["risk_ucb"] == pytest.approx(clopper_pearson_upper(1, 3, 0.95))


def test_fp32_input_refuses_a_changed_file(tmp_path):
    f = tmp_path / "m.fp32.onnx"
    f.write_bytes(b"onnx")
    report = {"version": "2", "onnx_file": f.as_posix(), "onnx_sha256": sha256_file(f)}
    (tmp_path / "r.json").write_text(json.dumps(report), encoding="utf-8")
    params = copy.deepcopy(PARAMS)
    params["export"]["report_file"] = str(tmp_path / "r.json")
    champ = {"version": "2", "tags": {"onnx_sha256": sha256_file(f)}}
    assert qz.fp32_input(params, champ)[0] == f
    with pytest.raises(qz.QuantizeError, match="another version"):
        qz.fp32_input(params, {**champ, "version": "3"})
    f.write_bytes(b"changed")
    with pytest.raises(qz.QuantizeError, match="not the FP32 file"):
        qz.fp32_input(params, champ)


def test_int8_artifacts_are_allowed_by_the_policy():
    assert is_allowed_artifact("onnx/steel-defect-classifier-v2.int8.onnx")
    assert is_allowed_artifact("onnx/quantization_report.json")


# ---------------------------------------------------------------- committed report


def _committed() -> dict:
    return json.loads((REPO_ROOT / CFG.report_file).read_text(encoding="utf-8"))


def test_committed_quantization_report_is_consistent():
    r = _committed()
    export = json.loads((REPO_ROOT / PARAMS["export"]["report_file"]).read_text(encoding="utf-8"))
    with (REPO_ROOT / PARAMS["data"]["split_file"]).open(encoding="utf-8") as f:
        n_cal = sum(row["split"] == CFG.calibration_split for row in csv.DictReader(f))
    s = r["settings"]
    assert r["decision_record"] == "ADR-026" and r["compared_on"] == PARAMS["train"]["val_split"]
    assert (r["version"], r["fp32_sha256"]) == (export["version"], export["onnx_sha256"])
    assert s["calibration_split"] == CFG.calibration_split and s["calibration_images"] == n_cal
    assert (s["calibrate_method"], s["per_channel"], s["format"]) == (
        CFG.calibrate_method,
        CFG.per_channel,
        "QDQ",
    )
    assert r["applied_with"] == export["parity_applied_with"]
    rule = r["rule"]
    for name, limit in [
        ("max_class_f1_drop", CFG.max_class_f1_drop),
        ("macro_f1_drop", CFG.max_macro_f1_drop),
        ("int8_risk_ucb", PARAMS["threshold"]["max_risk_ucb"]),
    ]:
        check = rule["checks"][name]
        assert check["limit"] == limit and check["passed"] == (check["value"] <= limit + 1e-12)
    expected = "int8" if all(c["passed"] for c in rule["checks"].values()) else "fp32"
    assert rule["serving_precision"] == r["serving_precision"] == expected
    assert r["agreement"]["same_class"] <= r["n_images"]
    assert r["provenance"]["git_dirty"] is False


@pytest.mark.needs_data
@pytest.mark.needs_model
def test_quantization_reproduces_committed(tmp_path, monkeypatch):
    """Re-quantise a copy of the FP32 file (registry answer from the committed reports)."""
    monkeypatch.chdir(REPO_ROOT)
    r = _committed()
    export = json.loads(Path(PARAMS["export"]["report_file"]).read_text(encoding="utf-8"))
    src = Path(r["fp32_file"])
    if not src.exists():
        pytest.skip(f"missing {src}")
    copy_fp32 = tmp_path / src.name
    shutil.copy(src, copy_fp32)
    champ = {
        "name": r["registered_model"],
        "version": r["version"],
        "alias": r["alias"],
        "run_id": r["run_id"],
        "tags": {
            "temperature": repr(r["applied_with"]["temperature"]),
            "review_threshold": repr(r["applied_with"]["review_threshold"]),
            "onnx_sha256": r["fp32_sha256"],
        },
    }
    monkeypatch.setattr(qz, "champion", lambda p: champ)
    monkeypatch.setattr(qz, "fp32_input", lambda p, c: (copy_fp32, export))
    params = copy.deepcopy(PARAMS)
    params["quantize"]["report_file"] = str(tmp_path / "q.json")
    new = qz.run(params)["report"]
    for key in ("fp32", "int8"):
        for part in ("accuracy", "macro_f1", "per_class", "routing"):
            assert new[key][part] == r[key][part], (key, part)
    assert new["agreement"]["same_class"] == r["agreement"]["same_class"]
    assert new["serving_precision"] == r["serving_precision"]
