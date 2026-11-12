"""TASK-017: energy OOD threshold and synthetic evaluation (ADR-027). Fast tests read no images."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from src.config import load_params  # noqa: E402
from src.data import synthetic_ood as so  # noqa: E402
from src.serving.preprocess import PreprocessConfig, load_image, to_model_input  # noqa: E402
from src.tracking.artifact_policy import is_allowed_artifact  # noqa: E402
from src.training import ood as od  # noqa: E402
from src.training.metrics import auroc, clopper_pearson_lower, fpr_at_tpr  # noqa: E402
from src.training.train import sha256_file  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
CFG = od.OodConfig.from_params(PARAMS)


# ---------------------------------------------------------------- synthetic images


def test_synthetic_images_are_deterministic_and_neu_shaped():
    a = so.generate(list(so.FAMILIES), 2, seed=1)
    b = so.generate(list(so.FAMILIES), 2, seed=1)
    assert a == b and len(a) == 2 * len(so.FAMILIES)
    assert so.generate(["stripes"], 2, seed=2) != so.generate(["stripes"], 2, seed=1)
    pre = PreprocessConfig.from_params(PARAMS)
    for _, png in a:
        img = load_image(png)
        assert img.mode == "L" and img.size == (200, 200)
        assert to_model_input(img, pre).shape == (3, pre.image_size, pre.image_size)


def test_synthetic_rejects_unknown_family():
    with pytest.raises(ValueError, match="unknown"):
        so.generate(["cats"], 1, seed=0)


def test_params_families_exist():
    assert set(CFG.families) <= set(so.FAMILIES)


# ---------------------------------------------------------------- metrics


def test_auroc_hand_values_and_ties():
    assert auroc([0, 1], [2, 3]) == 1.0
    assert auroc([2, 3], [0, 1]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5
    assert auroc([0, 2], [1, 3]) == 0.75  # pairs (0<1, 0<3, 2>1, 2<3)


def test_fpr_at_tpr():
    ood = np.arange(1, 101, dtype=float)  # 95% flagged from 6 upwards
    assert fpr_at_tpr(np.array([0.0, 5.0, 6.0, 50.0]), ood, 0.95) == 0.5


def test_clopper_pearson_lower():
    assert clopper_pearson_lower(0, 10, 0.95) == 0.0
    assert clopper_pearson_lower(10, 10, 0.95) == pytest.approx(0.05 ** (1 / 10))


def test_threshold_flags_the_budget_highest_energies():
    e = np.linspace(-20, -2, 180)
    t, m = od.threshold_for(e, 0.02)
    assert m == math.ceil(0.02 * 180) == 4 and int((e > t).sum()) == 4
    s = np.sort(e)
    assert t == pytest.approx((s[-4] + s[-5]) / 2)


# ---------------------------------------------------------------- guards


def test_run_refuses_test_frozen():
    params = copy.deepcopy(PARAMS)
    params["train"]["val_split"] = "test_frozen"
    with pytest.raises(od.OodError, match="test_frozen"):
        od.run(params)


def test_serving_onnx_follows_the_precision_tag(tmp_path):
    f = tmp_path / "m.fp32.onnx"
    f.write_bytes(b"onnx")
    (tmp_path / "r.json").write_text(json.dumps({"onnx_file": f.as_posix()}), encoding="utf-8")
    params = copy.deepcopy(PARAMS)
    params["export"]["report_file"] = str(tmp_path / "r.json")
    champ = {
        "alias": "champion",
        "tags": {"serving_precision": "fp32", "onnx_sha256": sha256_file(f)},
    }
    assert od.serving_onnx(params, champ) == (f, "fp32")
    with pytest.raises(od.OodError, match="not the fp32 file"):
        od.serving_onnx(params, {**champ, "tags": {**champ["tags"], "onnx_sha256": "0" * 64}})
    with pytest.raises(od.OodError, match="serving_precision"):
        od.serving_onnx(params, {**champ, "tags": {}})


def test_histogram_name_is_allowlisted():
    assert is_allowed_artifact(f"plots/{od.PLOT_NAME}")


# ---------------------------------------------------------------- committed result


def _committed() -> dict:
    res = REPO_ROOT / PARAMS["compare_val"]["models"][0]["results_dir"]
    return json.loads((res / CFG.out_file).read_text(encoding="utf-8"))


def test_committed_ood_result_is_consistent():
    r = _committed()
    res = REPO_ROOT / PARAMS["compare_val"]["models"][0]["results_dir"]
    cal = json.loads((res / "calibration.json").read_text(encoding="utf-8"))
    log = json.loads((REPO_ROOT / PARAMS["promotion"]["log_file"]).read_text(encoding="utf-8"))
    # results/train/ is the first champion's evidence (ADR-024); later promotions (ADR-033/035)
    # keep their own OOD result in results/retrain/<run_uid>/
    first = next(e for e in log if e["decision"] == "initial_champion")
    assert r["decision_record"] == "ADR-027" and r["version"] == first["version"]
    assert r["temperature"] == cal["temperature"] == first["temperature"]
    assert r["review_threshold"] == first["review_threshold"]
    assert r["calibration_sha256"] == sha256_file(res / "calibration.json")
    assert r["rule"]["id_flag_budget"] == CFG.id_flag_budget
    assert r["fitted_on"]["split"] == PARAMS["train"]["val_split"] != "test_frozen"
    v = r["val"]
    assert v["flagged_ood"] == math.ceil(round(CFG.id_flag_budget * v["n_images"], 9))
    assert v["routed_any_reason"] == v["routed_low_confidence"] + v["routed_ood_only"]
    lo, hi = v["ood_threshold_interval"]["ci_low"], v["ood_threshold_interval"]["ci_high"]
    assert lo <= hi
    syn = r["synthetic"]
    assert set(syn["per_family"]) == set(CFG.families)
    assert syn["pooled"]["n"] == CFG.n_per_family * len(CFG.families)
    assert sum(f["flagged_ood"] for f in syn["per_family"].values()) == syn["pooled"]["flagged_ood"]
    assert r["provenance"]["git_dirty"] is False
    assert (res / od.PLOT_NAME).is_file()


@pytest.mark.needs_data
@pytest.mark.needs_model
def test_ood_reproduces_committed(tmp_path, monkeypatch):
    """Re-run with the registry answer taken from the committed result (no DagsHub)."""
    monkeypatch.chdir(REPO_ROOT)
    r = _committed()
    onnx_file = Path(r["serving_onnx_file"])
    if not onnx_file.exists():
        pytest.skip(f"missing {onnx_file}")
    champ = {
        "name": r["registered_model"],
        "version": r["version"],
        "alias": r["alias"],
        "run_id": r["run_id"],
        "tags": {
            "temperature": repr(r["temperature"]),
            "review_threshold": repr(r["review_threshold"]),
            "serving_precision": r["serving_precision"],
        },
    }
    monkeypatch.setattr(od, "champion", lambda p: champ)
    monkeypatch.setattr(od, "serving_onnx", lambda p, c: (onnx_file, r["serving_precision"]))
    params = copy.deepcopy(PARAMS)
    res = tmp_path / "res"
    res.mkdir()
    main = {**params["compare_val"]["models"][0]}
    # outputs go to tmp: copy the files ood.py reads from the results dir
    for name in ("calibration.json", "val_metrics.json"):
        (res / name).write_bytes((REPO_ROOT / main["results_dir"] / name).read_bytes())
    monkeypatch.setattr(
        od,
        "specs_from_params",
        lambda cfg: [
            type(
                "S",
                (),
                {"name": main["name"], "checkpoint": Path(main["checkpoint"]), "results_dir": res},
            )()
        ],
    )
    monkeypatch.setattr(
        od, "load_temperature", lambda p, s: (r["temperature"], res / "calibration.json")
    )
    new = od.run(params)["result"]
    assert new["ood_energy_threshold"] == pytest.approx(r["ood_energy_threshold"], abs=1e-9)
    for key in ("flagged_ood", "routed_any_reason", "combined_rule"):
        assert new["val"][key] == r["val"][key], key
    assert new["synthetic"]["pooled"]["flagged_ood"] == r["synthetic"]["pooled"]["flagged_ood"]
