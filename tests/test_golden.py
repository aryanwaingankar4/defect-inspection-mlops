"""TASK-020: golden images (ADR-017). CI part: the committed synthetic image, its preprocessing and
a small test model. Real-model part: the @champion bundle on the synthetic image (needs_model)
and on two fixed NEU val images (needs_data), never committed."""

import csv
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.config import load_params  # noqa: E402
from src.serving import app as api  # noqa: E402
from src.serving import golden as gd  # noqa: E402
from src.serving.bundle import MANIFEST_NAME  # noqa: E402
from src.serving.preprocess import PreprocessConfig  # noqa: E402
from src.serving.validation import ValidationConfig, decode_checked  # noqa: E402
from tests import serving_helpers as sh  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = sh.without_audit(load_params(REPO_ROOT / "configs" / "params.yaml"))  # exact routing
GOLDEN = REPO_ROOT / "tests" / "golden"
PNG = (GOLDEN / gd.SYNTHETIC_NAME).read_bytes()
EXPECTED = json.loads((GOLDEN / gd.EXPECTED_NAME).read_text(encoding="utf-8"))
TOL = EXPECTED["tolerance"]
KEY = "golden-key"


@pytest.fixture(autouse=True)
def database_url(tmp_path, monkeypatch):
    url = "sqlite:///" + (tmp_path / "db" / "p.db").as_posix()
    monkeypatch.setenv("DATABASE_URL", url)
    return url


def predict(client, data: bytes) -> dict:
    r = client.post(
        "/predict", files={"file": ("g", data, "image/png")}, headers={"X-API-Key": KEY}
    )
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------- CI: the synthetic image


def test_committed_image_is_exactly_the_formula():
    """Proves the committed PNG holds no dataset pixels (ADR-015, ADR-017)."""
    committed = np.asarray(Image.open(io.BytesIO(PNG)))
    assert committed.dtype == np.uint8 and committed.shape == (gd.SIZE, gd.SIZE)
    np.testing.assert_array_equal(committed, gd.synthetic_pixels())
    pixels_sha = gd.sha256_bytes(gd.synthetic_pixels().tobytes())
    assert pixels_sha == EXPECTED["synthetic_image"]["pixels_sha256"]


def test_only_synthetic_pngs_in_the_golden_folder():
    images = [p.name for p in GOLDEN.iterdir() if p.suffix.lower() != ".json"]
    assert images == [gd.SYNTHETIC_NAME]


def test_synthetic_image_passes_input_checks():
    d = decode_checked(PNG, ValidationConfig.from_params(PARAMS))
    assert (d.format, d.width, d.height) == ("PNG", gd.SIZE, gd.SIZE)


def test_preprocessing_of_the_golden_image_is_unchanged():
    got = gd.tensor_summary(PNG, PreprocessConfig.from_params(PARAMS))
    want = EXPECTED["synthetic_image"]["tensor"]
    assert got["shape"] == want["shape"]
    np.testing.assert_allclose(got["channel_mean"], want["channel_mean"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(got["channel_std"], want["channel_std"], rtol=0, atol=1e-6)
    assert got["samples"].keys() == want["samples"].keys()
    for k, v in want["samples"].items():
        assert got["samples"][k] == pytest.approx(v, abs=1e-6), k


def test_api_on_the_test_model_gives_the_independently_computed_output(tmp_path):
    """The tiny model's logits are SLOPES x mean(channel 0); the rest is done by hand here."""
    bundle = sh.make_bundle(tmp_path / "bundle", PARAMS["preprocess"])
    with TestClient(api.create_app(PARAMS, api_key=KEY, bundle_dir=bundle)) as c:
        body = predict(c, PNG)
    m0 = EXPECTED["synthetic_image"]["tensor"]["channel_mean"][0]
    z = [s * m0 / sh.TEMPERATURE for s in sh.SLOPES]
    top = max(z)
    lse = top + math.log(sum(math.exp(v - top) for v in z))
    conf = math.exp(top - lse)
    energy = -sh.TEMPERATURE * lse
    assert body["prediction"] == sh.CLASSES[z.index(top)]
    assert body["calibrated_confidence"] == pytest.approx(conf, abs=1e-5)
    assert body["energy"] == pytest.approx(energy, abs=1e-4)
    reasons = (["low_confidence"] if conf < sh.REVIEW else []) + (
        ["ood"] if energy > sh.OOD else []
    )
    assert body["reasons"] == reasons


# ---------------------------------------------------------------- real model


def champion_client():
    bundle_dir = REPO_ROOT / PARAMS["serving"]["bundle_dir"]
    if not (bundle_dir / MANIFEST_NAME).is_file():
        pytest.skip("no serving bundle: run `make serving-bundle`")
    manifest = json.loads((bundle_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    c = EXPECTED["champion"]
    if (manifest["version"], manifest["onnx_sha256"]) != (c["version"], c["onnx_sha256"]):
        pytest.fail(
            f"golden values are for version {c['version']}, the bundle is version "
            f"{manifest['version']}: run `make golden` on purpose and review the diff"
        )
    return TestClient(api.create_app(PARAMS, api_key=KEY, bundle_dir=bundle_dir))


def assert_matches(body: dict, want: dict) -> None:
    assert body["prediction"] == want["prediction"]
    assert body["reasons"] == want["reasons"]
    assert body["routed_to_review"] == want["routed_to_review"]
    assert body["calibrated_confidence"] == pytest.approx(want["calibrated_confidence"], abs=1e-4)
    assert body["energy"] == pytest.approx(want["energy"], abs=1e-3)


def entry(image_id: str) -> dict:
    return next(e for e in EXPECTED["champion"]["images"] if e["id"] == image_id)


@pytest.mark.needs_model
def test_champion_on_the_synthetic_image():
    want = entry("synthetic")
    assert want["image_sha256"] == gd.sha256_bytes(PNG)
    with champion_client() as c:
        assert_matches(predict(c, PNG), want)
        logits = gd.model_output(c.app.state.bundle, PNG, ValidationConfig.from_params(PARAMS))
    ref = np.asarray(want["logits"])
    gap = np.abs(np.asarray(logits["logits"]) - ref)
    assert (gap <= TOL["atol"] + TOL["rtol"] * np.abs(ref)).all()


@pytest.mark.needs_data
@pytest.mark.needs_model
@pytest.mark.parametrize("image_id", ["val_first", "val_lowest_confidence"])
def test_champion_on_real_neu_val_images(image_id):
    want = entry(image_id)
    path = REPO_ROOT / PARAMS["data"]["raw_dir"] / want["source"]
    if not path.is_file():
        pytest.skip("NEU images not on disk")
    data = path.read_bytes()
    assert gd.sha256_bytes(data) == want["image_sha256"], "the data file changed"
    with champion_client() as c:
        body = predict(c, data)
        out = gd.model_output(c.app.state.bundle, data, ValidationConfig.from_params(PARAMS))
    assert_matches(body, want)
    ref = np.asarray(want["logits"])
    assert (
        np.abs(np.asarray(out["logits"]) - ref) <= TOL["atol"] + TOL["rtol"] * np.abs(ref)
    ).all()


@pytest.mark.needs_data
def test_golden_val_images_follow_the_rule_and_are_never_test_frozen(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    chosen = {v["id"]: v["path"] for v in gd.val_images(PARAMS)}
    assert chosen == {e["id"]: e["source"] for e in EXPECTED["champion"]["images"][1:]}
    with (REPO_ROOT / PARAMS["data"]["split_file"]).open(encoding="utf-8", newline="") as f:
        split_of = {r["path"]: r["split"] for r in csv.DictReader(f)}
    assert {split_of[p] for p in chosen.values()} == {"val"}


def test_golden_refuses_test_frozen():
    params = {**PARAMS, "train": {**PARAMS["train"], "val_split": "test_frozen"}}
    with pytest.raises(gd.GoldenError, match="test_frozen"):
        gd.val_images(params)


def test_golden_predictions_must_belong_to_the_bundle_checkpoint(tmp_path):
    """ADR-035: after a promotion, golden reads the new champion's val_predictions.csv."""
    import copy as _copy

    from src.serving import golden as gd

    folder = tmp_path / "results"
    folder.mkdir()
    (folder / "val_predictions.csv").write_text("path,pred\n", encoding="utf-8")
    (folder / "calibration.json").write_text('{"checkpoint_sha256": "abc"}', encoding="utf-8")
    params = gd.with_predictions(_copy.deepcopy(PARAMS), folder / "val_predictions.csv")
    assert params["compare_val"]["models"][0]["results_dir"] == str(folder)
    gd.check_predictions_belong_to(params, {"checkpoint_sha256": "abc", "version": "3"})
    with pytest.raises(gd.GoldenError, match="another checkpoint"):
        gd.check_predictions_belong_to(params, {"checkpoint_sha256": "def", "version": "3"})
    with pytest.raises(gd.GoldenError, match="val_predictions.csv"):
        gd.with_predictions(PARAMS, folder / "other.csv")
