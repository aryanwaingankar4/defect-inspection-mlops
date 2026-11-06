"""Training pipeline (TASK-008). Synthetic images only; no pretrained download; CPU."""

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("mlflow")

from src.config import load_params  # noqa: E402
from src.data.split import write_split_csv  # noqa: E402
from src.serving.preprocess import PreprocessConfig, load_image, to_model_input  # noqa: E402
from src.training import train as train_mod  # noqa: E402
from src.training.augment import AugmentConfig, augment  # noqa: E402
from src.training.dataset import SplitDataset, TestSetAccessError, class_names  # noqa: E402
from src.training.metrics import accuracy, confusion_matrix, macro_f1, per_class_f1  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS_FILE = REPO_ROOT / "configs" / "params.yaml"
PARAMS = load_params(PARAMS_FILE)
PRE = PreprocessConfig.from_params(PARAMS)
AUG = AugmentConfig.from_params(PARAMS)
CLASSES = ["alpha", "beta", "gamma"]
PER_SPLIT = {"train_v1": 3, "val": 2, "test_frozen": 1}
# Only the promotion gate and final evaluation may read test_frozen.
ALLOW_TEST_ALLOWED = ("src/training/evaluate.py", "src/pipelines/")


@pytest.fixture
def synthetic_data(tmp_path: Path) -> tuple[Path, Path]:
    """Tiny NEU-like tree + split CSV. Each class has its own brightness so it is learnable."""
    raw = tmp_path / "raw"
    rows = []
    rng = np.random.default_rng(0)
    for ci, cls in enumerate(CLASSES):
        folder = raw / "NEU-DET" / "train" / "images" / cls
        folder.mkdir(parents=True)
        i = 0
        for split, n in PER_SPLIT.items():
            for _ in range(n):
                base = 40 + 80 * ci
                arr = np.clip(base + rng.normal(0, 10, (40, 40)), 0, 255).astype(np.uint8)
                rel = f"NEU-DET/train/images/{cls}/{cls}_{i}.jpg"
                Image.fromarray(arr).convert("RGB").save(raw / rel, quality=95)
                rows.append((rel, cls, split))
                i += 1
    csv_path = tmp_path / "split.csv"
    write_split_csv(rows, csv_path)
    return raw, csv_path


def tiny_params(raw: Path, csv_path: Path, out: Path, **train_overrides) -> dict:
    p = copy.deepcopy(PARAMS)
    p["data"] = {"raw_dir": raw.as_posix(), "split_file": csv_path.as_posix()}
    p["preprocess"]["image_size"] = 32
    p["train"].update(
        pretrained=False,
        batch_size=4,
        max_epochs=1,
        patience=1,
        num_workers=0,
        out_checkpoint=(out / "best.pt").as_posix(),
        out_dir=(out / "results").as_posix(),
        **train_overrides,
    )
    return p


# ---------------------------------------------------------------- dataset


def test_parity_dataset_without_augmentation_equals_serving(synthetic_data) -> None:
    raw, csv_path = synthetic_data
    ds = SplitDataset(csv_path, raw, ["train_v1", "val"], CLASSES, PRE)
    assert len(ds) == len(CLASSES) * (PER_SPLIT["train_v1"] + PER_SPLIT["val"])
    for i in range(len(ds)):
        x, label = ds[i]
        served = torch.from_numpy(to_model_input(load_image(raw / ds.paths[i]), PRE))
        assert x.dtype == torch.float32
        assert torch.equal(x, served)
        assert CLASSES[label] in ds.paths[i]


def test_test_frozen_is_refused_without_flag(synthetic_data) -> None:
    raw, csv_path = synthetic_data
    with pytest.raises(TestSetAccessError):
        SplitDataset(csv_path, raw, ["test_frozen"], CLASSES, PRE)
    with pytest.raises(TestSetAccessError):
        SplitDataset(csv_path, raw, ["train_v1", "test_frozen"], CLASSES, PRE)


def test_test_frozen_loads_with_explicit_flag(synthetic_data) -> None:
    raw, csv_path = synthetic_data
    ds = SplitDataset(csv_path, raw, ["test_frozen"], CLASSES, PRE, allow_test=True)
    assert len(ds) == len(CLASSES) * PER_SPLIT["test_frozen"]


def test_allow_test_true_only_in_gate_or_evaluation() -> None:
    """A code scan: nothing in src/ except evaluation / the gate may pass allow_test=True."""
    offenders = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.startswith(ALLOW_TEST_ALLOWED):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            is_allow_test = isinstance(node, ast.keyword) and node.arg == "allow_test"
            if is_allow_test and not (
                isinstance(node.value, ast.Constant) and node.value.value is False
            ):
                offenders.append(f"{rel}:{node.value.lineno}")
    assert not offenders, f"allow_test used outside evaluation/gate: {offenders}"


def test_images_per_class_limit_and_class_names(synthetic_data) -> None:
    raw, csv_path = synthetic_data
    ds = SplitDataset(csv_path, raw, ["train_v1"], CLASSES, PRE, images_per_class=2)
    assert ds.class_counts() == {0: 2, 1: 2, 2: 2}
    assert class_names(csv_path, ["train_v1"]) == CLASSES


def test_unknown_class_rejected(synthetic_data) -> None:
    raw, csv_path = synthetic_data
    with pytest.raises(ValueError, match="not in the model's class list"):
        SplitDataset(csv_path, raw, ["train_v1"], ["alpha", "beta"], PRE)


# ---------------------------------------------------------------- augmentation


def textured(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (60, 80), dtype=np.uint8), mode="L")


def test_augment_is_repeatable_and_keeps_mode_and_size() -> None:
    img = textured()
    a = augment(img, AUG, np.random.default_rng(7))
    b = augment(img, AUG, np.random.default_rng(7))
    assert a.tobytes() == b.tobytes()
    assert (a.mode, a.size) == ("L", img.size)
    outputs = {augment(img, AUG, np.random.default_rng(s)).tobytes() for s in range(5)}
    assert len(outputs) > 1


def test_augment_with_zero_strength_is_identity() -> None:
    off = AugmentConfig(hflip_p=0, vflip_p=0, rotation_deg=0, brightness=0, contrast=0)
    img = textured()
    assert augment(img, off, np.random.default_rng(0)).tobytes() == img.tobytes()


def test_rotation_fills_corners_with_mean_grey() -> None:
    only_rot = AugmentConfig(hflip_p=0, vflip_p=0, rotation_deg=10, brightness=0, contrast=0)
    flat = Image.new("L", (100, 100), color=100)
    out = np.asarray(augment(flat, only_rot, np.random.default_rng(3)))
    assert out.min() == 100 and out.max() == 100  # black fill would put 0s in the corners


def test_augment_rejects_non_l_and_bad_config() -> None:
    with pytest.raises(ValueError):
        augment(Image.new("RGB", (10, 10)), AUG, np.random.default_rng(0))
    with pytest.raises(ValueError):
        AugmentConfig(hflip_p=1.5, vflip_p=0, rotation_deg=0, brightness=0, contrast=0)
    with pytest.raises(ValueError):
        AugmentConfig(hflip_p=0, vflip_p=0, rotation_deg=0, brightness=1.0, contrast=0)


# ---------------------------------------------------------------- metrics + loop pieces


def test_metrics_on_hand_made_case() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    assert confusion_matrix(y_true, y_pred, 3).tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    # class 0: P=1/2 R=1/2 F1=0.5; class 1: P=2/3 R=1 F1=0.8; class 2: P=1 R=1/2 F1=2/3
    np.testing.assert_allclose(per_class_f1(y_true, y_pred, 3), [0.5, 0.8, 2 / 3])
    assert macro_f1(y_true, y_pred, 3) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert accuracy(y_true, y_pred) == pytest.approx(4 / 6)
    assert macro_f1(y_true, y_true, 3) == 1.0


def test_absent_class_scores_zero_and_bad_labels_rejected() -> None:
    np.testing.assert_allclose(per_class_f1([0, 1], [0, 1], 3), [1.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        confusion_matrix([0, 3], [0, 0], 3)


def test_early_stopping() -> None:
    stop = train_mod.EarlyStopping(patience=2)
    assert stop.update(0.5) and not stop.should_stop
    assert not stop.update(0.5)  # a tie is not an improvement
    assert stop.update(0.6)
    assert not stop.update(0.4) and not stop.should_stop
    assert not stop.update(0.4) and stop.should_stop


def test_balanced_sampler_equalises_classes() -> None:
    labels = [0] * 90 + [1] * 10
    gen = torch.Generator().manual_seed(0)
    sampler = train_mod.balanced_sampler(labels, gen)
    drawn = [labels[i] for _ in range(50) for i in sampler]
    share_minority = sum(drawn) / len(drawn)
    assert 0.45 < share_minority < 0.55


# ---------------------------------------------------------------- CLI + device guard


def test_cli_overrides_paths_and_workers() -> None:
    args = train_mod.parse_args(
        [
            "--config",
            str(PARAMS_FILE),
            "--raw-dir",
            "/kaggle/input/neu",
            "--split-file",
            "/kaggle/working/split.csv",
            "--out-checkpoint",
            "/kaggle/working/m.pt",
            "--out-dir",
            "/kaggle/working/results",
            "--num-workers",
            "4",
        ]
    )
    p = train_mod.apply_overrides(PARAMS, args)
    assert p["data"] == {"raw_dir": "/kaggle/input/neu", "split_file": "/kaggle/working/split.csv"}
    assert p["train"]["out_checkpoint"] == "/kaggle/working/m.pt"
    assert p["train"]["out_dir"] == "/kaggle/working/results"
    assert p["train"]["num_workers"] == 4
    assert PARAMS["data"]["raw_dir"] != "/kaggle/input/neu"  # original params untouched


def test_smoke_defaults_then_cli_wins() -> None:
    smoke = PARAMS["train"]["smoke"]
    p = train_mod.apply_overrides(PARAMS, train_mod.parse_args(["--config", "x", "--smoke"]))
    assert p["train"]["max_epochs"] == smoke["max_epochs"]
    assert p["train"]["out_dir"].startswith(smoke["out_dir"])
    p = train_mod.apply_overrides(
        PARAMS, train_mod.parse_args(["--config", "x", "--smoke", "--out-dir", "elsewhere"])
    )
    assert p["train"]["out_dir"] == "elsewhere"


def test_device_guard(capsys) -> None:
    cpu, gpu = torch.device("cpu"), torch.device("cuda")
    assert train_mod.check_device(gpu, smoke=False, allow_cpu=False)
    assert train_mod.check_device(cpu, smoke=True, allow_cpu=False)
    assert not train_mod.check_device(cpu, smoke=False, allow_cpu=False)
    assert "no GPU found" in capsys.readouterr().err
    assert train_mod.check_device(cpu, smoke=False, allow_cpu=True)
    assert "no GPU found" in capsys.readouterr().err  # still warns


def test_main_refuses_full_cpu_run(monkeypatch, capsys) -> None:
    monkeypatch.setattr(train_mod, "resolve_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(train_mod, "run", lambda *a, **k: pytest.fail("training must not start"))
    assert train_mod.main(["--config", str(PARAMS_FILE)]) == train_mod.EXIT_NEEDS_GPU
    assert "--allow-cpu" in capsys.readouterr().err


# ---------------------------------------------------------------- end-to-end smoke


def test_tiny_training_run_writes_outputs(synthetic_data, tmp_path) -> None:
    raw, csv_path = synthetic_data
    train_mod.run(tiny_params(raw, csv_path, tmp_path), torch.device("cpu"))

    out = tmp_path / "results"
    saved = json.loads((out / "val_metrics.json").read_text(encoding="utf-8"))
    assert saved["classes"] == CLASSES
    assert saved["epochs_run"] == 1 and saved["best_epoch"] == 1
    assert saved["n_train"] == 9 and saved["n_val"] == 6
    assert 0.0 <= saved["val_macro_f1"] <= 1.0
    assert set(saved["val_per_class_f1"]) == set(CLASSES)
    assert {"git_commit", "dvc_data_md5", "data_fingerprint"} <= set(saved["provenance"])
    assert saved["params"]["train"]["seed"] == PARAMS["train"]["seed"]
    assert (out / "training_curves.png").stat().st_size > 0
    assert len((out / "training_curves.csv").read_text().splitlines()) == 2
    assert (out / "params.yaml").exists()

    # Tracked in the (local, per-test) MLflow store under the real names (not smoke).
    from mlflow import MlflowClient

    tracked = saved["mlflow"]
    assert tracked["experiment"] == PARAMS["tracking"]["experiment"]
    assert tracked["registered_model"] == PARAMS["tracking"]["registered_model"]
    run = MlflowClient().get_run(tracked["run_id"])
    assert run.data.tags["git_commit"] == saved["provenance"]["git_commit"]
    assert run.data.tags["data_fingerprint"] == saved["provenance"]["data_fingerprint"]
    assert run.data.params["train.lr"] == str(PARAMS["train"]["lr"])
    assert "best_val_macro_f1" in run.data.metrics

    ckpt = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=True)
    assert ckpt["classes"] == CLASSES and ckpt["preprocess"]["image_size"] == 32
    model = train_mod.build_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])


@pytest.mark.slow
def test_tiny_training_run_is_repeatable(synthetic_data, tmp_path) -> None:
    """Same seed, same data, same CPU -> same weights and metrics."""
    raw, csv_path = synthetic_data
    cpu = torch.device("cpu")
    m1 = train_mod.run(tiny_params(raw, csv_path, tmp_path / "a"), cpu)
    m2 = train_mod.run(tiny_params(raw, csv_path, tmp_path / "b"), cpu)
    a = torch.load(tmp_path / "a" / "best.pt", map_location="cpu", weights_only=True)
    b = torch.load(tmp_path / "b" / "best.pt", map_location="cpu", weights_only=True)
    for k, v in a["state_dict"].items():
        assert torch.equal(v, b["state_dict"][k]), k
    assert m1["val_macro_f1"] == m2["val_macro_f1"]
    assert m1["provenance"]["data_fingerprint"] == m2["provenance"]["data_fingerprint"]
