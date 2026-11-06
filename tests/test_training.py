"""Training pipeline (TASK-008). Synthetic images only; no pretrained download; CPU."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from src.serving.preprocess import PreprocessConfig, load_image, to_model_input  # noqa: E402
from src.training import train as train_mod  # noqa: E402
from src.training.augment import AugmentConfig, augment  # noqa: E402
from src.training.dataset import SplitDataset, TestSetAccessError, class_names  # noqa: E402
from src.training.metrics import accuracy, confusion_matrix, macro_f1, per_class_f1  # noqa: E402
from tests.training_helpers import (  # noqa: E402
    CLASSES,
    PARAMS,
    PARAMS_FILE,
    PER_SPLIT,
    REPO_ROOT,
    make_synthetic_data,
    tiny_params,
)

PRE = PreprocessConfig.from_params(PARAMS)
AUG = AugmentConfig.from_params(PARAMS)
# Only the promotion gate and final evaluation may read test_frozen.
ALLOW_TEST_ALLOWED = ("src/training/evaluate.py", "src/pipelines/")


@pytest.fixture
def synthetic_data(tmp_path: Path) -> tuple[Path, Path]:
    return make_synthetic_data(tmp_path)


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
            "--model",
            "mobilenetv3_small_100.lamb_in1k",
        ]
    )
    p = train_mod.apply_overrides(PARAMS, args)
    assert p["data"] == {"raw_dir": "/kaggle/input/neu", "split_file": "/kaggle/working/split.csv"}
    assert p["train"]["out_checkpoint"] == "/kaggle/working/m.pt"
    assert p["train"]["out_dir"] == "/kaggle/working/results"
    assert p["train"]["num_workers"] == 4
    assert p["train"]["model"] == "mobilenetv3_small_100.lamb_in1k"
    assert PARAMS["data"]["raw_dir"] != "/kaggle/input/neu"  # original params untouched
    assert PARAMS["train"]["model"] != p["train"]["model"]


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

    # Everything `make register` needs to publish the run later (ADR-020); no MLflow here.
    assert "mlflow" not in saved
    assert saved["model"] == PARAMS["train"]["model"] and saved["n_params"] > 0
    assert len(saved["run_uid"]) == 32
    timing = saved["timing"]
    assert 0 < timing["started_at_ms"] <= timing["finished_at_ms"]
    digest = hashlib.sha256((tmp_path / "best.pt").read_bytes()).hexdigest()
    assert saved["checkpoint_sha256"] == digest
    assert saved["provenance"]["runtime"] == "local"
    assert {"git_dirty", "torch", "timm"} <= set(saved["provenance"])

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


# ---------------------------------------------------------------- GPU support check


@pytest.mark.parametrize(
    ("cap", "arch", "ok"),
    [
        ((7, 5), ["sm_75", "sm_80", "sm_90", "compute_90"], True),  # T4
        ((6, 0), ["sm_75", "sm_80", "sm_90", "compute_90"], False),  # P100 on a CUDA 13 build
        ((12, 0), ["sm_75", "sm_90", "compute_90"], True),  # newer GPU via PTX JIT
        ((6, 0), ["sm_60", "sm_75"], True),
    ],
)
def test_gpu_supported(cap, arch, ok) -> None:
    assert train_mod.gpu_supported(cap, arch) is ok


def test_check_gpu_stops_on_unsupported_gpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Tesla P100-PCIE-16GB")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (6, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80"])
    with pytest.raises(RuntimeError, match="no kernels for Tesla P100"):
        train_mod.check_gpu(torch.device("cuda"))
    train_mod.check_gpu(torch.device("cpu"))  # no-op on CPU


def test_write_curve_ignores_jupyter_mplbackend(tmp_path: Path) -> None:
    """Kaggle runs us under Jupyter, which exports an inline MPLBACKEND our venv cannot load.

    Fresh process: matplotlib reads MPLBACKEND only on its first import.
    """
    import os
    import subprocess
    import sys

    code = (
        "import sys; from pathlib import Path; from src.training.train import write_curve; "
        "write_curve([{'epoch': 1, 'train_loss': 1.0, 'val_loss': 1.0, 'val_macro_f1': 0.5}], "
        "Path(sys.argv[1]))"
    )
    env = {**os.environ, "MPLBACKEND": "module://matplotlib_inline.backend_inline"}
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=REPO_ROOT, env=env, check=True)
    assert (tmp_path / "training_curves.png").stat().st_size > 0


def test_training_never_talks_to_mlflow() -> None:
    """Kaggle jobs hold no credentials (ADR-020): train.py must not import MLflow or tracking."""
    tree = ast.parse((REPO_ROOT / "src" / "training" / "train.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not [m for m in imported if m.startswith(("mlflow", "src.tracking", "dagshub"))]


def test_runtime_name(monkeypatch) -> None:
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    assert train_mod.runtime_name() == "local"
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    assert train_mod.runtime_name() == "kaggle"
