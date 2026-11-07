"""Kaggle launcher rendering and push guards (TASK-010). No network, no Kaggle account."""

import ast
import json
from pathlib import Path

import pytest

from src.config import load_params
from src.pipelines import kaggle_job as kj

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
TEMPLATE = REPO_ROOT / PARAMS["kaggle"]["notebook_template"]
SHA = "0123456789abcdef0123456789abcdef01234567"


def fake_git(status: str = "", remote_branches: str = "  origin/main") -> kj.Git:
    def run(*args: str) -> str:
        if args[0] == "status":
            return status
        if args[0] == "rev-parse":
            return SHA
        if args[:2] == ("branch", "-r"):
            return remote_branches
        raise AssertionError(args)

    return run


def test_pinned_commit_ok() -> None:
    assert kj.pinned_commit(fake_git()) == SHA


def test_pinned_commit_refuses_dirty_tree() -> None:
    with pytest.raises(kj.LaunchError, match="uncommitted"):
        kj.pinned_commit(fake_git(status=" M src/training/train.py"))


def test_pinned_commit_refuses_unpushed_commit() -> None:
    with pytest.raises(kj.LaunchError, match="not pushed"):
        kj.pinned_commit(fake_git(remote_branches=""))


def test_build_kernel_dir(tmp_path: Path) -> None:
    out = kj.build_kernel_dir(PARAMS, "me", SHA, tmp_path / "kernel")
    nb_text = (out / kj.NOTEBOOK_FILE).read_text(encoding="utf-8")
    nb = json.loads(nb_text)
    assert "{{" not in nb_text
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert f'COMMIT = "{SHA}"' in code
    assert PARAMS["kaggle"]["repo_url"] in code
    assert "src.training.train" in code and "--raw-dir" in code

    meta = json.loads((out / kj.METADATA_FILE).read_text(encoding="utf-8"))
    k = PARAMS["kaggle"]
    assert meta["id"] == f"me/{k['kernel_slug']}"
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is True and meta["enable_internet"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["dataset_sources"] == [f"me/{k['dataset_slug']}"]
    assert meta["code_file"] == kj.NOTEBOOK_FILE


def code_of(notebook_text: str) -> str:
    nb = json.loads(notebook_text)
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_train_args_reach_the_training_command(tmp_path: Path) -> None:
    model = "mobilenetv3_small_100.lamb_in1k"
    out = kj.build_kernel_dir(PARAMS, "me", SHA, tmp_path / "k", kj.train_args(model))
    code = code_of((out / kj.NOTEBOOK_FILE).read_text(encoding="utf-8"))
    assert f'TRAIN_ARGS = "--model {model}".split()  # noqa' in code
    assert "*TRAIN_ARGS," in code
    ast.parse(code)  # the rendered notebook is valid Python

    out = kj.build_kernel_dir(PARAMS, "me", SHA, tmp_path / "k2")
    assert 'TRAIN_ARGS = "".split()  # noqa' in code_of((out / kj.NOTEBOOK_FILE).read_text("utf-8"))
    assert kj.train_args(None) == []


@pytest.mark.parametrize("bad", ['x" ; import os #', "a b", "a\\b", "", "-x"])
def test_unsafe_train_args_are_refused(bad: str, tmp_path: Path) -> None:
    with pytest.raises(kj.LaunchError):
        kj.train_args(bad)
    with pytest.raises(kj.LaunchError):
        kj.build_kernel_dir(PARAMS, "me", SHA, tmp_path / "k", ["--model", bad])


def test_template_has_no_credentials() -> None:
    """ADR-020: the Kaggle job holds no secrets; the laptop publishes to MLflow."""
    text = TEMPLATE.read_text(encoding="utf-8")
    for word in ("kaggle_secrets", "UserSecretsClient", "get_secret", "MLFLOW", "DAGSHUB"):
        assert word not in text, word


def test_template_writes_where_register_reads() -> None:
    """The launcher's output paths match kaggle.output_results / output_checkpoint."""
    code = code_of(TEMPLATE.read_text(encoding="utf-8"))
    k = PARAMS["kaggle"]
    assert 'f"{OUT}/' + k["output_results"] + '"' in code
    assert 'f"{OUT}/' + k["output_checkpoint"] + '"' in code


def test_render_requires_every_placeholder() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        kj.render_notebook(template, {"COMMIT": SHA})


def test_template_has_placeholders_and_no_logic() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    for key in kj.PLACEHOLDERS:
        assert "{{" + key + "}}" in text
    code = "".join(
        "".join(c["source"]) for c in json.loads(text)["cells"] if c["cell_type"] == "code"
    )
    assert "import torch" not in code and "import timm" not in code  # logic lives in src/


def test_template_never_prints_secrets() -> None:
    """print() may only show string literals or the dataset path: never a variable that could
    hold a secret, os.environ, or a get_secret() result."""
    nb = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    code = chr(10).join("".join(c["source"]) for c in cells)
    allowed_names = {"raw_dir"}
    prints = [
        n
        for n in ast.walk(ast.parse(code))
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print"
    ]
    assert prints, "expected at least one print"
    for call in prints:
        for arg in call.args:
            ok = isinstance(arg, ast.Constant) or (
                isinstance(arg, ast.Name) and arg.id in allowed_names
            )
            assert ok, ast.unparse(call)


def test_template_keeps_repo_and_venv_out_of_kaggle_output() -> None:
    code = TEMPLATE.read_text(encoding="utf-8")
    assert 'REPO = \\"/tmp/repo\\"' in code


def report_root(tmp_path: Path, **report) -> Path:
    path = tmp_path / PARAMS["near_duplicates_cnn"]["out_summary"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return tmp_path


def test_push_needs_a_clean_near_copy_report_for_this_split(tmp_path: Path) -> None:
    split = PARAMS["data"]["split_file"]
    kj.check_split_is_clean(
        PARAMS, report_root(tmp_path / "ok", split_file=split, confirmed_cross_split=0)
    )
    with pytest.raises(kj.LaunchError, match="no near-copy report"):
        kj.check_split_is_clean(PARAMS, tmp_path / "missing")
    with pytest.raises(kj.LaunchError, match="cross splits"):
        kj.check_split_is_clean(
            PARAMS, report_root(tmp_path / "dirty", split_file=split, confirmed_cross_split=3)
        )
    with pytest.raises(kj.LaunchError, match="not data.split_file"):
        kj.check_split_is_clean(
            PARAMS,
            report_root(
                tmp_path / "old", split_file="data/splits/old.csv", confirmed_cross_split=0
            ),
        )


def test_committed_split_passes_the_push_guard() -> None:
    kj.check_split_is_clean(PARAMS)
