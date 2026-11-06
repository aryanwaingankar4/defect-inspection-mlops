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
