"""Static checks on the DVC setup (no data or DVC run needed)."""

import configparser
from pathlib import Path

import pytest
import yaml

from src.config import load_params

REPO_ROOT = Path(__file__).resolve().parents[1]
DVC_YAML = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")


def out_options(stage: str, kind: str = "outs") -> dict[str, dict]:
    """Map each output path of a stage to its options ({} if none)."""
    result = {}
    for entry in DVC_YAML["stages"][stage].get(kind, []):
        if isinstance(entry, str):
            result[entry] = {}
        else:
            ((path, opts),) = entry.items()
            result[path] = opts or {}
    return result


@pytest.mark.parametrize(
    ("stage", "kind", "path"),
    [
        ("split", "outs", "data/splits/neu_split_v1.csv"),
        ("near_dups", "outs", "results/near_duplicates.csv"),
        ("near_dups", "metrics", "results/near_duplicates_summary.json"),
    ],
)
def test_git_tracked_outputs_are_not_cached(stage: str, kind: str, path: str) -> None:
    assert out_options(stage, kind)[path].get("cache") is False


def test_split_output_matches_params() -> None:
    assert PARAMS["data"]["split_file"] in out_options("split")


def test_private_pixels_are_never_git_outputs() -> None:
    for stage in DVC_YAML["stages"]:
        for kind in ("outs", "metrics", "plots"):
            for path, opts in out_options(stage, kind).items():
                if path.startswith("results/private/"):
                    assert opts.get("cache", True) is True, f"{path} must stay out of git"


def _lookup(params: dict, dotted: str) -> object:
    node = params
    for key in dotted.split("."):
        node = node[key]
    return node


def test_stage_params_exist_in_params_yaml() -> None:
    for stage in DVC_YAML["stages"].values():
        for entry in stage.get("params", []):
            for keys in entry.values():
                for dotted in keys:
                    _lookup(PARAMS, dotted)  # raises KeyError if missing


def test_stage_deps_exist_or_are_dvc_tracked() -> None:
    for stage in DVC_YAML["stages"].values():
        for dep in stage.get("deps", []):
            path = REPO_ROOT / dep
            assert path.exists() or path.with_suffix(".dvc").exists(), dep


def test_default_remote_is_a_private_local_folder() -> None:
    config = configparser.ConfigParser()
    config.read(REPO_ROOT / ".dvc" / "config", encoding="utf-8")
    remote = config["core"]["remote"]
    # DVC writes the header as ['remote "name"']; configparser keeps the single quotes.
    (section,) = [s for s in config.sections() if s.strip("'") == f'remote "{remote}"']
    url = config[section]["url"]
    assert "://" not in url, f"DVC remote must be a local folder, not a cloud URL: {url}"
    resolved = (REPO_ROOT / ".dvc" / url).resolve()
    assert not resolved.is_relative_to(REPO_ROOT), "DVC remote must live outside the repo"
