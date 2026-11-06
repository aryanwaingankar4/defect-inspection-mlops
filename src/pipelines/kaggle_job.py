"""Run training on Kaggle GPU from a pinned commit (TASK-010, ADR-019). Reused by TASK-028.

    python -m src.pipelines.kaggle_job --config configs/params.yaml push [--no-run]
    python -m src.pipelines.kaggle_job --config configs/params.yaml status
    python -m src.pipelines.kaggle_job --config configs/params.yaml output

``push`` refuses uncommitted changes and commits that are not on GitHub, renders the launcher
notebook with the HEAD commit into ``kaggle.kernel_dir`` with a PRIVATE kernel-metadata.json
(GPU T4, internet on, private dataset attached), and pushes it. The notebook itself has no
logic: it clones the repo at that commit, installs the lockfile and runs ``src.training.train``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from src.config import load_params

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_FILE = "kaggle_train_launcher.ipynb"
METADATA_FILE = "kernel-metadata.json"
PLACEHOLDERS = ("COMMIT", "REPO_URL", "UV_VERSION", "NUM_WORKERS")

Git = Callable[..., str]


class LaunchError(RuntimeError):
    """The job cannot be launched safely (e.g. unpushed or uncommitted code)."""


def git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def pinned_commit(run_git: Git = git) -> str:
    """HEAD, only if the tree is clean and the commit is on a remote branch."""
    if run_git("status", "--porcelain", "--untracked-files=no"):
        raise LaunchError("uncommitted changes: commit and push first (the run must be pinned)")
    sha = run_git("rev-parse", "HEAD")
    if not run_git("branch", "-r", "--contains", sha):
        raise LaunchError(f"commit {sha[:7]} is not pushed: Kaggle clones it from GitHub")
    return sha


def render_notebook(template: str, values: Mapping[str, str]) -> str:
    """Fill ``{{KEY}}`` placeholders; every placeholder must be filled."""
    missing = [k for k in PLACEHOLDERS if k not in values]
    if missing:
        raise ValueError(f"missing values for {missing}")
    text = template
    for key in PLACEHOLDERS:
        text = text.replace("{{" + key + "}}", str(values[key]))
    if "{{" in text:
        raise ValueError("unfilled placeholder left in the notebook")
    json.loads(text)  # still a valid notebook
    return text


def kernel_metadata(owner: str, k: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": f"{owner}/{k['kernel_slug']}",
        "title": k["kernel_title"],
        "code_file": NOTEBOOK_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": k["accelerator"],
        "dataset_sources": [f"{owner}/{k['dataset_slug']}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def build_kernel_dir(params: Mapping[str, Any], owner: str, commit: str, out_dir: Path) -> Path:
    k = params["kaggle"]
    template = (REPO_ROOT / k["notebook_template"]).read_text(encoding="utf-8")
    values = {
        "COMMIT": commit,
        "REPO_URL": k["repo_url"],
        "UV_VERSION": k["uv_version"],
        "NUM_WORKERS": str(k["num_workers"]),
    }
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / NOTEBOOK_FILE).write_text(render_notebook(template, values), encoding="utf-8")
    meta = json.dumps(kernel_metadata(owner, k), indent=2) + "\n"
    (out_dir / METADATA_FILE).write_text(meta, encoding="utf-8")
    return out_dir


def _api() -> Any:
    from kaggle.api.kaggle_api_extended import KaggleApi  # lazy: import needs credentials

    api = KaggleApi()
    api.authenticate()
    return api


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    sub = parser.add_subparsers(dest="cmd", required=True)
    push = sub.add_parser("push", help="render the launcher and push it to Kaggle")
    push.add_argument("--no-run", action="store_true", help="save a version without running it")
    sub.add_parser("status", help="show the latest run status")
    sub.add_parser("output", help="download the latest run's output files")
    args = parser.parse_args(argv)

    params = load_params(args.config)
    k = params["kaggle"]
    api = _api()
    owner = api.get_config_value("username")
    ref = f"{owner}/{k['kernel_slug']}"

    if args.cmd == "push":
        commit = pinned_commit()
        folder = build_kernel_dir(params, owner, commit, Path(k["kernel_dir"]))
        try:
            print("GPU quota:", api.quota_view())
        except Exception as exc:  # quota is informational only
            print(f"GPU quota: unavailable ({type(exc).__name__})")
        resp = api.kernels_push(str(folder), acc=k["accelerator"], no_run=args.no_run)
        error = getattr(resp, "error", None)
        if error:
            print(f"push failed: {error}", file=sys.stderr)
            return 1
        mode = "saved without running" if args.no_run else "running"
        print(f"pushed {ref} at commit {commit[:7]} ({mode}): {getattr(resp, 'url', '')}")
    elif args.cmd == "status":
        print(api.kernels_status(ref))
    else:
        out = Path(k["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        files, _ = api.kernels_output(ref, str(out), force=True)
        print(f"downloaded {len(files)} file(s) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
