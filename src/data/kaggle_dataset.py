"""Upload ``data/raw`` to Kaggle as a PRIVATE dataset (TASK-010, ADR-018).

    python -m src.data.kaggle_dataset --config configs/params.yaml

Steps: stage a copy (``NEU-DET/...`` layout + manifest) in ``kaggle.staging_dir``, create the
dataset (or a new version) with ``public=False``, wait until Kaggle has processed it, then
check three things and fail loudly otherwise:
  1. the Kaggle API reports the dataset as private;
  2. an anonymous request (no credentials) cannot see it;
  3. the uploaded images are exactly the images listed in the split CSV.

The NEU images have no licence to redistribute: this module has no way to upload publicly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from src.config import load_params
from src.data.download import MANIFEST_NAME
from src.data.split import read_split_csv

METADATA_FILE = "dataset-metadata.json"
DATA_ROOT = "NEU-DET"
PUBLIC_CONTROL = "kaustubhdikshit/neu-surface-defect-database"  # known public dataset (ADR-015)


class DatasetPrivacyError(RuntimeError):
    """The dataset is, or may be, visible to the public."""


def dataset_metadata(owner: str, slug: str, title: str) -> dict[str, Any]:
    """Kaggle ``dataset-metadata.json``. Privacy is not a field here: it is set at creation."""
    return {"title": title, "id": f"{owner}/{slug}", "licenses": [{"name": "unknown"}]}


def stage(raw_dir: Path, staging_dir: Path, metadata: dict[str, Any]) -> Path:
    """Copy ``NEU-DET/`` and the manifest into a fresh staging folder with the metadata file."""
    raw_dir, staging_dir = raw_dir.resolve(), staging_dir.resolve()
    if staging_dir == raw_dir or raw_dir in staging_dir.parents:
        raise ValueError("staging_dir must be outside raw_dir (it would change the DVC hash)")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    shutil.copytree(raw_dir / DATA_ROOT, staging_dir / DATA_ROOT)
    shutil.copy2(raw_dir / MANIFEST_NAME, staging_dir / MANIFEST_NAME)
    (staging_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n", "utf-8")
    return staging_dir


def find_own_dataset(api: Any, ref: str) -> Any | None:
    for ds in api.dataset_list(mine=True, search=ref.split("/")[1]) or []:
        if ds is not None and str(ds.ref).lower() == ref.lower():
            return ds
    return None


def upload(api: Any, staging_dir: Path, ref: str) -> str:
    """Create the dataset privately, or add a version if it already exists."""
    if find_own_dataset(api, ref) is None:
        api.dataset_create_new(str(staging_dir), public=False, quiet=True, dir_mode="zip")
        return "created"
    api.dataset_create_version(
        str(staging_dir), version_notes="re-upload data/raw", quiet=True, dir_mode="zip"
    )
    return "new version"


def wait_until_ready(api: Any, ref: str, timeout_s: int = 900, poll_s: int = 15) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        status = str(api.dataset_status(ref)).lower()
        if "ready" in status:
            return
        if "error" in status:
            raise RuntimeError(f"Kaggle failed to process {ref}: {status}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"{ref} not ready after {timeout_s}s (status: {status})")
        time.sleep(poll_s)


def verify_private(api: Any, ref: str) -> None:
    ds = find_own_dataset(api, ref)
    if ds is None:
        raise DatasetPrivacyError(f"{ref} not found among your datasets")
    if ds.is_private is not True:
        raise DatasetPrivacyError(f"{ref} is NOT private on Kaggle: make it private now")


Fetch = Callable[[str], int]


def http_status(url: str) -> int:
    """Status code of an anonymous GET (no cookies, no credentials)."""
    req = urllib.request.Request(url, headers={"User-Agent": "defect-inspection-mlops-check"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def anonymous_check(ref: str, fetch: Fetch = http_status) -> str:
    """``"blocked"`` if anonymous users cannot open the dataset page.

    A known public dataset is fetched first as a control: if the control is not visible
    either, the result is ``"inconclusive"`` (e.g. bot protection), not a pass.
    """
    page = "https://www.kaggle.com/datasets/{}"
    if fetch(page.format(PUBLIC_CONTROL)) != 200:
        return "inconclusive"
    status = fetch(page.format(ref))
    if status == 200:
        raise DatasetPrivacyError(f"{ref} is visible to anonymous users (HTTP 200)")
    return "blocked" if status in (403, 404) else "inconclusive"


def list_files(api: Any, ref: str, page_size: int = 200) -> list[str]:
    names, token = [], None
    while True:
        result = api.dataset_list_files(ref, page_token=token, page_size=page_size)
        names += [f.name for f in (result.files or [])]
        token = getattr(result, "next_page_token", None) or getattr(result, "nextPageToken", None)
        if not token:
            return names


def verify_files(uploaded: Iterable[str], expected: Iterable[str]) -> None:
    """Uploaded ``.jpg`` paths must equal the split CSV paths (``NEU-DET/...``)."""
    got = {PurePosixPath(p).as_posix() for p in uploaded if p.lower().endswith(".jpg")}
    want = set(expected)
    if got != want:
        missing, extra = sorted(want - got)[:5], sorted(got - want)[:5]
        raise RuntimeError(
            f"uploaded images differ from the split CSV: {len(want - got)} missing "
            f"(e.g. {missing}), {len(got - want)} unexpected (e.g. {extra})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    params = load_params(args.config)
    k = params["kaggle"]

    from kaggle.api.kaggle_api_extended import KaggleApi  # lazy: import needs credentials

    api = KaggleApi()
    api.authenticate()
    owner = api.get_config_value("username")
    ref = f"{owner}/{k['dataset_slug']}"

    staging = stage(
        Path(params["data"]["raw_dir"]),
        Path(k["staging_dir"]),
        dataset_metadata(owner, k["dataset_slug"], k["dataset_title"]),
    )
    print(f"staged {staging}; uploading {ref} as PRIVATE ...", flush=True)
    print(f"{ref}: {upload(api, staging, ref)}; waiting for Kaggle to process it ...", flush=True)
    wait_until_ready(api, ref)

    verify_private(api, ref)
    print(f"[ok] Kaggle API: {ref} is private")
    result = anonymous_check(ref)
    print(f"[{'ok' if result == 'blocked' else '??'}] anonymous access: {result}")
    expected = [path for path, _, _ in read_split_csv(Path(params["data"]["split_file"]))]
    verify_files(list_files(api, ref), expected)
    print(f"[ok] {len(expected)} images uploaded at their NEU-DET/... paths")
    return 0 if result == "blocked" else 1


if __name__ == "__main__":
    sys.exit(main())
