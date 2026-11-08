"""``make serving-bundle`` (TASK-018, ADR-028): build ``models/serving/`` from ``@champion``.

Read-only on the registry: resolve the alias, download the ONNX file named by the version's
``serving_precision`` tag from its run, check the SHA-256 against the tag, write the file and
``manifest.json`` (T, review and OOD thresholds, class order, preprocessing), then verify the
result with the same ``load_bundle`` the API runs at startup. The old bundle is replaced only
after the new one passes. Never prints credentials.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import load_params
from src.serving.bundle import (
    BUNDLE_FORMAT,
    MANIFEST_NAME,
    PRECISIONS,
    BundleError,
    load_bundle,
    sha256_file,
)
from src.serving.onnx_model import OnnxClassifier
from src.tracking import mlflow_tracking as mt

DECISION_RECORD = "ADR-028"
VALUE_TAGS = ("temperature", "review_threshold", "ood_energy_threshold")
FILE_TAGS = {
    "fp32": ("onnx_sha256", "onnx_artifact"),
    "int8": ("onnx_int8_sha256", "onnx_int8_artifact"),
}


class ServingBundleError(RuntimeError):
    pass


def champion(params: Mapping[str, Any]) -> dict[str, Any]:
    """The ``@champion`` version with every tag serving needs."""
    cfg = mt.TrackingConfig.from_params(params, smoke=False)
    alias = str(params["promotion"]["champion_alias"])
    found = mt.version_by_alias(cfg, alias)
    if found is None:
        raise ServingBundleError(f"no @{alias} on {cfg.registered_model}")
    tags = found["tags"]
    precision = tags.get("serving_precision")
    if precision not in PRECISIONS:
        raise ServingBundleError(f"@{alias} has no valid serving_precision tag (ADR-026)")
    missing = [k for k in (*VALUE_TAGS, *FILE_TAGS[precision]) if not tags.get(k)]
    if missing:
        raise ServingBundleError(f"@{alias} version {found['version']} lacks tags {missing}")
    return {**found, "name": cfg.registered_model, "alias": alias, "precision": precision}


def download(cfg: mt.TrackingConfig, run_id: str, artifact: str, dst: Path) -> Path:
    try:
        return mt.download_run_artifact(cfg, run_id, artifact, dst)
    except Exception as exc:  # noqa: BLE001 - any failure means the file is not there
        raise ServingBundleError(f"cannot download {artifact} from run {run_id}") from exc


def manifest_for(champ: Mapping[str, Any], onnx_path: Path) -> dict[str, Any]:
    tags = champ["tags"]
    sha_tag, artifact_tag = FILE_TAGS[champ["precision"]]
    meta = OnnxClassifier(onnx_path).metadata
    return {
        "bundle_format": BUNDLE_FORMAT,
        "decision_record": DECISION_RECORD,
        "registered_model": champ["name"],
        "version": str(champ["version"]),
        "alias": champ["alias"],
        "run_id": champ["run_id"],
        "precision": champ["precision"],
        "onnx_file": onnx_path.name,
        "onnx_sha256": tags[sha_tag],
        "onnx_artifact": tags[artifact_tag],
        "checkpoint_sha256": tags.get("checkpoint_sha256"),
        "temperature": float(tags["temperature"]),
        "review_threshold": float(tags["review_threshold"]),
        "ood_energy_threshold": float(tags["ood_energy_threshold"]),
        "classes": json.loads(meta["classes"]),
        "preprocess": json.loads(meta["preprocess"]),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def build(params: Mapping[str, Any], out_dir: Path | None = None) -> dict[str, Any]:
    """Build and verify the bundle; returns the manifest."""
    cfg = mt.TrackingConfig.from_params(params, smoke=False)
    out_dir = Path(out_dir or params["serving"]["bundle_dir"])
    champ = champion(params)
    sha_tag, artifact_tag = FILE_TAGS[champ["precision"]]
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir.parent, prefix=".bundle-") as tmp:
        tmp_path = Path(tmp)
        fetched = download(cfg, champ["run_id"], champ["tags"][artifact_tag], tmp_path / "dl")
        if sha256_file(fetched) != champ["tags"][sha_tag]:
            raise ServingBundleError(f"downloaded {fetched.name} does not match {sha_tag}")
        staged = tmp_path / "bundle"
        staged.mkdir()
        onnx_path = staged / fetched.name
        shutil.move(fetched, onnx_path)
        manifest = manifest_for(champ, onnx_path)
        (staged / MANIFEST_NAME).write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
        try:
            load_bundle(staged, params["preprocess"])  # the same checks the API runs
        except BundleError as exc:
            raise ServingBundleError(f"the new bundle fails its own checks: {exc}") from exc
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.move(staged, out_dir)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/params.yaml")
    args = parser.parse_args(argv)
    params = load_params(args.config)
    m = build(params)
    out = params["serving"]["bundle_dir"]
    print(f"serving bundle -> {out}/ ({DECISION_RECORD})")
    print(f"  {m['registered_model']} v{m['version']} @{m['alias']}, {m['precision']}")
    print(f"  {m['onnx_file']}  sha256 {m['onnx_sha256'][:12]}... (matches the version tag)")
    print(
        f"  T = {m['temperature']:.4f}, review threshold = {m['review_threshold']:.4f}, "
        f"OOD energy threshold = {m['ood_energy_threshold']:.4f}"
    )
    print(f"  classes: {', '.join(m['classes'])}")


if __name__ == "__main__":
    main()
