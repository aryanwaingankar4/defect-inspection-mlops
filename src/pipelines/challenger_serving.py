"""Make the ``@challenger`` version servable (ADR-035), e.g. one registered before this step
existed. ``make retrain`` runs the same step for new challengers.

    make challenger-serving

Uses the kept checkpoint ``models/versions/v<v>/best.pt`` and the results folder
``results/retrain/<train_run_uid>/`` (calibration and threshold already there): FP32 ONNX with the
ADR-025 parity check on val, ``serving_precision=fp32``, energy OOD threshold on val (ADR-027).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import load_params
from src.pipelines.retrain import RetrainError, prepare_serving
from src.tracking.gate import version_checkpoint
from src.training import export_onnx as ex


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # MLflow / the ONNX exporter print emoji
        stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/params.yaml"))
    args = parser.parse_args(argv)
    params = load_params(args.config)
    alias = str(params["promotion"]["challenger_alias"])
    try:
        version = ex.registered_version(params, alias)
        uid = version["tags"].get("train_run_uid")
        if not uid:
            raise RetrainError(f"@{alias} v{version['version']} has no train_run_uid tag")
        results_dir = Path(params["retrain"]["results_dir"]) / uid
        done = prepare_serving(
            params, version, version_checkpoint(params, version["version"]), results_dir
        )
    except (RetrainError, ex.ExportError) as exc:
        print(f"challenger-serving: {exc}", file=sys.stderr)
        return 1
    print(f"@{alias} v{version['version']} is servable (ADR-035): {json.dumps(done)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
