"""OpenAPI snapshot (TASK-020): ``make openapi-snapshot`` writes the API contract that
``tests/test_contract.py`` compares with the live schema. Update it only for an intended API change.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.config import load_params


def openapi_schema(params: Mapping[str, Any]) -> dict[str, Any]:
    """The app's OpenAPI schema (no model, key or database is needed to build it)."""
    from src.serving.app import create_app

    return create_app(params, api_key="schema-only", database_url="sqlite://").openapi()


def render(schema: Mapping[str, Any]) -> bytes:
    return (json.dumps(schema, indent=2, sort_keys=True) + "\n").encode()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/params.yaml")
    parser.add_argument("--out", default="tests/contract/openapi.json")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(render(openapi_schema(load_params(args.config))))
    print(f"OpenAPI snapshot -> {out.as_posix()}")


if __name__ == "__main__":
    main()
