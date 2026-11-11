"""Simulated production stream (TASK-025, ADR-031): send ``production_stream`` images to the API.

THE STREAM IS SIMULATED: images come from the NEU ``production_stream`` split (never another
split) in a seeded order. ``--start`` continues where an earlier run stopped. The API key comes
from env ``API_KEY`` and is never printed. Standard library only.

    make stream N=40                     # first 40 images of the seeded order
    make stream N=40 ARGS="--start 40"   # the next 40
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import urllib.error
import urllib.request
import uuid
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.config import load_params

STREAM_SPLIT = "production_stream"


class StreamError(RuntimeError):
    pass


def stream_order(params: Mapping[str, Any]) -> list[dict[str, str]]:
    """All ``production_stream`` rows in a seeded order (the same every run)."""
    cfg = params["review"]["stream"]
    if cfg["split"] != STREAM_SPLIT:
        raise StreamError(f"the stream only sends {STREAM_SPLIT}, not {cfg['split']!r} (ADR-031)")
    with Path(params["data"]["split_file"]).open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == STREAM_SPLIT]
    rows.sort(key=lambda r: r["path"])
    random.Random(int(cfg["seed"])).shuffle(rows)
    return rows


def post_image(url: str, key: str, data: bytes, name: str) -> tuple[int, dict]:
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + data
        + f"\r\n--{boundary}--\r\n".encode()
    )
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}", "X-API-Key": key}
    req = urllib.request.Request(f"{url}/predict", data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def send(params: Mapping[str, Any], url: str, key: str, n: int, start: int) -> dict[str, Any]:
    rows = stream_order(params)[start : start + n]
    raw = Path(params["data"]["raw_dir"])
    statuses: Counter[int] = Counter()
    reasons: Counter[str] = Counter()
    for row in rows:
        status, body = post_image(url, key, (raw / row["path"]).read_bytes(), "stream.jpg")
        statuses[status] += 1
        if status == 200:
            for reason in body["reasons"] or ["accepted"]:
                reasons[reason] += 1
    return {
        "stream": f"simulated, {STREAM_SPLIT} (ADR-031)",
        "start": start,
        "sent": len(rows),
        "statuses": dict(sorted(statuses.items())),
        "accepted_or_reasons": dict(sorted(reasons.items())),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Simulated production stream (ADR-031)")
    parser.add_argument("--config", default="configs/params.yaml")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("-n", type=int, default=40)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args(argv)
    key = os.environ.get("API_KEY", "")
    if not key:
        sys.exit("API_KEY is not set (make stream loads it from .env)")
    summary = send(load_params(args.config), args.url, key, args.n, args.start)
    print("SIMULATED stream:", json.dumps(summary))


if __name__ == "__main__":
    main()
