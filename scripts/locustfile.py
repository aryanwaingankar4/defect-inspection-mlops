"""Locust user for TASK-043: POST /predict with the committed SYNTHETIC golden image only
(ADR-017; no NEU pixels go to the cloud). The API key comes from the environment."""

from __future__ import annotations

import os
from pathlib import Path

from locust import HttpUser, between, task

PNG = (
    Path(__file__).resolve().parents[1] / "tests" / "golden" / "synthetic_golden.png"
).read_bytes()


class PredictUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task
    def predict(self) -> None:
        self.client.post(
            "/predict",
            files={"file": ("golden.png", PNG, "image/png")},
            headers={"X-API-Key": os.environ["API_KEY"]},
            name="/predict",
        )
