import re
from pathlib import Path

import pytest

from src.tracking.artifact_policy import (
    ArtifactPolicyError,
    find_violations,
    is_allowed_artifact,
    safe_log_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("artifact_path", "allowed"),
    [
        # metric plots: fine
        ("plots/confusion_matrix.png", True),
        ("plots/reliability_diagram_after.png", True),
        ("plots/risk_coverage.png", True),
        # non-image artifacts: fine
        ("model/model.onnx", True),
        ("metrics.json", True),
        ("params.yaml", True),
        # anything that could hold NEU pixels: rejected
        ("gradcam/crazing_12.png", False),
        ("plots/gradcam_crazing_12.png", False),
        ("plots/sample_predictions.png", False),
        ("sample_predictions/grid.png", False),
        ("near_duplicates_grid.png", False),
        ("plots/confusion_matrix.jpg", False),
        ("plots/confusion_matrix.svg", False),
        ("report.pdf", False),
        ("inputs/crazing_1.bmp", False),
        ("plots/nested/confusion_matrix.png", False),
        ("plots\\gradcam.PNG", False),
    ],
)
def test_artifact_rule(artifact_path: str, allowed: bool) -> None:
    assert is_allowed_artifact(artifact_path) is allowed


def test_find_violations_on_a_run_listing() -> None:
    listing = ["metrics.json", "plots/confusion_matrix.png", "gradcam/a.png", "samples/b.jpg"]
    assert find_violations(listing) == ["gradcam/a.png", "samples/b.jpg"]


def test_safe_log_artifact_blocks_before_logging(tmp_path: Path) -> None:
    calls: list[tuple[str, str | None]] = []
    gradcam = tmp_path / "gradcam_crazing_1.png"
    gradcam.write_bytes(b"x")
    with pytest.raises(ArtifactPolicyError):
        safe_log_artifact(gradcam, "plots", lambda p, d: calls.append((p, d)))
    assert calls == []

    plot = tmp_path / "confusion_matrix.png"
    plot.write_bytes(b"x")
    safe_log_artifact(plot, "plots", lambda p, d: calls.append((p, d)))
    assert calls == [(str(plot), "plots")]


# Direct MLflow artifact/image calls bypass the guard; only src/tracking/ may make them.
DIRECT_CALL = re.compile(r"\b(log_artifacts?|log_image|log_figure|log_table)\s*\(")


def test_no_direct_mlflow_artifact_calls_outside_tracking() -> None:
    offenders = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        if "tracking" in path.relative_to(REPO_ROOT / "src").parts:
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT)
        offenders += [f"{rel}: {m.group(0)}" for m in DIRECT_CALL.finditer(text)]
    assert offenders == [], f"Use src.tracking.artifact_policy.safe_log_artifact: {offenders}"
