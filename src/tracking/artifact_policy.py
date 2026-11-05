"""Guard for MLflow artifacts: never log images that contain NEU pixels (ADR-015, ADR-018).

The DagsHub MLflow server is PUBLIC, so anything logged there is redistributed. Metric plots
(confusion matrix, reliability diagram, ...) are fine; Grad-CAM overlays, sample predictions,
duplicate grids or any other picture made from dataset images are not.

Rule: an image-like artifact is allowed only as ``plots/<name>.png`` where ``<name>`` is in
``ALLOWED_PLOT_NAMES``. Adding a plot name is a deliberate code change that gets reviewed.
Every MLflow artifact call in ``src/`` must go through ``safe_log_artifact`` (enforced by a
test that scans the source code).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

# Formats that can hold raster pixels (SVG/PDF can embed bitmaps).
IMAGE_LIKE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".svg", ".pdf"}
)
PLOTS_DIR = "plots"
ALLOWED_PLOT_NAMES = frozenset(
    {
        "confusion_matrix",
        "confusion_matrix_normalized",
        "reliability_diagram",
        "reliability_diagram_before",
        "reliability_diagram_after",
        "risk_coverage",
        "per_class_f1",
        "training_curves",
        "ood_score_histogram",
        "confidence_histogram",
    }
)


class ArtifactPolicyError(ValueError):
    """Raised when an artifact could contain dataset pixels."""


def is_allowed_artifact(artifact_path: str) -> bool:
    """True if logging ``artifact_path`` (path inside the run) cannot leak dataset pixels."""
    path = PurePosixPath(artifact_path.replace("\\", "/"))
    if path.suffix.lower() not in IMAGE_LIKE_SUFFIXES:
        return True
    return (
        path.suffix.lower() == ".png"
        and len(path.parts) == 2
        and path.parts[0] == PLOTS_DIR
        and path.stem in ALLOWED_PLOT_NAMES
    )


def find_violations(artifact_paths: Iterable[str]) -> list[str]:
    """Return the artifact paths that break the policy (e.g. from a run's artifact listing)."""
    return sorted(p for p in artifact_paths if not is_allowed_artifact(p))


def safe_log_artifact(
    local_path: Path | str,
    artifact_dir: str | None,
    log_fn: Callable[[str, str | None], object],
) -> None:
    """Check the policy, then call ``log_fn(local_path, artifact_dir)``.

    ``log_fn`` is ``mlflow.log_artifact`` in real use (wired in TASK-009); injected here so the
    guard is testable without MLflow installed.
    """
    name = Path(local_path).name
    target = f"{artifact_dir}/{name}" if artifact_dir else name
    if not is_allowed_artifact(target):
        raise ArtifactPolicyError(
            f"Refusing to log {target!r}: image-like artifacts are allowed only as "
            f"{PLOTS_DIR}/<name>.png with <name> in ALLOWED_PLOT_NAMES (ADR-018)."
        )
    log_fn(str(local_path), artifact_dir)
