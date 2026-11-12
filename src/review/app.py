"""Review queue UI (TASK-024, ADR-031). Streamlit, local only.

    make review-ui        # http://127.0.0.1:8501, REVIEWER_PASSWORD and DATABASE_URL from .env

One card at a time, oldest first: the image (looked up by hash in the local archive, never stored),
why it was routed, and label buttons (the classes plus "unknown"). Low-confidence and OOD cards
show the model's prediction, calibrated confidence and energy; on a random-audit card these stay
hidden until the label is submitted (blind audit, ADR-031). The stream and any automated reviewer
are simulated in this project.
"""

from __future__ import annotations

import hmac
import json
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # `streamlit run` executes this file as a script
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_params  # noqa: E402
from src.db.store import PredictionLog  # noqa: E402
from src.review.archive import ImageArchive  # noqa: E402
from src.review.queue import REASONS, ReviewError, class_names, labels, reasons_of, submit  # noqa: E402

PASSWORD_ENV = "REVIEWER_PASSWORD"
DATABASE_ENV = "DATABASE_URL"
TEST_SPLIT = "test_frozen"
BANNER = (
    "Simulated environment: images come from the NEU production stream and any automated "
    "reviewer is simulated (ADR-031). Labels entered here are stored as human labels."
)
REASON_TEXT = {
    "low_confidence": "low confidence",
    "ood": "OOD (unfamiliar input)",
    "audit": "random audit",
}


def config_path() -> Path:
    args = sys.argv[1:]
    if "--config" in args and args.index("--config") + 1 < len(args):
        return Path(args[args.index("--config") + 1])
    return REPO_ROOT / "configs" / "params.yaml"


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


@st.cache_resource
def resources(config: str, db_url: str) -> dict[str, Any]:
    params = load_params(config)
    db = PredictionLog(db_url)
    db.require_head()  # never migrates (ADR-038): run `make db-migrate` first
    archive = ImageArchive.from_split_file(
        resolve(params["data"]["split_file"]), resolve(params["data"]["raw_dir"])
    )
    params["data"]["split_file"] = str(resolve(params["data"]["split_file"]))
    manifest_path = resolve(params["serving"]["bundle_dir"]) / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    return {"db": db, "archive": archive, "classes": class_names(params), "manifest": manifest}


def check_password() -> None:
    expected = os.environ.get(PASSWORD_ENV, "")
    if not expected.strip():
        st.error(f"{PASSWORD_ENV} is not set: the review UI refuses to start.")
        st.stop()
    if st.session_state.get("authenticated"):
        return
    given = st.text_input("Reviewer password", type="password", key="password")
    if given:
        if hmac.compare_digest(given.encode(), expected.encode()):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


def model_output(item: dict[str, Any], manifest: dict[str, Any]) -> str:
    ood = manifest.get("ood_energy_threshold")
    threshold = f" (OOD above {ood:.3f})" if isinstance(ood, float) else ""
    return (
        f"**Model prediction:** {item['prediction']}  \n"
        f"**Calibrated confidence:** {item['calibrated_confidence']:.4f}  \n"
        f"**Energy (OOD score):** {item['energy']:.3f}{threshold}  \n"
        f"Model version {item['model_version']} ({item['model_precision']})"
    )


def show_revealed(res: dict[str, Any]) -> None:
    shown = st.session_state["revealed"]
    st.subheader("Blind audit submitted")
    st.markdown(f"**Your label:** {shown['label']}")
    st.markdown(model_output(shown["item"], res["manifest"]))
    agree = "agrees" if shown["label"] == shown["item"]["prediction"] else "disagrees"
    st.markdown(f"The model {agree} with your label.")
    if st.button("Next card", key="next"):
        del st.session_state["revealed"]
        st.rerun()
    st.stop()


def show_card(res: dict[str, Any], item: dict[str, Any], reviewer: str) -> None:
    reasons = reasons_of(item)
    blind = "audit" in reasons
    left, right = st.columns([1, 1])
    with left:
        entry = res["archive"].lookup(item["image_sha256"])
        if entry is not None:
            st.image(str(entry.path), width=320)
            if entry.split == TEST_SPLIT:
                st.warning(
                    "This image is in test_frozen: its label will never be used for training."
                )
        else:
            st.info("Image not in the local archive (only its hash is logged).")
        st.caption(
            f"brightness {item['brightness']:.1f} · contrast {item['contrast']:.1f} · "
            f"sharpness {item['sharpness']:.1f} · {item['image_width']}x{item['image_height']} "
            f"{item['image_format']}"
        )
    with right:
        st.markdown("**Routed because:** " + ", ".join(REASON_TEXT[r] for r in reasons))
        if blind:
            st.warning("Blind audit: the model's answer is hidden until you submit your label.")
        else:
            st.markdown(model_output(item, res["manifest"]))
        st.caption(f"request {item['request_id']} · image sha256 {item['image_sha256'][:12]}…")
    st.markdown("**Your label:**")
    options = labels(res["classes"])
    for col, label in zip(st.columns(len(options)), options, strict=True):
        if col.button(label, key=f"label-{label}"):
            try:
                submit(
                    res["db"],
                    item["request_id"],
                    label,
                    res["classes"],
                    reviewer_kind="human",
                    reviewer_id=reviewer,
                )
            except ReviewError as exc:
                st.error(str(exc))
                st.stop()
            if blind:
                st.session_state["revealed"] = {"label": label, "item": item}
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Defect review queue", layout="wide")
    st.title("Defect review queue")
    st.info(BANNER)
    check_password()
    db_url = os.environ.get(DATABASE_ENV, "").strip()
    if not db_url:
        st.error(f"{DATABASE_ENV} is not set.")
        st.stop()
    res = resources(str(config_path()), db_url)
    reviewer = st.sidebar.text_input("Reviewer name", value="reviewer-1", key="reviewer")
    choice = st.sidebar.selectbox("Routing reason", ["all", *REASONS], key="reason")
    reason = None if choice == "all" else choice
    st.sidebar.metric("Waiting for review", res["db"].review_queue_size(reason))
    if "revealed" in st.session_state:
        show_revealed(res)
    queue = res["db"].review_queue(limit=1, reason=reason)
    if not queue:
        st.success("The review queue is empty.")
        st.stop()
    show_card(res, queue[0], reviewer)


main()
