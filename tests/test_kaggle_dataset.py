"""Private Kaggle dataset upload (TASK-010). Fake Kaggle API and HTTP; no network."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data import kaggle_dataset as kd
from src.data.download import MANIFEST_NAME

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = REPO_ROOT / "src" / "data" / "kaggle_dataset.py"
REF = "me/neu-det-private"


class FakeApi:
    def __init__(self, existing=None, is_private=True):
        self.existing = existing
        self.is_private = is_private
        self.calls = []

    def dataset_list(self, mine=False, search=None):
        if self.existing is None:
            return []
        return [SimpleNamespace(ref=self.existing, is_private=self.is_private)]

    def dataset_create_new(self, folder, public=False, quiet=False, dir_mode="skip"):
        self.calls.append(("create", folder, public, dir_mode))

    def dataset_create_version(self, folder, version_notes, quiet=False, dir_mode="skip"):
        self.calls.append(("version", folder, dir_mode))


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    img = raw / "NEU-DET" / "train" / "images" / "crazing" / "crazing_1.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"jpg")
    (raw / MANIFEST_NAME).write_text("{}", encoding="utf-8")
    return raw


def test_stage_copies_layout_and_leaves_raw_untouched(raw: Path, tmp_path: Path) -> None:
    before = sorted(p.relative_to(raw).as_posix() for p in raw.rglob("*"))
    meta = kd.dataset_metadata("me", "neu-det-private", "NEU-DET private copy")
    staged = kd.stage(raw, tmp_path / "build" / "stage", meta)
    assert (staged / "NEU-DET/train/images/crazing/crazing_1.jpg").read_bytes() == b"jpg"
    assert (staged / MANIFEST_NAME).exists()
    assert json.loads((staged / kd.METADATA_FILE).read_text(encoding="utf-8"))["id"] == REF
    assert sorted(p.relative_to(raw).as_posix() for p in raw.rglob("*")) == before


def test_stage_refuses_a_folder_inside_raw(raw: Path) -> None:
    with pytest.raises(ValueError, match="outside raw_dir"):
        kd.stage(raw, raw / "staging", {})


def test_upload_is_always_private(tmp_path: Path) -> None:
    api = FakeApi(existing=None)
    assert kd.upload(api, tmp_path, REF) == "created"
    assert api.calls == [("create", str(tmp_path), False, "zip")]
    api = FakeApi(existing=REF)
    assert kd.upload(api, tmp_path, REF) == "new version"
    assert api.calls[0][0] == "version"


def test_module_never_asks_for_a_public_dataset() -> None:
    """No code path may pass public=True (or any non-False value)."""
    for node in ast.walk(ast.parse(MODULE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.keyword) and node.arg == "public":
            assert isinstance(node.value, ast.Constant) and node.value.value is False


def test_verify_private_fails_loudly() -> None:
    kd.verify_private(FakeApi(existing=REF, is_private=True), REF)
    with pytest.raises(kd.DatasetPrivacyError, match="NOT private"):
        kd.verify_private(FakeApi(existing=REF, is_private=False), REF)
    with pytest.raises(kd.DatasetPrivacyError, match="not found"):
        kd.verify_private(FakeApi(existing=None), REF)


@pytest.mark.parametrize(
    ("control", "target", "expected"),
    [
        (200, 404, "blocked"),
        (200, 403, "blocked"),
        (403, 404, "inconclusive"),
        (200, 500, "inconclusive"),
    ],
)
def test_anonymous_check(control: int, target: int, expected: str) -> None:
    def fetch(url: str) -> int:
        return control if kd.PUBLIC_CONTROL in url else target

    assert kd.anonymous_check(REF, fetch) == expected


def test_anonymous_check_raises_if_public() -> None:
    with pytest.raises(kd.DatasetPrivacyError, match="visible"):
        kd.anonymous_check(REF, lambda url: 200)


def test_verify_files() -> None:
    expected = ["NEU-DET/a/x.jpg", "NEU-DET/b/y.jpg"]
    kd.verify_files(["NEU-DET/a/x.jpg", "NEU-DET/b/y.jpg", "NEU-DET/a/x.xml"], expected)
    with pytest.raises(RuntimeError, match="1 missing"):
        kd.verify_files(["NEU-DET/a/x.jpg"], expected)
