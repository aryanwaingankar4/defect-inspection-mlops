"""TASK-028 (ADR-034): the retrain flow and its pieces. Fake Kaggle API, generated files; no
network, no GPU, no NEU pixels."""

import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import load_params
from src.data import kaggle_dataset as kd
from src.pipelines import kaggle_job as kj
from src.pipelines import retrain as rt
from src.training import feedback as fbk

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
MD5 = "0123456789abcdef0123456789abcdef"


def write_split(path: Path) -> Path:
    rows = [
        ("a/crazing_1.jpg", "crazing", "production_stream"),
        ("a/inclusion_1.jpg", "inclusion", "production_stream"),
        ("a/patches_1.jpg", "patches", "val"),
        ("a/scratches_1.jpg", "scratches", "test_frozen"),
        ("a/pitted_1.jpg", "pitted_surface", "train_v1"),
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["path", "class", "split"])
        w.writerows(rows)
    return path


def write_feedback(path: Path, rows) -> str:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["path", "label", "reviewer_kind", "request_id"])
        w.writerows([(p, lbl, "simulated", f"{i:032x}") for i, (p, lbl) in enumerate(rows)])
    return hashlib.md5(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- feedback for training


def test_load_feedback_accepts_production_stream_rows(tmp_path):
    split = write_split(tmp_path / "split.csv")
    csv_path = tmp_path / "feedback.csv"
    md5 = write_feedback(
        csv_path, [("a/crazing_1.jpg", "crazing"), ("a/inclusion_1.jpg", "patches")]
    )
    fb = fbk.load_feedback(csv_path, md5, split, CLASSES)
    assert fb.rows == [("a/crazing_1.jpg", "crazing"), ("a/inclusion_1.jpg", "patches")]
    assert fb.summary() == {
        "md5": md5,
        "rows": 2,
        "per_class": {"crazing": 1, "patches": 1},
        "split": "production_stream",
    }


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([("a/patches_1.jpg", "patches")], "val"),
        ([("a/scratches_1.jpg", "scratches")], "test_frozen"),
        ([("a/pitted_1.jpg", "pitted_surface")], "train_v1"),
        ([("a/unknown.jpg", "crazing")], "None"),
        ([("a/crazing_1.jpg", "unknown")], "label"),
        ([("a/crazing_1.jpg", "crazing"), ("a/crazing_1.jpg", "crazing")], "twice"),
    ],
)
def test_load_feedback_refuses_anything_else(tmp_path, rows, message):
    split = write_split(tmp_path / "split.csv")
    csv_path = tmp_path / "feedback.csv"
    md5 = write_feedback(csv_path, rows)
    with pytest.raises(fbk.FeedbackError, match=message):
        fbk.load_feedback(csv_path, md5, split, CLASSES)


def test_load_feedback_refuses_another_export(tmp_path):
    split = write_split(tmp_path / "split.csv")
    csv_path = tmp_path / "feedback.csv"
    write_feedback(csv_path, [("a/crazing_1.jpg", "crazing")])
    with pytest.raises(fbk.FeedbackError, match="MD5"):
        fbk.load_feedback(csv_path, MD5, split, CLASSES)


def test_split_dataset_adds_feedback_rows(tmp_path):
    pytest.importorskip("torch")
    from src.serving.preprocess import PreprocessConfig
    from src.training.dataset import SplitDataset

    split = write_split(tmp_path / "split.csv")
    ds = SplitDataset(
        split,
        tmp_path,
        ["train_v1"],
        CLASSES,
        PreprocessConfig.from_params(PARAMS),
        images_per_class=1,
        extra_rows=[("a/crazing_1.jpg", "crazing"), ("a/inclusion_1.jpg", "inclusion")],
    )
    assert ds.paths == ["a/pitted_1.jpg", "a/crazing_1.jpg", "a/inclusion_1.jpg"]
    assert ds.labels == [3, 0, 1]


def test_train_cli_needs_both_feedback_arguments(tmp_path):
    pytest.importorskip("torch")
    from src.training import train

    args = train.parse_args(["--config", "c.yaml", "--feedback", "f.csv"])
    with pytest.raises(SystemExit, match="go together"):
        train.feedback_from_args(PARAMS, args)
    assert train.feedback_from_args(PARAMS, train.parse_args(["--config", "c.yaml"])) is None


def test_training_never_loads_the_previous_model():
    """Retraining starts from ImageNet weights (CLAUDE.md): train.py has no resume path."""
    text = (REPO_ROOT / "src" / "training" / "train.py").read_text(encoding="utf-8")
    assert "load_state_dict" not in text and "--resume" not in text
    assert PARAMS["train"]["pretrained"] is True


# ---------------------------------------------------------------- Kaggle job + dataset


def test_train_args_carry_only_a_valid_md5():
    assert kj.train_args(None, MD5) == ["--feedback-md5", MD5]
    with pytest.raises(kj.LaunchError):
        kj.train_args(None, "abc; rm -rf /")


def test_kernel_attaches_the_feedback_dataset_and_renders_the_md5(tmp_path):
    folder = kj.build_kernel_dir(
        PARAMS, "me", "a" * 40, tmp_path / "k", kj.train_args(None, MD5), ["me/neu-feedback"]
    )
    meta = json.loads((folder / kj.METADATA_FILE).read_text(encoding="utf-8"))
    assert meta["dataset_sources"] == [f"me/{PARAMS['kaggle']['dataset_slug']}", "me/neu-feedback"]
    assert meta["is_private"] is True
    nb = (folder / kj.NOTEBOOK_FILE).read_text(encoding="utf-8")
    assert MD5 in nb and "--feedback-md5" in nb and "*feedback_args" in nb


def test_status_reads_the_enum_name():
    api = SimpleNamespace(
        get_config_value=lambda key: "me",
        kernels_status=lambda ref: SimpleNamespace(status=SimpleNamespace(name="COMPLETE")),
    )
    assert kj.status(api, PARAMS) == "complete"


class FakeDatasetApi:
    def __init__(self, files=("feedback.csv",), private=True):
        self.files, self.private, self.calls = list(files), private, []

    def get_config_value(self, key):
        return "me"

    def dataset_list(self, mine=False, search=None):
        ref = f"me/{PARAMS['kaggle']['feedback_dataset_slug']}"
        return [SimpleNamespace(ref=ref, is_private=self.private)]

    def dataset_create_version(self, folder, version_notes, quiet=False, dir_mode="skip"):
        self.calls.append(
            ("version", sorted(p.name for p in Path(folder).iterdir()), version_notes)
        )
        return SimpleNamespace(error="")

    def dataset_list_files(self, ref, page_token=None, page_size=200):
        return SimpleNamespace(files=[SimpleNamespace(name=n) for n in self.files])


def feedback_params(tmp_path):
    p = copy.deepcopy(PARAMS)
    p["kaggle"]["feedback_staging_dir"] = str(tmp_path / "stage")
    return p


def test_upload_feedback_stages_only_the_csv_and_checks_privacy(tmp_path):
    csv_path = tmp_path / "feedback.csv"
    write_feedback(csv_path, [("a/crazing_1.jpg", "crazing")])
    api = FakeDatasetApi()
    ok = {"https://www.kaggle.com/datasets/" + kd.PUBLIC_CONTROL: 200}
    ref = kd.upload_feedback(
        api,
        feedback_params(tmp_path),
        csv_path,
        MD5,
        fetch=lambda url: ok.get(url, 404),
        wait=lambda a, r: None,
    )
    assert ref == f"me/{PARAMS['kaggle']['feedback_dataset_slug']}"
    assert api.calls == [("version", ["dataset-metadata.json", "feedback.csv"], api.calls[0][2])]
    assert MD5 in api.calls[0][2]


def test_upload_feedback_fails_if_not_private_or_extra_files(tmp_path):
    csv_path = tmp_path / "feedback.csv"
    write_feedback(csv_path, [("a/crazing_1.jpg", "crazing")])
    blocked = {"https://www.kaggle.com/datasets/" + kd.PUBLIC_CONTROL: 200}

    def fetch(url):
        return blocked.get(url, 404)

    with pytest.raises(kd.DatasetPrivacyError):
        kd.upload_feedback(
            FakeDatasetApi(private=False), feedback_params(tmp_path), csv_path, MD5,
            fetch=fetch, wait=lambda a, r: None,
        )  # fmt: skip
    with pytest.raises(RuntimeError, match="only feedback.csv"):
        kd.upload_feedback(
            FakeDatasetApi(files=["feedback.csv", "x.jpg"]), feedback_params(tmp_path), csv_path,
            MD5, fetch=fetch, wait=lambda a, r: None,
        )  # fmt: skip


# ---------------------------------------------------------------- flow: pure parts


def export_params(tmp_path, rows=3, minimum=2, tamper=False):
    p = copy.deepcopy(PARAMS)
    out = tmp_path / "feedback"
    out.mkdir()
    md5 = write_feedback(out / "feedback.csv", [("a/crazing_1.jpg", "crazing")] * rows)
    report = tmp_path / "export_report.json"
    report.write_text(
        json.dumps({"rows": rows, "csv_md5": "f" * 32 if tamper else md5, "per_class": {}}),
        encoding="utf-8",
    )
    p["review"]["export"] = {"out_dir": str(out), "report": str(report)}
    p["retrain"]["min_feedback_rows"] = minimum
    return p, md5


def test_check_export(tmp_path):
    p, md5 = export_params(tmp_path)
    got = rt.check_export(p)
    assert got["md5"] == md5 and got["rows"] == 3


def test_check_export_refuses_too_few_rows(tmp_path):
    p, _ = export_params(tmp_path, rows=1, minimum=2)
    with pytest.raises(rt.RetrainError, match="min_feedback_rows"):
        rt.check_export(p)


def test_check_export_refuses_a_changed_csv(tmp_path):
    p, _ = export_params(tmp_path, tamper=True)
    with pytest.raises(rt.RetrainError, match="differs"):
        rt.check_export(p)


def metrics_for(ckpt: Path, commit="c" * 40, md5=MD5, smoke=False):
    return {
        "smoke": smoke,
        "provenance": {"git_commit": commit},
        "feedback": {"md5": md5, "rows": 3},
        "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    }


def test_verify_output(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"w")
    rt.verify_output(metrics_for(ckpt), ckpt, "c" * 40, MD5)
    cases = [
        (metrics_for(ckpt, commit="d" * 40), "commit"),
        (metrics_for(ckpt, md5="e" * 32), "MD5"),
        (metrics_for(ckpt, smoke=True), "smoke"),
        ({**metrics_for(ckpt), "checkpoint_sha256": "0" * 64}, "SHA-256"),
    ]
    for metrics, message in cases:
        with pytest.raises(rt.RetrainError, match=message):
            rt.verify_output(metrics, ckpt, "c" * 40, MD5)


def run_wait(statuses, timeout_s=1000.0):
    seq, t = iter(statuses), [0.0]

    def sleep(s):
        t[0] += s

    return rt.wait_for_kaggle(
        lambda: next(seq), poll_s=60, timeout_s=timeout_s, sleep=sleep, clock=lambda: t[0]
    )


def test_wait_needs_an_active_status_before_complete():
    assert run_wait(["running", "running", "complete"]) == "complete"
    # a stale "complete" from the previous run is not trusted at once
    assert run_wait(["complete", "queued", "running", "complete"]) == "complete"
    assert run_wait(["complete", "complete", "complete"]) == "complete"  # after stale_polls


def test_wait_fails_on_error_and_timeout():
    with pytest.raises(rt.RetrainError, match="error"):
        run_wait(["running", "error"])
    with pytest.raises(rt.RetrainError, match="not finished"):
        run_wait(["running"] * 100, timeout_s=300)


def test_challenger_tags():
    metrics = {
        "model": "efficientnet_b0",
        "checkpoint_sha256": "a" * 64,
        "feedback": {"md5": MD5, "rows": 7},
        "run_uid": "u" * 32,
    }
    tags = rt.challenger_tags(metrics, 1.5, 0.8)
    for key in ("model", "checkpoint_sha256", "temperature", "review_threshold"):
        assert key in tags  # what the gate and the ONNX export need
    assert tags["feedback_md5"] == MD5 and tags["feedback_rows"] == "7"
    assert all(isinstance(v, str) for v in tags.values())


# ---------------------------------------------------------------- flow: order and resume


def test_run_executes_steps_in_order_and_resumes(tmp_path, monkeypatch):
    p = copy.deepcopy(PARAMS)
    p["retrain"]["state_dir"] = str(tmp_path / "state")
    calls = []

    def fake(name, fail=False):
        def step(params, state, *api):
            calls.append(name)
            if fail:
                raise rt.RetrainError(f"{name} failed")
            state.finish(name)

        return step

    for name in rt.STEPS:
        monkeypatch.setattr(rt, f"step_{name}", fake(name, fail=name == "wait"))
    with pytest.raises(rt.RetrainError, match="wait failed"):
        rt.run(p, api_factory=lambda: "api")
    assert calls == ["preflight", "feedback_dataset", "push", "wait"]

    calls.clear()
    monkeypatch.setattr(rt, "step_wait", fake("wait"))
    monkeypatch.setattr(
        rt, "step_record", lambda params, state: state.finish("record", cycle_file="c")
    )
    rt.run(p, api_factory=lambda: "api")
    assert calls == ["wait", "download", "register", "calibrate", "challenger", "versions", "gate"]
    with pytest.raises(rt.RetrainError, match="finished"):
        rt.run(p, api_factory=lambda: "api")
    calls.clear()
    monkeypatch.setattr(rt, "step_record", fake("record"))
    rt.run(p, fresh=True, api_factory=lambda: "api")
    assert calls == list(rt.STEPS)


def test_cycle_record_reports_what_the_owner_asked_for(tmp_path):
    state = rt.State(tmp_path / "s.json")
    state.data.update(
        commit="c" * 40,
        feedback_md5=MD5,
        feedback_rows=90,
        feedback_per_class={"crazing": 15},
        kaggle_pushed_at="t0",
        kaggle_completed_at="t1",
        kaggle_wall_time_s=900,
        training_time_s=300,
        device="cuda",
        run_uid="u" * 32,
        n_train=240,
        champion_version="2",
        version="3",
        train_val_macro_f1=0.98,
    )
    gate = {
        "decision": "promote",
        "failed_checks": [],
        "champion_metrics": {"macro_f1": 0.978},
        "challenger_metrics": {"macro_f1": 0.983},
    }
    rec = rt.cycle_record(state, gate, "results/promotion/gate_3.json")
    assert rec["feedback"]["rows"] == 90 and rec["kaggle"]["wall_time_s"] == 900
    assert rec["gate"]["challenger_val_macro_f1"] == 0.983
    assert rec["gate"]["champion_val_macro_f1"] == 0.978
    assert rec["gate"]["file"] == "results/promotion/gate_3.json"
    assert "record" in rec["gate"]["mode"] and "SIMULATED" in rec["note"]


def test_retrain_params_and_gitignore():
    r = PARAMS["retrain"]
    assert r["versions_dir"] == "models/versions" and r["min_feedback_rows"] >= 1
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/models/versions/" in ignore  # weights: DVC, never git
    # a wildcard makes DVC's matcher treat models/ as ignored and skip models/versions.dvc
    assert not any(line.startswith("/models/*") for line in ignore)
    assert PARAMS["kaggle"]["feedback_dataset_slug"] != PARAMS["kaggle"]["dataset_slug"]
