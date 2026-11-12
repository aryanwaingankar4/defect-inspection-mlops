"""TASK-030 (ADR-036): loop chunks, the fault-injection copy, cycle records and the summary.
Synthetic files only (no NEU pixels); the committed chunks file is checked against the split."""

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.config import load_params
from src.db import migrate
from src.pipelines import loop as lp
from src.pipelines import stream as st
from tests import review_helpers as rh

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO_ROOT / "configs" / "params.yaml")
CLASSES = rh.CLASSES


# ---------------------------------------------------------------- chunks


def test_assign_chunks_keeps_the_pilot_and_cuts_the_rest_in_order():
    order = [f"p{i}" for i in range(10)]
    rows = lp.assign_chunks(order, {"p1", "p4"}, {"a": 5, "b": 3})
    assert [r["path"] for r in rows] == order and [r["order"] for r in rows] == list(
        map(str, range(10))
    )
    chunk = {r["path"]: r["chunk"] for r in rows}
    assert chunk["p1"] == chunk["p4"] == "pilot"
    assert [p for p in order if chunk[p] == "a"] == ["p0", "p2", "p3", "p5", "p6"]
    assert [p for p in order if chunk[p] == "b"] == ["p7", "p8", "p9"]


@pytest.mark.parametrize(
    ("pilot", "sizes", "message"),
    [
        ({"p1"}, {"a": 5, "b": 3}, "remain"),
        ({"zz"}, {"a": 10}, "not in the stream order"),
        (set(), {"pilot": 10}, "reserved"),
    ],
)
def test_assign_chunks_refuses_bad_inputs(pilot, sizes, message):
    with pytest.raises(lp.LoopError, match=message):
        lp.assign_chunks([f"p{i}" for i in range(10)], pilot, sizes)


def read_chunks(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_committed_chunks_follow_the_stream_order_and_are_disjoint():
    rows = read_chunks(REPO_ROOT / PARAMS["loop"]["chunks_file"])
    order = st.stream_order(PARAMS)
    assert [r["path"] for r in rows] == [r["path"] for r in order]  # every stream image, once
    counts = {c: sum(r["chunk"] == c for r in rows) for c in {r["chunk"] for r in rows}}
    pilot = json.loads((REPO_ROOT / PARAMS["loop"]["pilot"]["cycle_record"]).read_text("utf-8"))
    assert counts == {"pilot": pilot["feedback"]["rows"], **PARAMS["loop"]["chunks"]}
    for name in PARAMS["loop"]["chunks"]:  # each chunk is one contiguous run of the non-pilot order
        idx = [
            i for i, r in enumerate(r for r in rows if r["chunk"] != "pilot") if r["chunk"] == name
        ]
        assert idx == list(range(idx[0], idx[0] + len(idx)))
    raw = (REPO_ROOT / PARAMS["loop"]["chunks_file"]).read_bytes()
    assert b"\r\n" not in raw


def test_loop_params():
    lo = PARAMS["loop"]
    assert lo["fault_injection"]["rule"] == "shift_to_next_class"
    assert lo["fault_injection"]["out_dir"] != PARAMS["review"]["export"]["out_dir"]
    k = PARAMS["kaggle"]
    assert lo["fault_injection"]["feedback_dataset_slug"] != k["feedback_dataset_slug"]
    assert lo["fault_injection"]["state_dir"] != PARAMS["retrain"]["state_dir"]
    assert "/data/feedback_fault/" in (REPO_ROOT / ".gitignore").read_text("utf-8").splitlines()


@pytest.fixture
def stream_params(tmp_path):
    items = [("crazing", "production_stream", i) for i in range(6)] + [("patches", "val", 9)]
    split, raw, _ = rh.make_archive(tmp_path / "s", items)
    p = copy.deepcopy(PARAMS)
    p["data"]["split_file"], p["data"]["raw_dir"] = str(split), str(raw)
    order = [r["path"] for r in st.stream_order(p)]
    rows = lp.assign_chunks(order, {order[2]}, {"cycle2": 3, "cycle3": 2})
    chunks = tmp_path / "chunks.csv"
    with chunks.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lp.CHUNK_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    p["loop"]["chunks_file"] = str(chunks)
    p["loop"]["chunks"] = {"cycle2": 3, "cycle3": 2}
    return p, rows


def test_stream_sends_exactly_one_chunk(stream_params, monkeypatch):
    params, rows = stream_params
    sent = []

    def fake_post(url, key, data, name):
        sent.append(data)
        return 200, {"reasons": ["low_confidence"], "routed_to_review": True}

    monkeypatch.setattr(st, "post_image", fake_post)
    out = st.send(params, "http://x", "k", n=99, start=99, chunk="cycle3")
    raw = Path(params["data"]["raw_dir"])
    want = [r["path"] for r in rows if r["chunk"] == "cycle3"]
    assert sent == [(raw / p).read_bytes() for p in want]
    assert out["chunk"] == "cycle3" and out["sent"] == 2 and out["routed_to_review"] == 2
    assert "not brand-new" in out["note"]


def test_stream_refuses_the_pilot_unknown_chunks_and_a_stale_file(stream_params):
    params, rows = stream_params
    with pytest.raises(st.StreamError, match="never sent again"):
        st.chunk_rows(params, "pilot")
    with pytest.raises(st.StreamError, match="no chunk"):
        st.chunk_rows(params, "cycle9")
    params["review"]["stream"]["seed"] = 999  # another order: the chunks file no longer matches
    with pytest.raises(st.StreamError, match="stream order"):
        st.chunk_rows(params, "cycle2")


def test_make_chunks_needs_the_pilot_export(tmp_path, stream_params):
    params, _ = stream_params
    order = [r["path"] for r in st.stream_order(params)]
    export_dir = tmp_path / "feedback"
    md5 = write_export(export_dir, [{"path": order[2], "label": "crazing"}])
    record = tmp_path / "pilot.json"
    record.write_text(json.dumps({"feedback": {"md5": md5, "rows": 1}}), "utf-8")
    params["review"]["export"]["out_dir"] = str(export_dir)
    params["loop"]["pilot"]["cycle_record"] = str(record)
    params["loop"]["chunks_file"] = str(tmp_path / "out" / "chunks.csv")
    out = lp.make_chunks(params)
    assert out["chunks"] == {"cycle2": 3, "cycle3": 2, "pilot": 1}
    rows = read_chunks(tmp_path / "out" / "chunks.csv")
    assert [r["path"] for r in rows if r["chunk"] == "pilot"] == [order[2]]
    write_export(export_dir, [{"path": order[3], "label": "crazing"}])  # a later export
    with pytest.raises(lp.LoopError, match="not the pilot export"):
        lp.make_chunks(params)


# ---------------------------------------------------------------- fault injection


def write_export(folder: Path, rows, report: Path | None = None) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "feedback.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["path", "label", "reviewer_kind", "request_id"], lineterminator="\n"
        )
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"reviewer_kind": "simulated", "request_id": f"{i:032x}", **r})
    md5 = hashlib.md5(csv_path.read_bytes()).hexdigest()
    if report is not None:
        per_class = {c: sum(r["label"] == c for r in rows) for c in CLASSES}
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps({"rows": len(rows), "csv_md5": md5, "per_class": per_class}), "utf-8"
        )
    return md5


def test_shift_label_is_cyclic_and_never_the_truth():
    assert [lp.shift_label(c, CLASSES) for c in CLASSES] == [*CLASSES[1:], CLASSES[0]]


@pytest.fixture
def fault_params(tmp_path):
    split, raw, _ = rh.make_archive(
        tmp_path / "a",
        [(c, "production_stream", i) for i, c in enumerate(CLASSES)]
        + [("crazing", "production_stream", 99)],
    )
    p = copy.deepcopy(PARAMS)
    p["data"]["split_file"], p["data"]["raw_dir"] = str(split), str(raw)
    p["review"]["export"]["out_dir"] = str(tmp_path / "feedback")
    p["review"]["export"]["report"] = str(tmp_path / "results" / "export_report.json")
    p["loop"]["fault_injection"]["out_dir"] = str(tmp_path / "feedback_fault")
    p["loop"]["fault_injection"]["report"] = str(tmp_path / "results" / "fault" / "report.json")
    rows = [{"path": f"{c}/{c}_{i}.png", "label": c} for i, c in enumerate(CLASSES)]
    rows += [{"path": "crazing/crazing_6.png", "label": "crazing"}]
    md5 = write_export(
        Path(p["review"]["export"]["out_dir"]), rows, Path(p["review"]["export"]["report"])
    )
    return p, rows, md5


def test_fault_export_shifts_every_label_on_a_copy(fault_params):
    params, rows, md5 = fault_params
    real_csv = Path(params["review"]["export"]["out_dir"]) / "feedback.csv"
    before = real_csv.read_bytes()
    report = lp.fault_export(params)
    assert real_csv.read_bytes() == before  # the real export is untouched
    shifted = read_chunks(Path(params["loop"]["fault_injection"]["out_dir"]) / "feedback.csv")
    assert [r["path"] for r in shifted] == [r["path"] for r in rows]
    assert all(
        s["label"] == lp.shift_label(r["label"], CLASSES)
        for s, r in zip(shifted, rows, strict=True)
    )
    assert {r["reviewer_kind"] for r in shifted} == {"fault_injection"}
    assert report["FAULT_INJECTION"] is True and report["labels_changed"] == report["rows"] == 7
    assert report["source"]["csv_md5"] == md5 and report["per_class"]["inclusion"] == 2
    saved = json.loads(Path(params["loop"]["fault_injection"]["report"]).read_text("utf-8"))
    out_csv = Path(params["loop"]["fault_injection"]["out_dir"]) / "feedback.csv"
    assert saved["csv_md5"] == hashlib.md5(out_csv.read_bytes()).hexdigest()


def test_fault_export_refuses_a_changed_export_or_the_real_folder(fault_params):
    params, rows, _ = fault_params
    bad = copy.deepcopy(params)
    bad["loop"]["fault_injection"]["out_dir"] = params["review"]["export"]["out_dir"]
    with pytest.raises(lp.LoopError, match="real feedback folder"):
        lp.fault_export(bad)
    write_export(Path(params["review"]["export"]["out_dir"]), rows[:2])  # report not updated
    with pytest.raises(lp.LoopError, match="not the committed export"):
        lp.fault_export(params)


def test_fault_feedback_passes_the_training_checks(fault_params):
    """The shifted copy is still production_stream paths with valid labels: training accepts it,
    so only the gate can stop it (that is what the run tests)."""
    from src.training.feedback import load_feedback

    params, _, _ = fault_params
    report = lp.fault_export(params)
    fb = load_feedback(
        Path(report["csv"]), report["csv_md5"], Path(params["data"]["split_file"]), CLASSES
    )
    assert len(fb.rows) == 7


# ---------------------------------------------------------------- cycle records and summary


def gate_file(
    folder: Path, name: str, champ: str, chall: str, decision: str, applied: bool, failed=()
):
    checks = {
        k: {"reason": f"{k} ok", "passed": k not in failed}
        for k in ("macro_f1", "class_recall", "mcnemar", "onnx_parity")
    }
    record = {
        "decision": decision,
        "failed_checks": list(failed),
        "champion": {"version": champ},
        "challenger": {"version": chall},
        "champion_metrics": {"macro_f1": 0.98},
        "challenger_metrics": {"macro_f1": 0.5 if failed else 0.99},
        "checks": checks,
        "applied": applied,
    }
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), "utf-8")
    return path


def retrain_record(
    folder: Path, uid: str, version: str, md5: str, rows: int, gate: Path, fault=False, cal=None
):
    calibration = cal or {"calibration": "fitted", "temperature": 1.5}
    (folder / uid).mkdir(parents=True, exist_ok=True)
    (folder / uid / "calibration.json").write_text(json.dumps(calibration), "utf-8")
    record = {
        "run_uid": uid,
        "fault_injection": fault,
        "challenger_version": version,
        "feedback": {"md5": md5, "rows": rows},
        "kaggle": {"wall_time_s": 300, "training_time_s": 40},
        "gate": {"file": gate.as_posix()},
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"cycle_{uid}.json").write_text(json.dumps(record), "utf-8")


@pytest.fixture
def loop_files(fault_params, tmp_path):
    params, _, md5 = fault_params
    params["loop"]["out_dir"] = str(tmp_path / "loop")
    params["retrain"]["results_dir"] = str(tmp_path / "retrain")
    params["promotion"]["log_file"] = str(tmp_path / "promotion_log.json")
    params["loop"]["final_out"] = str(tmp_path / "loop" / "final.json")
    promo = tmp_path / "promotion"
    rec_gate = gate_file(promo, "gate_4.json", "3", "4", "promote", False)
    applied = gate_file(promo, "gate_4.2.json", "3", "4", "promote", True)
    retrain_record(Path(params["retrain"]["results_dir"]), "aaa", "4", md5, 7, rec_gate)
    folder = lp.cycle_dir(params, "2")
    lp.write_json(folder / "stream.json", {"chunk": "cycle2", "sent": 520, "routed_to_review": 30})
    lp.write_json(folder / "review.json", {"error_rate": 0.0, "labelled": 30, "flipped": 0})
    return params, rec_gate, applied


COUNTS = {"predictions": {"3": {"n": 520, "routed_to_review": 30}}, "labels": []}


def test_record_cycle_points_at_every_file(loop_files):
    params, rec_gate, applied = loop_files
    r = lp.record_cycle(params, "2", applied, COUNTS)
    folder = lp.cycle_dir(params, "2")
    assert json.loads((folder / "cycle.json").read_text("utf-8")) == r
    assert (folder / "export_report.json").read_bytes() == Path(
        params["review"]["export"]["report"]
    ).read_bytes()
    assert (r["new_labels"], r["feedback_rows_used"], r["chunk"]) == (30, 7, "cycle2")
    assert (r["champion_version"], r["challenger_version"], r["decision"]) == ("3", "4", "promote")
    assert r["applied"] and r["applied_in"] == applied.as_posix()
    assert r["gate_record_file"] == rec_gate.as_posix() and r["prediction_log_counts"] == COUNTS
    assert r["reviewer_error"] == {"kind": "simulated", "error_rate": 0.0, "flipped": 0}


def test_record_cycle_refuses_another_export_or_missing_summaries(loop_files):
    params, _, applied = loop_files
    (lp.cycle_dir(params, "2") / "review.json").unlink()
    with pytest.raises(lp.LoopError, match="review.json"):
        lp.record_cycle(params, "2", applied, COUNTS)
    write_export(
        Path(params["review"]["export"]["out_dir"]), [], Path(params["review"]["export"]["report"])
    )
    with pytest.raises(lp.LoopError, match="not the export"):
        lp.record_cycle(params, "2", applied, COUNTS)


def test_pilot_facts_come_from_the_log_before_cycle_2():
    counts = {
        "predictions": {"2": {"n": 1110, "routed_to_review": 67}},
        "labels": [
            {
                "reviewer_kind": "simulated",
                "simulated_error_rate": 0.0,
                "n": 67,
                "simulated_flipped": 0,
            }
        ],
    }
    stream, review = lp.pilot_facts(counts, 67)
    assert (stream["sent"], stream["routed_to_review"]) == (1110, 67)
    assert (review["error_rate"], review["labelled"], review["flipped"]) == (0.0, 67, 0)
    later = copy.deepcopy(counts)
    later["predictions"]["3"] = {"n": 520, "routed_to_review": 30}
    with pytest.raises(lp.LoopError, match="more than the pilot"):
        lp.pilot_facts(later, 67)


INHERITED_CAL = {
    "calibration": "inherited_T",
    "temperature": 1.5115,
    "fit": {"temperature": 0.25, "at_bound": True},
    "inherited_from": {"version": "4"},
}


def fault_cycle(params, tmp_path):
    report = lp.fault_export(params)
    promo = tmp_path / "promotion"
    gate = gate_file(
        promo, "gate_6.json", "4", "6", "reject", False, failed=("macro_f1", "class_recall")
    )
    retrain_record(
        Path(params["retrain"]["results_dir"]), "fff", "6", report["csv_md5"], 7, gate,
        fault=True, cal=INHERITED_CAL,
    )  # fmt: skip
    return gate


def test_fault_record_is_record_mode_and_labelled(loop_files, tmp_path):
    params, _, _ = loop_files
    gate = fault_cycle(params, tmp_path)
    r = lp.record_cycle(params, lp.FAULT, gate)
    assert r["fault_injection"] and "FAULT INJECTION" in r["note"] and r["new_labels"] == 0
    assert r["reviewer_error"]["kind"] == "broken" and r["reviewer_error"]["labels_changed"] == 7
    assert r["failed_checks"] == ["macro_f1", "class_recall"] and r["applied_in"] is None
    assert (lp.cycle_dir(params, lp.FAULT) / "cycle.json").is_file()
    with pytest.raises(lp.LoopError, match="wrong cycle"):  # a fault run recorded as a normal one
        lp.record_cycle(params, "3", gate, COUNTS)
    applied = gate_file(tmp_path / "promotion", "gate_6.2.json", "4", "6", "reject", True)
    with pytest.raises(lp.LoopError, match="record mode"):
        lp.record_cycle(params, lp.FAULT, applied)


def test_summary_and_readme_table(loop_files, tmp_path):
    params, _, applied = loop_files
    lp.record_cycle(params, "2", applied, COUNTS)
    lp.record_cycle(params, lp.FAULT, fault_cycle(params, tmp_path))
    s = lp.summarise(params)
    assert [c["cycle"] for c in s["cycles"]] == ["2", "fault"]
    readme = (Path(params["loop"]["out_dir"]) / "README.md").read_text("utf-8")
    assert lp.render_table(s) in readme and "SIMULATED" in readme and "not brand-new" in readme
    table = lp.render_table(s).splitlines()
    assert table[0].split(" | ")[1:6] == [
        "Labels",
        "Reviewer error",
        "Challenger T",
        "Val macro-F1 (champion vs challenger)",
        "Verdict",
    ]
    assert "PROMOTE (applied)" in table[2] and "gate_4.2.json" in table[2]
    assert "fitted (1.5000)" in table[2] and "ADR-037" in readme
    assert "inherited_T from v4 (1.5115; fit on bound 0.25)" in table[3]
    assert s["cycles"][1]["calibration"]["source"] == "inherited_T"
    assert "REJECT (record only: fault injection)" in table[3]
    assert "failed: macro_f1, class_recall" in table[3] and "BROKEN: all 7 labels" in table[3]
    assert s["kaggle_wall_time_s_total"] == 600


def test_verdict_names_where_a_record_only_gate_was_applied():
    c = {
        "decision": "promote",
        "fault_injection": False,
        "applied": False,
        "applied_in": "x/gate_3.2.json",
    }
    assert lp.verdict(c) == "PROMOTE (record only; applied in gate_3.2.json)"


def test_committed_loop_readme_matches_the_summary():
    out = REPO_ROOT / PARAMS["loop"]["out_dir"]
    if not (out / "loop_summary.json").is_file():
        pytest.skip("no loop summary committed yet")
    summary = json.loads((out / "loop_summary.json").read_text("utf-8"))
    assert (out / "README.md").read_text("utf-8") == lp.render_readme(summary)
    for c in summary["cycles"]:
        assert (REPO_ROOT / c["file"]).is_file()
        if c["gate_file"] is None:  # only a fault run stopped before the gate (ADR-036)
            assert c["fault_injection"] and c["decision"] == "stopped_before_gate"
            retrain = json.loads((REPO_ROOT / c["retrain_record"]).read_text("utf-8"))
            assert retrain["gate"]["file"] is None and retrain["gate"]["reason"]
            assert not c["applied"]
            continue
        gate = json.loads((REPO_ROOT / c["gate_file"]).read_text("utf-8"))
        assert gate["decision"] == c["decision"] and gate["applied"] == c["applied"]
        if c["fault_injection"]:
            assert gate["applied"] is False


# ---------------------------------------------------------------- final test (report only)


def test_paired_difference_uses_the_same_resamples():
    y = np.array([0, 0, 1, 1, 2, 2])
    a = np.array([0, 1, 1, 1, 2, 0])
    samples = [np.arange(6), np.array([0, 0, 2, 3, 4, 5])]
    d = lp.paired_difference(y, a, y.copy(), 3, samples, 0.95)
    assert d["value"] > 0 and 0 < d["ci_low"] <= d["ci_high"]
    same = lp.paired_difference(y, a, a.copy(), 3, samples, 0.95)
    assert same == {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def test_final_versions_need_the_initial_champion(tmp_path, monkeypatch):
    from src.training import export_onnx as ex

    p = copy.deepcopy(PARAMS)
    p["promotion"]["log_file"] = str(tmp_path / "log.json")
    (tmp_path / "log.json").write_text(
        json.dumps([{"decision": "initial_champion", "version": "2"}]), "utf-8"
    )
    monkeypatch.setattr(ex, "registered_version", lambda params, alias: {"version": "5"})
    assert lp.final_versions(p) == ("2", "5")
    monkeypatch.setattr(ex, "registered_version", lambda params, alias: {"version": "2"})
    with pytest.raises(lp.LoopError, match="nothing to compare"):
        lp.final_versions(p)


def test_final_test_runs_once(tmp_path):
    p = copy.deepcopy(PARAMS)
    p["loop"]["final_out"] = str(tmp_path / "final.json")
    (tmp_path / "final.json").write_text("{}", "utf-8")
    with pytest.raises(lp.LoopError, match="runs once"):
        lp.final_test(p)


# ---------------------------------------------------------------- prediction log counts


def test_loop_counts(tmp_path):
    pytest.importorskip("sqlalchemy")
    from src.db.store import PredictionLog
    from src.review.queue import submit

    db = PredictionLog("sqlite:///" + (tmp_path / "p.db").as_posix())
    migrate.upgrade(db.engine)
    try:
        db.add(rh.prediction(f"{1:032x}", reasons=("low_confidence",), model_version="3"))
        db.add(rh.prediction(f"{2:032x}", model_version="3"))
        db.add(rh.prediction(f"{3:032x}", reasons=("ood",), model_version="4"))
        submit(db, f"{1:032x}", "crazing", CLASSES, reviewer_kind="simulated", reviewer_id="s",
               simulated_error_rate=0.05, simulated_flipped=True)  # fmt: skip
        submit(db, f"{3:032x}", "patches", CLASSES, reviewer_kind="simulated", reviewer_id="s",
               simulated_error_rate=0.05, simulated_flipped=False)  # fmt: skip
        c = db.loop_counts()
    finally:
        db.dispose()
    assert c["predictions"] == {
        "3": {"n": 2, "routed_to_review": 1},
        "4": {"n": 1, "routed_to_review": 1},
    }
    assert c["labels"] == [
        {"reviewer_kind": "simulated", "simulated_error_rate": 0.05, "n": 2, "simulated_flipped": 1}
    ]


def test_re_recording_a_cycle_uses_its_saved_export_report(loop_files):
    params, _, applied = loop_files
    first = lp.record_cycle(params, "2", applied, COUNTS)
    write_export(  # a later cycle exports again
        Path(params["review"]["export"]["out_dir"]), [], Path(params["review"]["export"]["report"])
    )
    assert lp.record_cycle(params, "2", applied, COUNTS) == first


def test_a_fault_run_stopped_before_the_gate_is_recorded_without_a_gate_file(loop_files, tmp_path):
    params, _, applied = loop_files
    report = lp.fault_export(params)
    log = [{"decision": "promote", "version": "4", "gate_file": applied.as_posix()}]
    Path(params["promotion"]["log_file"]).write_text(json.dumps(log), "utf-8")
    folder = Path(params["retrain"]["results_dir"])
    (folder / "fff").mkdir(parents=True)
    (folder / "fff" / "calibration.json").write_text(
        json.dumps({"calibration": "fitted", "temperature": 1.3}), "utf-8"
    )
    record = {
        "run_uid": "fff",
        "fault_injection": True,
        "champion_version": "4",
        "challenger_version": "6",
        "challenger_val_macro_f1_at_training": 0.74,
        "feedback": {"md5": report["csv_md5"], "rows": 7},
        "kaggle": {"wall_time_s": 183, "training_time_s": 26},
        "gate": {
            "decision": "stopped_before_gate",
            "failed_checks": ["review_threshold_risk (ADR-023)"],
            "reason": "risk upper bound 0.29 above 0.05",
            "file": None,
        },
    }
    (folder / "cycle_fff.json").write_text(json.dumps(record), "utf-8")
    r = lp.record_cycle(params, lp.FAULT, None)
    assert r["decision"] == "stopped_before_gate" and r["gate_file"] is None
    assert r["val_macro_f1"] == {"champion": 0.99, "challenger": 0.74}
    assert "gate did not run" in r["val_macro_f1_source"]
    with pytest.raises(lp.LoopError, match="only the fault-injection run"):
        lp.record_cycle(params, "2", None, COUNTS)
    lp.record_cycle(params, "2", applied, COUNTS)
    row = lp.render_table(lp.summarise(params)).splitlines()[3]
    assert "STOPPED before the gate" in row and "none (gate not reached)" in row
    assert "review_threshold_risk (ADR-023)" in row
