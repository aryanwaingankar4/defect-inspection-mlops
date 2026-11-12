"""Loop cycles after the pilot (TASK-030, ADR-036). THE STREAM AND THE REVIEWER ARE SIMULATED.

    make loop-chunks                        # data/splits/loop_chunks.csv from the pilot export
    make loop-record CYCLE=2 GATE=results/promotion/gate_4.2.json
    make fault-export                       # BROKEN reviewer: copy of the export, labels shifted
    make loop-record CYCLE=fault GATE=results/promotion/gate_6.json
    make loop-final                         # once: v2 vs final @champion on test_frozen (report)
    make loop-summary                       # results/loop/loop_summary.json + README.md table

Chunks: every ``production_stream`` image in the seeded stream order. The pilot's 67 labelled
images are ``pilot``; the rest are cut in order into the sizes of ``loop.chunks``. They are NOT
brand-new traffic: v2 predicted all of them in the pilot, but none was labelled or trained on.

The fault-injection copy never enters the feedback history (prediction log, ``data/feedback``
DVC versions, the real private Kaggle feedback dataset).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import load_params
from src.pipelines.stream import PILOT_CHUNK, stream_order
from src.review.queue import class_names
from src.training.feedback import file_md5

DECISION_RECORD = "ADR-036"
SIMULATED = "SIMULATED stream and SIMULATED reviewer (ADR-031); real Kaggle GPU training"
NOT_NEW = "not brand-new traffic: v2 predicted every chunk image in the pilot (none labelled)"
FAULT = "fault"
FAULT_DIR = "fault_injection"
CHUNK_COLUMNS = ("order", "path", "chunk")


class LoopError(RuntimeError):
    """Nothing was written."""


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data, indent=2) + "\n").encode())


def feedback_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- chunks


def assign_chunks(
    order: Sequence[str], pilot: set[str], sizes: Mapping[str, int]
) -> list[dict[str, str]]:
    """``order,path,chunk`` for every stream image: pilot images keep ``pilot``, the rest are cut
    in stream order into ``sizes`` (which must use them all)."""
    missing = pilot - set(order)
    if missing:
        raise LoopError(f"{len(missing)} pilot images are not in the stream order")
    if PILOT_CHUNK in sizes:
        raise LoopError(f"{PILOT_CHUNK!r} is reserved for the pilot")
    rest = [p for p in order if p not in pilot]
    if sum(sizes.values()) != len(rest):
        raise LoopError(f"loop.chunks sum to {sum(sizes.values())}, but {len(rest)} images remain")
    names = iter([name for name, n in sizes.items() for _ in range(int(n))])
    return [
        {"order": str(i), "path": p, "chunk": PILOT_CHUNK if p in pilot else next(names)}
        for i, p in enumerate(order)
    ]


def make_chunks(params: Mapping[str, Any]) -> dict[str, Any]:
    """Write ``loop.chunks_file`` from the pilot export (its MD5 must be the pilot's)."""
    pilot_record = read_json(params["loop"]["pilot"]["cycle_record"])
    csv_path = Path(params["review"]["export"]["out_dir"]) / "feedback.csv"
    if not csv_path.is_file() or file_md5(csv_path) != pilot_record["feedback"]["md5"]:
        raise LoopError(
            f"{csv_path} is not the pilot export (MD5 {pilot_record['feedback']['md5']}): check "
            "out the pilot's data/feedback.dvc and run `dvc checkout data/feedback`"
        )
    pilot = {r["path"] for r in feedback_csv_rows(csv_path)}
    order = [r["path"] for r in stream_order(params)]
    rows = assign_chunks(order, pilot, params["loop"]["chunks"])
    out = Path(params["loop"]["chunks_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CHUNK_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {"file": out.as_posix(), "chunks": dict(Counter(r["chunk"] for r in rows))}


# ---------------------------------------------------------------- fault injection


def shift_label(label: str, classes: Sequence[str]) -> str:
    """The BROKEN reviewer: the next class in sorted class order (cyclic)."""
    return classes[(list(classes).index(label) + 1) % len(classes)]


def fault_export(params: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the latest real export with EVERY label shifted (ADR-036); returns the report."""
    f = params["loop"]["fault_injection"]
    if f["rule"] != "shift_to_next_class":
        raise LoopError(f"unknown fault rule {f['rule']!r}")
    real_report_path = Path(params["review"]["export"]["report"])
    real_csv = Path(params["review"]["export"]["out_dir"]) / "feedback.csv"
    real_report = read_json(real_report_path)
    if file_md5(real_csv) != real_report["csv_md5"]:
        raise LoopError(f"{real_csv} is not the committed export: re-export first")
    out_dir = Path(f["out_dir"])
    if out_dir.resolve() == real_csv.parent.resolve():
        raise LoopError("the fault copy must not be written into the real feedback folder")
    classes = class_names(params)
    rows = feedback_csv_rows(real_csv)
    shifted = [
        {**r, "label": shift_label(r["label"], classes), "reviewer_kind": "fault_injection"}
        for r in rows
    ]
    from src.review.export import write_csv

    csv_path = out_dir / "feedback.csv"
    md5 = write_csv(shifted, csv_path)
    report = {
        "FAULT_INJECTION": True,
        "decision_record": DECISION_RECORD,
        "note": "BROKEN simulated reviewer: every label of the real export moved to the next "
        "class. A copy for one gate test; never part of the feedback history.",
        "rule": f["rule"],
        "class_order": classes,
        "source": {
            "report": real_report_path.as_posix(),
            "csv_md5": real_report["csv_md5"],
            "rows": real_report["rows"],
        },
        "rows": len(shifted),
        "labels_changed": sum(a["label"] != b["label"] for a, b in zip(rows, shifted, strict=True)),
        "per_class": {c: sum(r["label"] == c for r in shifted) for c in classes},
        "per_class_true": real_report["per_class"],
        "csv": csv_path.as_posix(),
        "csv_md5": md5,
        "exported_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_json(Path(f["report"]), report)
    return report


# ---------------------------------------------------------------- cycle records


def cycle_dir(params: Mapping[str, Any], cycle: str) -> Path:
    out = Path(params["loop"]["out_dir"])
    return out / FAULT_DIR if cycle == FAULT else out / f"cycle_{int(cycle)}"


def retrain_record_for(params: Mapping[str, Any], version: str) -> Path:
    found = [
        p
        for p in sorted(Path(params["retrain"]["results_dir"]).glob("cycle_*.json"))
        if str(read_json(p)["challenger_version"]) == str(version)
    ]
    if len(found) != 1:
        raise LoopError(f"expected one retrain record for v{version}, found {len(found)}")
    return found[0]


def applied_in(params: Mapping[str, Any], version: str) -> str | None:
    """The first applied gate file for ``version`` in the promotion log (None if never)."""
    log_file = Path(params["promotion"]["log_file"])
    for entry in read_json(log_file) if log_file.is_file() else []:
        if str(entry.get("challenger_version")) == str(version) and entry.get("gate_file"):
            return str(entry["gate_file"])
    return None


def calibration_of(params: Mapping[str, Any], retrain: Mapping[str, Any]) -> dict[str, Any]:
    """How the challenger's T was set (ADR-006 fitted, or ADR-037 inherited), from its
    ``calibration.json``."""
    path = Path(params["retrain"]["results_dir"]) / retrain["run_uid"] / "calibration.json"
    cal = read_json(path)
    out = {
        "source": cal.get("calibration", "fitted"),
        "temperature": cal["temperature"],
        "file": path.as_posix(),
    }
    if out["source"] == "inherited_T":
        out["inherited_from_version"] = cal["inherited_from"]["version"]
        out["fit_on_bound"] = cal["fit"]["temperature"]
    return out


def pilot_facts(counts: Mapping[str, Any], pilot_rows: int) -> tuple[dict, dict]:
    """The pilot's stream and review summaries, read back from the prediction log. Only valid
    before cycle 2 starts: every prediction must be v2's and every label the pilot's."""
    preds, labels = counts["predictions"], counts["labels"]
    if set(preds) != {"2"} or sum(x["n"] for x in labels) != pilot_rows:
        raise LoopError("the prediction log holds more than the pilot: record cycle 1 first")
    (lab,) = labels
    stream = {
        "stream": "simulated, production_stream (ADR-031)",
        "chunk": "all 1,110 production_stream images (pilot)",
        "sent": preds["2"]["n"],
        "routed_to_review": preds["2"]["routed_to_review"],
        "source": "prediction log (loop_counts), read before cycle 2",
    }
    review = {
        "reviewer": "simulated (ADR-031)",
        "error_rate": lab["simulated_error_rate"],
        "labelled": lab["n"],
        "flipped": lab["simulated_flipped"],
        "source": "prediction log (loop_counts), read before cycle 2",
    }
    return stream, review


def record_cycle(
    params: Mapping[str, Any],
    cycle: str,
    gate_file: Path,
    db_counts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """``results/loop/<cycle>/cycle.json`` (+ the export report copy, + the pilot's stream and
    review facts from the log)."""
    fault = cycle == FAULT
    folder = cycle_dir(params, cycle)
    gate = read_json(gate_file)
    version = str(gate["challenger"]["version"])
    retrain_path = retrain_record_for(params, version)
    retrain = read_json(retrain_path)
    if bool(retrain.get("fault_injection")) != fault:
        raise LoopError(f"v{version} fault_injection={retrain.get('fault_injection')}: wrong cycle")
    if fault and gate["applied"]:
        raise LoopError("a fault-injection gate file must be record mode (applied: false)")
    report_src = Path(
        params["loop"]["fault_injection"]["report"]
        if fault
        else params["review"]["export"]["report"]
    )
    saved = folder / "export_report.json"
    if not fault and saved.is_file():  # re-recording: the copy made when the cycle ran
        report_src = saved
    report = read_json(report_src)
    if report["csv_md5"] != retrain["feedback"]["md5"]:
        raise LoopError(f"{report_src} is not the export v{version} was trained on (MD5)")
    folder.mkdir(parents=True, exist_ok=True)
    if cycle == "1" and not (folder / "stream.json").is_file():  # read once, before cycle 2
        if db_counts is None:
            raise LoopError("the pilot record needs the prediction log (DATABASE_URL)")
        stream, review = pilot_facts(db_counts, int(retrain["feedback"]["rows"]))
        write_json(folder / "stream.json", stream)
        write_json(folder / "review.json", review)
    for name in () if fault else ("stream.json", "review.json"):
        if not (folder / name).is_file():
            raise LoopError(f"missing {folder / name}: run the stream / reviewer with --out")
    review = None if fault else read_json(folder / "review.json")
    stream = None if fault else read_json(folder / "stream.json")
    record = {
        "decision_record": DECISION_RECORD,
        "cycle": cycle,
        "fault_injection": fault,
        "note": ("FAULT INJECTION: BROKEN reviewer, never promoted. " if fault else "") + SIMULATED,
        "chunk": None if fault else (stream.get("chunk") or "pilot"),
        "stream": stream,
        "review": review,
        "reviewer_error": (
            {"kind": "broken", "rule": report["rule"], "labels_changed": report["labels_changed"]}
            if fault
            else {
                "kind": "simulated",
                "error_rate": review["error_rate"],
                "flipped": review["flipped"],
            }
        ),
        "new_labels": 0 if fault else review["labelled"],
        "feedback_rows_used": int(retrain["feedback"]["rows"]),
        "feedback_md5": retrain["feedback"]["md5"],
        "export_report": (
            report_src.as_posix() if fault else (folder / "export_report.json").as_posix()
        ),
        "calibration": calibration_of(params, retrain),
        "kaggle_wall_time_s": retrain["kaggle"]["wall_time_s"],
        "kaggle_training_time_s": retrain["kaggle"]["training_time_s"],
        "retrain_record": retrain_path.as_posix(),
        "champion_version": str(gate["champion"]["version"]),
        "challenger_version": version,
        "val_macro_f1": {
            "champion": gate["champion_metrics"]["macro_f1"],
            "challenger": gate["challenger_metrics"]["macro_f1"],
        },
        "decision": gate["decision"],
        "failed_checks": gate["failed_checks"],
        "checks": {k: v["reason"] for k, v in gate["checks"].items()},
        "applied": gate["applied"],
        "gate_file": Path(gate_file).as_posix(),
        "gate_record_file": retrain["gate"]["file"],
        "applied_in": gate_file.as_posix() if gate["applied"] else applied_in(params, version),
        "prediction_log_counts": None if fault else db_counts,
    }
    if not fault and report_src != saved:  # only once every check has passed
        shutil.copyfile(report_src, saved)
    write_json(folder / "cycle.json", record)
    return record


# ---------------------------------------------------------------- summary


def verdict(c: Mapping[str, Any]) -> str:
    v = c["decision"].upper()
    if c["fault_injection"]:
        return f"{v} (record only: fault injection)"
    if c["applied"]:
        return f"{v} (applied)"
    later = c.get("applied_in")
    return f"{v} (record only; applied in {Path(later).name})" if later else f"{v} (record only)"


def table_rows(summary: Mapping[str, Any]) -> list[list[str]]:
    rows = []
    for c in summary["cycles"]:
        err = c["reviewer_error"]
        if err["kind"] == "broken":
            error = f"BROKEN: all {err['labels_changed']} labels shifted"
        else:
            error = f"{err['error_rate']:.0%} ({err['flipped']} flipped)"
        f1 = c["val_macro_f1"]
        cal = c["calibration"]
        if cal["source"] == "inherited_T":
            t = (
                f"inherited_T from v{cal['inherited_from_version']} ({cal['temperature']:.4f}; "
                f"fit on bound {cal['fit_on_bound']:.2f})"
            )
        else:
            t = f"fitted ({cal['temperature']:.4f})"
        rows.append(
            [
                "fault injection" if c["fault_injection"] else str(c["cycle"]),
                f"{c['new_labels']} new / {c['feedback_rows_used']} used",
                error,
                t,
                f"v{c['champion_version']} {f1['champion']:.4f} vs "
                f"v{c['challenger_version']} {f1['challenger']:.4f}",
                verdict(c)
                + (f" — failed: {', '.join(c['failed_checks'])}" if c["failed_checks"] else ""),
                f"`{c['gate_file']}`",
            ]
        )
    return rows


HEADER = [
    "Cycle",
    "Labels",
    "Reviewer error",
    "Challenger T",
    "Val macro-F1 (champion vs challenger)",
    "Verdict",
    "Gate file",
]


def render_table(summary: Mapping[str, Any]) -> str:
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    lines += ["| " + " | ".join(r) + " |" for r in table_rows(summary)]
    return "\n".join(lines) + "\n"


def render_readme(summary: Mapping[str, Any]) -> str:
    parts = [
        "# Loop cycles (TASK-030, ADR-036)\n",
        "Generated by `make loop-summary` from `loop_summary.json`; do not edit by hand.\n",
        "**The stream and the reviewer are SIMULATED.** Cycle 1 is the pilot (all "
        "`production_stream` images through v2). Cycles 2 and 3 stream disjoint chunks of the "
        f"images the pilot did not label: {NOT_NEW}. Each cycle retrains from the base checkpoint "
        "on `train_v1` + all feedback so far. The gate is non-inferiority on `val` (ADR-033); "
        "val has 180 images, so differences of a few images are within noise.\n",
        "**Challenger T:** the temperature fitted on val or, if that fit sat on a range bound "
        "(e.g. no val errors), the champion's T inherited under ADR-037, a rule added after "
        "cycle 2 exposed the gap. An inherited T is not validated for the challenger.\n",
        render_table(summary),
    ]
    final = summary.get("final_test_frozen")
    if final:
        a, b = final["models"]
        d = final["macro_f1_difference"]
        parts.append(
            "\n## Final comparison on `test_frozen` (report only, after the last gate)\n\n"
            "| Model | Macro-F1 [95% CI] | Errors |\n|---|---|---|\n"
            + "".join(
                f"| v{m['version']} ({m['role']}) | {m['macro_f1']['value']:.4f} "
                f"[{m['macro_f1']['ci_low']:.4f}, {m['macro_f1']['ci_high']:.4f}] | "
                f"{m['n_errors']} / {final['n_images']} |\n"
                for m in (a, b)
            )
            + f"\nDifference (v{b['version']} − v{a['version']}): {d['value']:+.4f} "
            f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}], same grouped bootstrap resamples. "
            f"Per-class recall: `{summary['final_file']}`. No decision used this result.\n"
        )
    return "\n".join(parts)


def summarise(params: Mapping[str, Any]) -> dict[str, Any]:
    out = Path(params["loop"]["out_dir"])
    files = sorted(out.glob("cycle_*/cycle.json"), key=lambda p: int(p.parent.name.split("_")[1]))
    if (out / FAULT_DIR / "cycle.json").is_file():
        files.append(out / FAULT_DIR / "cycle.json")
    if not files:
        raise LoopError(f"no cycle records under {out}: run make loop-record first")
    cycles = [read_json(p) for p in files]
    keep = (
        "cycle", "fault_injection", "chunk", "new_labels", "feedback_rows_used", "reviewer_error",
        "champion_version", "challenger_version", "val_macro_f1", "decision", "failed_checks",
        "applied", "applied_in", "gate_file", "gate_record_file", "retrain_record",
        "export_report", "calibration", "kaggle_wall_time_s", "kaggle_training_time_s",
    )  # fmt: skip
    summary: dict[str, Any] = {
        "decision_record": DECISION_RECORD,
        "note": f"{SIMULATED}. Chunks: {NOT_NEW}.",
        "cycles": [
            {k: c[k] for k in keep} | {"file": p.as_posix()}
            for c, p in zip(cycles, files, strict=True)
        ],
        "kaggle_wall_time_s_total": sum(
            c["kaggle_wall_time_s"] for c in cycles if c["cycle"] != "1"
        ),
    }
    final_file = Path(params["loop"]["final_out"])
    if final_file.is_file():
        final = read_json(final_file)
        summary["final_file"] = final_file.as_posix()
        summary["final_test_frozen"] = {
            "n_images": final["n_images"],
            "models": [
                {k: m[k] for k in ("version", "role", "macro_f1", "n_errors", "recall")}
                for m in final["models"]
            ],
            "macro_f1_difference": final["macro_f1_difference"],
        }
    write_json(out / "loop_summary.json", summary)
    (out / "README.md").write_bytes(render_readme(summary).encode())
    return summary


# ---------------------------------------------------------------- final test (report only)


def paired_difference(
    y: Any, pred_a: Any, pred_b: Any, n_classes: int, samples: Sequence[Any], level: float
) -> dict[str, float]:
    """Macro-F1(b) − macro-F1(a), with a percentile interval over the same resamples."""
    import numpy as np

    from src.training.metrics import macro_f1, percentile_interval

    diffs = np.array(
        [
            macro_f1(y[s], pred_b[s], n_classes) - macro_f1(y[s], pred_a[s], n_classes)
            for s in samples
        ]
    )
    lo, hi = percentile_interval(diffs, level)
    value = macro_f1(y, pred_b, n_classes) - macro_f1(y, pred_a, n_classes)
    return {"value": float(value), "ci_low": float(lo), "ci_high": float(hi)}


def final_versions(params: Mapping[str, Any]) -> tuple[str, str]:
    log = read_json(params["promotion"]["log_file"])
    initial = [e for e in log if e["decision"] == "initial_champion"]
    if len(initial) != 1:
        raise LoopError("the promotion log needs exactly one initial_champion entry")
    from src.training import export_onnx as ex

    final = ex.registered_version(params, str(params["promotion"]["champion_alias"]))["version"]
    if str(final) == str(initial[0]["version"]):
        raise LoopError(f"@champion is still v{final}: nothing to compare")
    return str(initial[0]["version"]), str(final)


def final_test(params: Mapping[str, Any]) -> dict[str, Any]:
    """Once, after the last gate: v2 vs the final @champion on test_frozen. Report only."""
    import numpy as np
    from torch.utils.data import DataLoader

    from src.serving.preprocess import PreprocessConfig
    from src.tracking import mlflow_tracking as mt
    from src.tracking.gate import version_checkpoint
    from src.training import evaluate as ev
    from src.training.compare_val import load_checkpoint, paired_table
    from src.training.dataset import PROTECTED_SPLIT, SplitDataset
    from src.training.metrics import grouped_bootstrap_indices
    from src.training.train import sha256_file

    out = Path(params["loop"]["final_out"])
    if out.exists():
        raise LoopError(f"{out} exists: the final test runs once (ADR-036)")
    ev.check_split(params)
    first, final = final_versions(params)
    cfg = ev.EvalConfig.from_params(params)
    tracking = mt.TrackingConfig.from_params(params, smoke=False)
    split_csv, raw = Path(params["data"]["split_file"]), Path(params["data"]["raw_dir"])
    pre = PreprocessConfig.from_params(params)
    preds, y, ds, classes = {}, None, None, None
    for version in (first, final):
        ckpt = version_checkpoint(params, version)
        tags = mt.version_tags(tracking, version)
        if not ckpt.is_file() or sha256_file(ckpt) != tags.get("checkpoint_sha256"):
            raise LoopError(f"{ckpt} is not the checkpoint of v{version}")
        model, model_classes, _ = load_checkpoint(ckpt)
        if classes is not None and model_classes != classes:
            raise LoopError("the two versions use different class orders")
        classes = model_classes
        # Report only (ADR-036): the second place outside evaluate.py and the gate that reads it.
        ds = SplitDataset(split_csv, raw, [PROTECTED_SPLIT], classes, pre, allow_test=True)
        y_v, _, probs = ev.predict_outputs(model, DataLoader(ds, batch_size=cfg.batch_size))
        if y is not None and not np.array_equal(y, y_v):
            raise LoopError("the two versions did not see the same images")
        y, preds[version] = y_v, probs.argmax(axis=1)
    assert ds is not None and classes is not None and y is not None
    groups = ev.near_copy_groups_within(ds.paths, cfg)
    units = ev.unit_ids(ds.paths, groups)
    samples = grouped_bootstrap_indices(y, units, cfg.n_resamples, cfg.seed)
    models = []
    for version, role in ((first, "original champion"), (final, "final champion")):
        m = ev.metrics_report(y, preds[version], classes, samples, cfg.level)
        models.append(
            {
                "version": version,
                "role": role,
                "macro_f1": m["macro_f1"],
                "accuracy": m["accuracy"],
                "n_errors": m["n_errors"],
                "recall": {c: m["per_class"][c]["recall"] for c in classes},
                "confusion_matrix": m["confusion_matrix"],
            }
        )
    table = paired_table(y, preds[first], preds[final])
    report = {
        "decision_record": DECISION_RECORD,
        "REPORT_ONLY": "run once after the last gate; no decision, threshold or cycle uses it",
        "note": SIMULATED,
        "split": PROTECTED_SPLIT,
        "split_file": split_csv.name,
        "split_sha256": sha256_file(split_csv),
        "n_images": int(len(y)),
        "classes": list(classes),
        "models": models,
        "macro_f1_difference": {
            "of": f"v{final} - v{first}",
            **paired_difference(y, preds[first], preds[final], len(classes), samples, cfg.level),
        },
        "paired": {
            "both_right": table.both_right,
            f"only_v{first}_right": table.a_only,
            f"only_v{final}_right": table.b_only,
            "both_wrong": table.both_wrong,
        },
        "bootstrap": {
            "method": "cluster bootstrap stratified by class, percentile interval (ADR-022)",
            "unit": "confirmed near-copy group (ADR-021) or single image",
            "n_units": int(len(np.unique(units))),
            "n_resamples": cfg.n_resamples,
            "seed": cfg.seed,
            "level": cfg.level,
        },
        "time_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_json(out, report)
    return report


# ---------------------------------------------------------------- CLI


def db_counts() -> dict[str, Any]:
    from src.db.store import PredictionLog, database_url_from_env

    db = PredictionLog(database_url_from_env())
    try:
        return db.loop_counts()
    finally:
        db.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/params.yaml"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chunks")
    sub.add_parser("fault-export")
    rec = sub.add_parser("record")
    rec.add_argument("--cycle", required=True, help="1, 2, 3, ... or 'fault'")
    rec.add_argument("--gate-file", type=Path, required=True)
    sub.add_parser("final")
    sub.add_parser("summary")
    args = parser.parse_args(argv)
    params = load_params(args.config)
    try:
        if args.cmd == "chunks":
            print("loop chunks:", json.dumps(make_chunks(params)), f"({NOT_NEW})")
        elif args.cmd == "fault-export":
            r = fault_export(params)
            print(f"FAULT INJECTION export: {r['rows']} rows, all labels shifted -> {r['csv']}")
        elif args.cmd == "record":
            counts = None if args.cycle == FAULT else db_counts()
            r = record_cycle(params, args.cycle, args.gate_file, counts)
            folder = cycle_dir(params, args.cycle).as_posix()
            print(f"cycle {r['cycle']}: {r['decision'].upper()} -> {folder}")
        elif args.cmd == "final":
            r = final_test(params)
            for m in r["models"]:
                f1 = m["macro_f1"]
                print(
                    f"v{m['version']} ({m['role']}): macro-F1 {f1['value']:.4f} "
                    f"[{f1['ci_low']:.4f}, {f1['ci_high']:.4f}], {m['n_errors']} errors"
                )
            print("REPORT ONLY (ADR-036):", params["loop"]["final_out"])
        else:
            s = summarise(params)
            print(render_table(s))
    except LoopError as exc:
        print(f"loop: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
